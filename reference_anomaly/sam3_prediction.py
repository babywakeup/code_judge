"""Local SAM3 box / anomaly-point prediction, with a model-free prompt preview.

GT is never accessed here. Existing graph-cut outputs are not read or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .box_graphcut import _bgr, read_boxes
from .box_prediction import _artifact, _document, _read, _selected_paths, _sha, _write
from .sam3_backend import OfficialSam3Backend, choose_backend, environment_report, resolve_checkpoint
from .sam3_prompts import CONFIG, plan_prompts
from .sam3_transformers import TransformersSam3Backend, bundle_sources, resolve_transformers_checkpoint


def select_candidate(masks, quality, shape, points_xy, point_labels):
    """Require every prompt to be respected; choose max predicted quality.

    No GT, graph-cut overlap, heatmap intersection, largest-component filtering
    or contour smoothing. Return explicit abstention if every mask contradicts
    a prompt. SAM's quality score is not a measured IoU with ground truth.
    """
    masks, quality = np.asarray(masks), np.asarray(quality)
    if (masks.ndim != 3 or masks.shape[1:] != shape or quality.shape != (len(masks),)
            or masks.dtype.kind not in 'buif' or not np.isfinite(masks).all()
            or not np.isin(masks, [0, 1]).all() or not np.isfinite(quality).all()):
        raise ValueError('SAM3 must return finite binary CxHxW masks and C quality scores at crop resolution')
    masks = masks.astype(bool)
    records, accepted = [], []
    for index, (mask, score) in enumerate(zip(masks, quality)):
        agrees = all(bool(mask[y, x]) == bool(label) for (x, y), label in zip(points_xy, point_labels))
        usable = agrees and bool(mask.any())
        records.append(dict(index=index, predicted_quality=float(score), prompt_consistent=agrees,
                            nonempty=bool(mask.any()), pixels=int(mask.sum())))
        if usable:
            accepted.append(index)
    selected = max(accepted, key=lambda i: (float(quality[i]), -i)) if accepted else None
    return (masks[selected] if selected is not None else None), selected, records


def _sources(result_path, config_path, target_path, boxes_path, mode):
    target, xml = _selected_paths(config_path, target_path, boxes_path)
    manifest = Path(result_path).resolve()
    if manifest.is_dir():
        manifest /= 'result.json'
    metadata = _document(manifest)
    if metadata.get('format_version') != 1 or metadata.get('accepted') is not True:
        raise ValueError('SAM3 experiment requires an accepted saved matcher result')
    paths = dict(target=target, boxes=xml, matcher_manifest=manifest,
                 input=_artifact(manifest, metadata, 'input'))
    # Reference validity is for reporting only, NEVER a hard SAM output mask.
    anomaly = manifest.parent / 'anomaly' / 'anomaly.json'
    evidence = _document(anomaly)
    origin = evidence.get('source', {}).get('manifest')
    if not isinstance(origin, str) or Path(origin).resolve() != manifest:
        raise ValueError('Anomaly evidence does not belong to this matcher result')
    paths.update(anomaly_manifest=anomaly, valid=_artifact(anomaly, evidence, 'valid'))
    if mode == 'anomaly_points':
        paths.update(scores=_artifact(anomaly, evidence, 'scores'),
                     confidence=_artifact(anomaly, evidence, 'confidence'))
    return paths, anomaly.parent


def run_prediction(result_path: str | Path, *, checkpoint: str | Path = 'weights/sam3.pt',
                   config_path: str | Path | None = None, target_path: str | Path | None = None,
                   boxes_path: str | Path | None = None, output_dir: str | Path | None = None,
                   mode: str = 'anomaly_points', prepare_only: bool = False, device: str = 'auto',
                   coordinates: str = 'voc_1based_inclusive', backend=None,
                   backend_type: str = 'auto', text: str | None = None,
                   threshold: float = .5, mask_threshold: float = .5) -> Path:
    """Freeze prompts or predictions to a new directory, without any GT argument.

    An injected backend supports synthetic contract tests, not surrogate model
    claims. Actual runs always record backend identity and checkpoint hashes.
    """
    if mode not in ('box', 'anomaly_points'):
        raise ValueError('mode must be box or anomaly_points')
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (threshold, mask_threshold)):
        raise ValueError('Model confidence and mask thresholds must be in [0,1]')
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError('text must be a nonempty string when supplied')
    selected_backend = choose_backend(checkpoint, backend_type)
    if selected_backend == 'transformers':
        model_dir = resolve_transformers_checkpoint(checkpoint)
        weight = model_dir / 'model.safetensors'
    else:
        model_dir = None
        weight = resolve_checkpoint(checkpoint)
    if text is not None and (mode != 'box' or selected_backend != 'transformers'):
        raise ValueError('Text prompts require --backend transformers --mode box')
    paths, anomaly_dir = _sources(result_path, config_path, target_path, boxes_path, mode)
    paths['checkpoint'] = weight
    if model_dir is not None:
        paths.update(bundle_sources(model_dir))
    stage = '_prepared' if prepare_only else ''
    suffix = '_transformers' if selected_backend == 'transformers' else ''
    output = Path(output_dir).resolve() if output_dir is not None else anomaly_dir / f'sam3_{mode}{suffix}{stage}'
    protected = set(paths.values())
    if config_path is not None:
        protected.add(Path(config_path).resolve())
    if output.exists() or output.is_symlink() or any(p.is_relative_to(output) for p in protected):
        raise ValueError('SAM3 output must be a new directory without source overlap')
    source = {key: dict(path=str(path), sha256=_sha(path)) for key, path in paths.items()}
    image = _bgr(_read(paths['input']))
    if not np.array_equal(image, _bgr(_read(paths['target']))):
        raise ValueError('Original target does not match saved matcher input')
    raw_valid = _read(paths['valid'])
    if (raw_valid.shape != image.shape[:2] or raw_valid.dtype != np.uint8
            or not np.isin(raw_valid, [0, 255]).all()):
        raise ValueError('Comparison validity must be input-sized uint8 0/255')
    valid, response, confidence = raw_valid != 0, None, None
    if mode == 'anomaly_points':
        confidence = _read(paths['confidence'])
        with np.load(paths['scores'], allow_pickle=False) as scores:
            if 'salience_score' not in scores:
                raise ValueError('Missing salience_score; regenerate anomaly evidence')
            response = scores['salience_score']
    boxes = read_boxes(paths['boxes'], image.shape[:2], coordinates=coordinates)
    plan = plan_prompts(image.shape[:2], boxes, mode=mode, score=response, valid=valid, confidence=confidence)
    if not prepare_only and backend is None and any(r['status'] == 'ok' for r in plan):
        if selected_backend == 'transformers':
            backend = TransformersSam3Backend(model_dir, device=device,
                task='image' if mode == 'box' else 'tracker', text=text,
                threshold=threshold, mask_threshold=mask_threshold)
        else:
            backend = OfficialSam3Backend(weight, device=device)
    preview = image.copy()
    seeds = np.zeros_like(image)
    mask = np.zeros(image.shape[:2], bool)
    raw_candidates = {}
    model_executed = False
    for record in plan:
        if record['status'] != 'ok':
            continue
        x0, y0, x1, y1 = record['roi_xyxy']
        cv2.rectangle(preview, (x0, y0), (x1-1, y1-1), (255, 160, 0), 1)
        crop = image[y0:y1, x0:x1]
        if not prepare_only:
            backend.set_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            model_executed = True
        ok_tasks = 0
        for task_index, task in enumerate(record['prompts']):
            bx0, by0, bx1, by1 = task['box_xyxy']
            cv2.rectangle(preview, (x0+bx0, y0+by0), (x0+bx1-1, y0+by1-1), (0, 200, 255), 1)
            for (x, y), label in zip(task['points_xy'], task['point_labels']):
                color = (0, 255, 0) if label else (255, 0, 0)
                cv2.circle(preview, (x0+x, y0+y), 3, color, -1)
                seeds[y0+y, x0+x] = color
            if prepare_only:
                continue
            masks, quality = backend.predict(box_xyxy=task['box_xyxy'], points_xy=task['points_xy'],
                                             point_labels=task['point_labels'])
            selected_mask, selected, candidates = select_candidate(
                masks, quality, crop.shape[:2], task['points_xy'], task['point_labels'])
            if getattr(backend, 'candidate_semantics', 'alternatives') == 'instances':
                # Sam3Model outputs distinct instances, not competing hypotheses.
                # Retain ALL nonempty instances already confidence-filtered by HF.
                selected = [c['index'] for c in candidates if c['nonempty'] and c['prompt_consistent']]
                selected_mask = np.asarray(masks, bool)[selected].any(axis=0) if selected else None
            key = f'box_{record["index"]}_task_{task_index}'
            raw_candidates[key] = np.asarray(masks, np.uint8)
            task.update(selected_candidate=selected, candidates=candidates, candidate_archive_key=key)
            if selected_mask is None:
                task['status'] = 'no_prompt_consistent_candidate'
                continue
            task['status'] = 'ok'
            ok_tasks += 1
            mask[y0:y1, x0:x1] |= selected_mask
            if selected_mask[0].any() or selected_mask[-1].any() or selected_mask[:, 0].any() or selected_mask[:, -1].any():
                if 'touches_roi' not in record['warnings']:
                    record['warnings'].append('touches_roi')
        if not prepare_only:
            record['status'] = 'ok' if ok_tasks == len(record['prompts']) else ('partial' if ok_tasks else 'abstained')
    # Protect evidence, model checkpoint and inputs against concurrent writes.
    for key, path in paths.items():
        if _sha(path) != source[key]['sha256']:
            raise ValueError(f'Source changed during SAM3 processing: {key}')
    good = sum(r['status'] == 'ok' for r in plan)
    status = ('ok' if good == len(plan) else 'partial') if good else ('no_boxes' if not plan else 'abstained')
    document = dict(format_version=1, kind='sam3_prompt_preparation' if prepare_only else 'sam3_prediction',
                    image_shape=list(image.shape[:2]), status=status, boxes=plan, source=source,
                    config=dict(CONFIG, mode=mode, xml_coordinates=coordinates,
                                backend=selected_backend, text=text, confidence_threshold=threshold,
                                mask_threshold=mask_threshold,
                                selection=('all_instances_above_model_threshold' if selected_backend == 'transformers' and mode == 'box'
                                           else 'all_prompt_consistent_then_max_sam_quality'),
                                reference_validity_role='guidance_and_coverage_only_not_output_clipping'),
                    prediction_uses_gt=False, model_executed=model_executed,
                    algorithm_sha256=_sha(Path(__file__)),
                    planner_sha256=_sha(Path(__file__).with_name('sam3_prompts.py')),
                    adapter_sha256=_sha(Path(__file__).with_name('sam3_backend.py')))
    if selected_backend == 'transformers':
        document['transformers_adapter_sha256'] = _sha(Path(__file__).with_name('sam3_transformers.py'))
    artifacts = dict(input='input.png', valid='comparison_valid.png', prompts='prompts.json',
                     seeds='seeds.png', preview='prompt_preview.png')
    if not prepare_only:
        document.update(predicted_pixels=int(mask.sum()),
                        predicted_without_reference_pixels=int((mask & ~valid).sum()),
                        backend=getattr(backend, 'provenance', {'backend': type(backend).__name__}))
        artifacts.update(mask='prediction_mask.png', contours='contours.json',
                         overlay='prediction_overlay.png', candidates='candidates.npz')
    output.mkdir(parents=True, exist_ok=False)
    for key, array in dict(input=image, valid=raw_valid, seeds=seeds, preview=preview).items():
        _write(output / artifacts[key], array)
    with (output / artifacts['prompts']).open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if not prepare_only:
        _write(output / artifacts['mask'], mask.astype(np.uint8)*255)
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        overlay = image.copy()
        overlay[mask] = np.rint(.55*image[mask]+.45*np.array([0, 0, 255])).astype(np.uint8)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        _write(output / artifacts['overlay'], overlay)
        with (output / artifacts['contours']).open('x', encoding='utf-8') as stream:
            json.dump(dict(authoritative_artifact=artifacts['mask'], contours=[
                dict(points=c[:, 0].tolist(), hierarchy=hierarchy[0, i].tolist())
                for i, c in enumerate(contours)]), stream, allow_nan=False)
        np.savez_compressed(output / artifacts['candidates'], **raw_candidates)
    document.update(artifacts=artifacts, artifact_sha256={k: _sha(output / v) for k, v in artifacts.items()})
    final = output / ('preparation.json' if prepare_only else 'prediction.json')
    with final.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--config', type=Path, help='Extract target/box only; GT is ignored')
    parser.add_argument('--target', type=Path)
    parser.add_argument('--boxes', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/sam3.pt'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mode', choices=['box', 'anomaly_points'], default='anomaly_points')
    parser.add_argument('--backend', choices=['auto', 'transformers', 'official'], default='auto')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--text', help='Optional text combined with detector boxes (Transformers box mode)')
    parser.add_argument('--threshold', type=float, default=.5, help='Transformers image instance confidence threshold')
    parser.add_argument('--mask-threshold', type=float, default=.5, help='Transformers image mask threshold')
    parser.add_argument('--xml-coordinates', choices=['voc_1based_inclusive', 'zero_half_open'], default='voc_1based_inclusive')
    parser.add_argument('--prepare-only', action='store_true', help='Save prompt preview, not a model prediction')
    parser.add_argument('--check', action='store_true', help='Report local checkpoint and runtime readiness')
    args = parser.parse_args(argv)
    try:
        if args.check:
            print(json.dumps(environment_report(args.checkpoint, args.backend), ensure_ascii=False, indent=2))
            return 0
        if args.result is None:
            parser.error('--result is required unless --check is used')
        saved = run_prediction(args.result, checkpoint=args.checkpoint, config_path=args.config,
                               target_path=args.target, boxes_path=args.boxes, output_dir=args.out,
                               mode=args.mode, prepare_only=args.prepare_only, device=args.device,
                               coordinates=args.xml_coordinates, backend_type=args.backend,
                               text=args.text, threshold=args.threshold, mask_threshold=args.mask_threshold)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        if getattr(error, 'winerror', None) == 1455 or '1455' in str(error):
            parser.exit(2, 'SAM3 CPU/local model loading failed: Windows memory/pagefile commit limit (1455). '
                        'Close memory-heavy programs or increase virtual memory, then retry. CUDA is not required.\n')
        parser.exit(2, f'SAM3 processing failed: {error}\n')
    print(f'{"Prompt preparation (no model inference)" if args.prepare_only else "Frozen SAM3 prediction"}: {saved}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())