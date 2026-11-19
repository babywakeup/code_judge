"""Freeze a detector-box graph-cut prediction. This module never reads GT files.

Run: python -m reference_anomaly.box_prediction --result <matcher-dir>
     --config <matcher-config>   (only target and box fields are extracted)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import scipy

from .box_graphcut import BoxGraphCutConfig, _bgr, predict_boxes, read_boxes


ARTIFACTS = dict(mask='prediction_mask.png', valid='comparison_valid.png', input='input.png',
                 overlay='prediction_overlay.png', seeds='seeds.png', contours='contours.json')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path.name}')
    return value


def _artifact(manifest: Path, metadata: dict, key: str) -> Path:
    name = metadata.get('artifacts', {}).get(key)
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f'Missing or invalid artifact: {key}')
    path = (manifest.parent / name).resolve()
    if not path.is_relative_to(manifest.parent) or path == manifest:
        raise ValueError(f'Artifact escapes its directory: {key}')
    return path


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot decode image: {path.name}')
    return image


def _write(path: Path, array: np.ndarray):
    ok, encoded = cv2.imencode('.png', array)
    if not ok:
        raise ValueError(f'Cannot encode image: {path.name}')
    with path.open('xb') as stream:
        stream.write(encoded.tobytes())


def _selected_paths(config_path, target_path, boxes_path):
    if config_path is not None:
        config_path = Path(config_path).resolve()
        # Deliberately do not use matcher config loader: it can open annotations.
        document = _document(config_path)
        selected = {key: document.get(key) for key in ('target', 'box')}
        for key in selected:
            if selected[key] is not None and not isinstance(selected[key], str):
                raise ValueError(f'Config {key} must be a path string')
        if target_path is None and selected['target']:
            target_path = config_path.parent / selected['target']
        if boxes_path is None and selected['box']:
            boxes_path = config_path.parent / selected['box']
    if target_path is None or boxes_path is None:
        raise ValueError('Provide --config with target/box or both --target and --boxes')
    return Path(target_path).resolve(), Path(boxes_path).resolve()


def run_prediction(result_path: str | Path, *, config_path: str | Path | None = None,
                   target_path: str | Path | None = None, boxes_path: str | Path | None = None,
                   anomaly_path: str | Path | None = None, output_dir: str | Path | None = None,
                   coordinates: str = 'voc_1based_inclusive', score_key: str = 'salience_score',
                   config: BoxGraphCutConfig | None = None) -> Path:
    """Consume existing evidence and save to a NEW directory; no overwrite mode.

    The matcher input must equal the decoded original target. Source bytes are
    hashed before computation and checked again before saving. Only selected
    config target/box paths are retained: GT_labelme is never resolved or opened.
    """
    cfg = config or BoxGraphCutConfig()
    cfg.validate()
    manifest = Path(result_path).resolve()
    if manifest.is_dir():
        manifest /= 'result.json'
    target, xml = _selected_paths(config_path, target_path, boxes_path)
    anomaly = Path(anomaly_path).resolve() if anomaly_path is not None else manifest.parent / 'anomaly' / 'anomaly.json'
    if anomaly.is_dir():
        anomaly /= 'anomaly.json'
    metadata, evidence = _document(manifest), _document(anomaly)
    if metadata.get('format_version') != 1 or metadata.get('accepted') is not True:
        raise ValueError('Requires an accepted matcher result with normal-reference artifacts')
    provenance = evidence.get('source', {})
    if not isinstance(provenance, dict) or not isinstance(provenance.get('manifest'), str):
        raise ValueError('Anomaly evidence must identify its source matcher manifest')
    if Path(provenance['manifest']).resolve() != manifest:
        raise ValueError('Anomaly evidence does not belong to this matcher result')
    if score_key not in ('salience_score', 'supported_salience_score'):
        raise ValueError('Score must be salience_score or supported_salience_score')
    paths = dict(target=target, boxes=xml, matcher_manifest=manifest, anomaly_manifest=anomaly,
                 input=_artifact(manifest, metadata, 'input'),
                 reference=_artifact(manifest, metadata, 'supposed_to_be_good'),
                 scores=_artifact(anomaly, evidence, 'scores'),
                 valid=_artifact(anomaly, evidence, 'valid'),
                 confidence=_artifact(anomaly, evidence, 'confidence'))
    output = (Path(output_dir).resolve() if output_dir is not None else anomaly.parent / 'box_segmentation')
    protected = set(paths.values())
    if config_path is not None:
        protected.add(Path(config_path).resolve())
    if output.exists() or output.is_symlink() or any(p.is_relative_to(output) for p in protected):
        raise ValueError('Prediction output must be a new directory, without source overlap')
    source = {key: dict(path=str(path), sha256=_sha(path)) for key, path in paths.items()}
    image, normal = _bgr(_read(paths['input'])), _bgr(_read(paths['reference']))
    if not np.array_equal(_bgr(_read(target)), image):
        raise ValueError('Original target does not match the saved matcher input')
    raw_valid, confidence = _read(paths['valid']), _read(paths['confidence'])
    if raw_valid.dtype != np.uint8 or not np.isin(raw_valid, [0, 255]).all():
        raise ValueError('Comparison validity must be a uint8 0/255 mask')
    with np.load(paths['scores'], allow_pickle=False) as scores:
        if score_key not in scores:
            raise ValueError(f'Missing {score_key}; regenerate anomaly evidence')
        score = scores[score_key]
    boxes = read_boxes(xml, image.shape[:2], coordinates=coordinates)
    prediction = predict_boxes(image, normal, raw_valid != 0, score, boxes, confidence=confidence, config=cfg)
    for key, path in paths.items():
        if _sha(path) != source[key]['sha256']:
            raise ValueError(f'Source changed during prediction: {key}')
    contours, hierarchy = cv2.findContours(prediction.mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_document = dict(coordinates='zero_based_pixel_centers', authoritative_artifact=ARTIFACTS['mask'],
                            contours=[dict(points=c[:, 0].tolist(), hierarchy=hierarchy[0, i].tolist())
                                      for i, c in enumerate(contours)])
    overlay = image.copy()
    overlay[prediction.mask] = np.rint(.55*image[prediction.mask] + .45*np.array([0, 0, 255])).astype(np.uint8)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    for box in boxes:
        cv2.rectangle(overlay, (max(0, box.x0), max(0, box.y0)),
                      (min(image.shape[1]-1, box.x1-1), min(image.shape[0]-1, box.y1-1)), (255, 160, 0), 1)
    seeds = np.zeros_like(image)
    seeds[~prediction.valid] = (48, 48, 48)
    seeds[prediction.background_seeds] = (255, 0, 0)
    seeds[prediction.foreground_seeds] = (0, 255, 0)
    document = dict(format_version=1, kind='box_graphcut_prediction', status=prediction.status,
                    image_shape=list(image.shape[:2]), predicted_pixels=int(prediction.mask.sum()),
                    comparable_pixels=int(prediction.valid.sum()), boxes=prediction.boxes,
                    config=dict(prediction.config, score_key=score_key, xml_coordinates=coordinates,
                                integer_capacity_scale=1024, neighbors=4), source=source,
                    prediction_uses_gt=False, artifacts=ARTIFACTS,
                    software=dict(numpy=np.__version__, opencv=cv2.__version__, scipy=scipy.__version__),
                    algorithm_sha256=_sha(Path(__file__).with_name('box_graphcut.py')),
                    exporter_sha256=_sha(Path(__file__)))
    output.mkdir(parents=True, exist_ok=False)
    for key, array in dict(mask=prediction.mask.astype(np.uint8)*255,
                           valid=prediction.valid.astype(np.uint8)*255, input=image,
                           overlay=overlay, seeds=seeds).items():
        _write(output / ARTIFACTS[key], array)
    with (output / ARTIFACTS['contours']).open('x', encoding='utf-8') as stream:
        json.dump(contour_document, stream, ensure_ascii=False, allow_nan=False)
    document['artifact_sha256'] = {key: _sha(output / name) for key, name in ARTIFACTS.items()}
    # Commit marker written last; an incomplete export is not a frozen prediction.
    prediction_manifest = output / 'prediction.json'
    with prediction_manifest.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return prediction_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Extract target/box only; GT fields ignored')
    parser.add_argument('--target', type=Path)
    parser.add_argument('--boxes', type=Path)
    parser.add_argument('--anomaly', type=Path, help='Existing anomaly directory or anomaly.json')
    parser.add_argument('--out', type=Path, help='New directory; existing directories are never overwritten')
    parser.add_argument('--xml-coordinates', choices=['voc_1based_inclusive', 'zero_half_open'], default='voc_1based_inclusive')
    parser.add_argument('--score', choices=['salience_score', 'supported_salience_score'], default='salience_score')
    args = parser.parse_args(argv)
    try:
        saved = run_prediction(args.result, config_path=args.config, target_path=args.target,
                               boxes_path=args.boxes, anomaly_path=args.anomaly, output_dir=args.out,
                               coordinates=args.xml_coordinates, score_key=args.score)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(2, f'Prediction failed: {error}\n')
    print(f'Frozen prediction: {saved}')
    metadata = _document(saved)
    print(json.dumps({key: metadata[key] for key in ('status', 'predicted_pixels', 'boxes')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())