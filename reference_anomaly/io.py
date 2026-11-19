"""Read matcher references and export reviewable evidence, without rerunning matching."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from periodic_template_matcher.utils import read_image, validate_image, write_image
from .core import AnomalyConfig, AnomalyResult, analyze_pair


ARTIFACTS = dict(scores='anomaly_scores.npz', heatmap='anomaly_heatmap.png',
                 tolerant_heatmap='anomaly_tolerant_heatmap.png', overlay='anomaly_overlay.png',
                 candidates='anomaly_candidates.png', priority='anomaly_priority.png',
                 valid='anomaly_valid.png', confidence='anomaly_confidence.png',
                 color='anomaly_color.png', structure='anomaly_structure.png',
                 multiscale='anomaly_multiscale.png', overview='anomaly_overview.jpg',
                 salience_heatmap='anomaly_salience_heatmap.png', salience_overlay='anomaly_salience_overlay.png',
                 salience_candidates='anomaly_salience_candidates.png', salience_overview='anomaly_salience_overview.jpg',
                 ssim_heatmap='anomaly_ssim_heatmap.png', ssim_overlay='anomaly_ssim_overlay.png',
                 ssim_valid='anomaly_ssim_valid.png', normal_heatmap='anomaly_normal_heatmap.png',
                 normal_overlay='anomaly_normal_overlay.png', normal_valid='anomaly_normal_valid.png',
                 methods_overview='anomaly_methods_overview.jpg',
                 supported_heatmap='anomaly_supported_salience_heatmap.png',
                 supported_overlay='anomaly_supported_salience_overlay.png')


def _bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[..., :3]


def _heatmap(score: np.ndarray, valid: np.ndarray, maximum: float = 3.) -> np.ndarray:
    # Fixed nonlinear display scale; raw float scores remain unchanged in NPZ.
    heat = cv2.applyColorMap(np.rint(np.clip(score/maximum, 0, 1)**1.4*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~valid] = (48, 48, 48)
    return heat


def _regions(result: AnomalyResult, limit: int = 200) -> tuple:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result.candidate_mask.astype(np.uint8))
    peaks = np.zeros(count, np.float32)
    np.maximum.at(peaks, labels.ravel(), result.strict_score.ravel())
    areas = stats[:, cv2.CC_STAT_AREA]
    order = sorted(range(1, count), key=lambda i: (-float(peaks[i]), -int(areas[i]), i))[:limit]
    regions = []
    for index in order:
        x, y, w, h, area = (int(v) for v in stats[index])
        roi = np.s_[y:y+h, x:x+w]
        mask = labels[roi] == index
        score = result.strict_score[roi]
        yy, xx = np.unravel_index(np.argmax(np.where(mask, score, -1)), mask.shape)
        regions.append(dict(component_id=int(index), area=area, peak_xy=[int(x+xx), int(y+yy)],
                            peak_score=float(peaks[index]),
                            priority_fraction=float(result.priority_mask[roi][mask].mean()),
                            high_confidence_fraction=float((result.confidence[roi][mask] == 255).mean())))
    return regions, count-1


def save_analysis(result: AnomalyResult, input_image: np.ndarray, reference: np.ndarray,
                  output_dir: str | Path, *, source_metadata: dict | None = None,
                  protected_paths: tuple = ()) -> Path:
    """Save full-resolution scores and masks; validate before replacing owned outputs."""
    validate_image(input_image)
    validate_image(reference)
    shape = input_image.shape[:2]
    if reference.shape[:2] != shape:
        raise ValueError('Reference dimensions must match input')
    arrays = dict(strict_score=result.strict_score, tolerant_score=result.tolerant_score,
                  color_response=result.color_response, structure_response=result.structure_response,
                  multiscale_response=result.multiscale_response,
                  salience_score=result.salience_score, salience_weight=result.salience_weight,
                  ssim_response=result.ssim_response, ssim_score=result.ssim_score,
                  normal_score=result.normal_score,
                  supported_salience_score=result.supported_salience_score,
                  **{f'ssim_response_{size}': response for size, response in result.ssim_windows.items()})
    for name, array in arrays.items():
        if (array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all()
                or (array < 0).any()):
            raise ValueError(f'{name} must be a finite nonnegative float32 input-sized array')
    masks = (result.comparison_valid, result.candidate_mask, result.priority_mask, result.salience_mask,
             result.ssim_valid, result.normal_valid)
    if any(m.shape != shape or m.dtype != bool for m in masks):
        raise ValueError('Analysis masks must be input-sized bool arrays')
    if (result.confidence.shape != shape or result.confidence.dtype != np.uint8
            or not np.isin(result.confidence, [0, 128, 255]).all()
            or np.any((result.confidence > 0) != result.comparison_valid)
            or np.any(result.candidate_mask & ~result.comparison_valid)
            or np.any(result.priority_mask & ~result.candidate_mask)
            or np.any(result.salience_mask & ~result.candidate_mask)
            or np.any(result.salience_weight > 1)
            or np.any(result.ssim_valid & ~result.comparison_valid)
            or np.any(result.normal_valid & ~result.comparison_valid)
            or np.any(result.ssim_score[~result.ssim_valid] != 0)
            or np.any(result.normal_score[~result.normal_valid] != 0)
            or np.any(result.supported_salience_score < result.salience_score-1e-6)
            or np.any(result.supported_salience_score > 2*result.salience_score+1e-6)
            or np.any(result.tolerant_score > result.strict_score+1e-6)):
        raise ValueError('Inconsistent analysis masks, confidence or scores')
    out = Path(output_dir).resolve()
    names = [*ARTIFACTS.values(), 'anomaly.json']
    protected = {Path(p).resolve() for p in protected_paths}
    if any((out / name).resolve() in protected for name in names):
        raise ValueError('Anomaly outputs must not overwrite source artifacts')
    regions, total = _regions(result)
    metadata = dict(result.metadata, artifacts=dict(ARTIFACTS), regions=regions,
                    region_count=total, regions_truncated=total > len(regions),
                    visualization=dict(score_range=[0, 3], display_gamma=1.4, candidate_seed_threshold=result.metadata['config']['high_threshold'],
                                       invalid_heatmap_color_bgr=[48, 48, 48], confidence_values=[0, 128, 255],
                                       supported_comparison_score_range=[0, 6]))
    if source_metadata is not None:
        metadata['source'] = source_metadata
    document = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    heat = _heatmap(result.strict_score, result.comparison_valid)
    tolerant_heat = _heatmap(result.tolerant_score, result.comparison_valid)
    salience_heat = _heatmap(result.salience_score, result.comparison_valid)
    image = _bgr(input_image)
    low = result.metadata['config']['low_threshold']
    alpha = (.65*np.clip((result.strict_score-low)/max(1., 3-low), 0, 1)*result.comparison_valid)[..., None]
    overlay = np.rint(image*(1-alpha)+heat*alpha).astype(np.uint8)
    salience_alpha = (.65*np.clip((result.salience_score-1.5)/1.5, 0, 1)*result.comparison_valid)[..., None]
    salience_overlay = np.rint(image*(1-salience_alpha)+salience_heat*salience_alpha).astype(np.uint8)
    ssim_heat = _heatmap(result.ssim_score, result.ssim_valid)
    normal_heat = _heatmap(result.normal_score, result.normal_valid)
    def overlay_score(score, heatmap, known):
        alpha = (.65*np.clip((score-1.5)/1.5, 0, 1)*known)[..., None]
        return np.rint(image*(1-alpha)+heatmap*alpha).astype(np.uint8)
    ssim_overlay = overlay_score(result.ssim_score, ssim_heat, result.ssim_valid)
    normal_overlay = overlay_score(result.normal_score, normal_heat, result.normal_valid)
    supported_heat = _heatmap(result.supported_salience_score, result.comparison_valid, 6.)
    salience_comparison_heat = _heatmap(result.salience_score, result.comparison_valid, 6.)
    supported_overlay = overlay_score(result.supported_salience_score/2, supported_heat, result.comparison_valid)
    panels = []
    for title, panel in (('INPUT', image), ('NORMAL REFERENCE', _bgr(reference)),
                         ('STRICT EVIDENCE  (0 - 3)', heat), ('TOLERANT EVIDENCE  (0 - 3)', tolerant_heat),
                         ('INPUT + STRICT EVIDENCE', overlay),
                         ('CANDIDATE / PRIORITY', np.dstack((result.candidate_mask*90,
                                                           result.priority_mask*255,
                                                           result.candidate_mask*255)).astype(np.uint8)),
                         ('TEXTURE + DISPLACEMENT SALIENCE', salience_heat),
                         ('INPUT + SALIENCE', salience_overlay),
                         ('SALIENCE CANDIDATES', cv2.cvtColor(result.salience_mask.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)),
                         ('MULTI-WINDOW SSIM', ssim_heat), ('NORMAL-ONLY CALIBRATION', normal_heat),
                         ('INPUT + NORMAL CALIBRATION', normal_overlay),
                         ('SUPPORTED SALIENCE (0 - 6)', supported_heat),
                         ('INPUT + SUPPORTED SALIENCE', supported_overlay),
                         ('ORIGINAL SALIENCE (0 - 6)', salience_comparison_heat)):
        width = min(520, shape[1])
        resized = cv2.resize(panel, (width, max(1, round(shape[0]*width/shape[1]))))
        resized = cv2.copyMakeBorder(resized, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
        cv2.putText(resized, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(resized)
    overview = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6])))
    salience_overview = np.vstack((np.hstack([panels[0], panels[2], panels[3]]), np.hstack(panels[6:9])))
    methods_overview = np.vstack((np.hstack([panels[0], panels[14], panels[12]]),
                                 np.hstack([panels[9], panels[10], panels[13]])))
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        (out / name).unlink(missing_ok=True)
    np.savez_compressed(out / ARTIFACTS['scores'], **arrays)
    for key, array in dict(heatmap=heat, tolerant_heatmap=tolerant_heat, overlay=overlay,
                          candidates=result.candidate_mask.astype(np.uint8)*255,
                          priority=result.priority_mask.astype(np.uint8)*255,
                          valid=result.comparison_valid.astype(np.uint8)*255, confidence=result.confidence,
                          overview=overview, salience_heatmap=salience_heat, salience_overlay=salience_overlay,
                          salience_candidates=result.salience_mask.astype(np.uint8)*255,
                          salience_overview=salience_overview, ssim_heatmap=ssim_heat, ssim_overlay=ssim_overlay,
                          ssim_valid=result.ssim_valid.astype(np.uint8)*255,
                          normal_heatmap=normal_heat, normal_overlay=normal_overlay,
                          normal_valid=result.normal_valid.astype(np.uint8)*255,
                          methods_overview=methods_overview,
                          supported_heatmap=supported_heat, supported_overlay=supported_overlay).items():
        write_image(out / ARTIFACTS[key], array)
    for name, response in zip(('color', 'structure', 'multiscale'),
                              (result.color_response, result.structure_response, result.multiscale_response)):
        scale = result.metadata['calibration'][name]['scale']
        if name == 'structure':
            scale /= result.metadata['config']['structure_weight']
        write_image(out / ARTIFACTS[name], np.rint(np.clip(response/(2*scale), 0, 1)*255).astype(np.uint8))
    path = out / 'anomaly.json'
    path.write_text(document, encoding='utf-8')
    return path


def run_result(path: str | Path, output_dir: str | Path | None = None, *,
               config: AnomalyConfig | None = None, normal_config: str | Path | None = None) -> Path:
    """Consume a saved match manifest, with all file paths confined to its directory."""
    manifest = Path(path).resolve()
    if manifest.is_dir():
        manifest /= 'result.json'
    metadata = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if (not isinstance(metadata, dict) or metadata.get('format_version') != 1
            or metadata.get('accepted') is not True or not isinstance(metadata.get('artifacts'), dict)):
        raise ValueError('Anomaly analysis requires an accepted matcher result with a normal reference')
    paths = {}
    arrays = {}
    for key in ('input', 'supposed_to_be_good', 'reference_valid', 'reference_photometric_valid', 'reference_source_map'):
        name = metadata['artifacts'].get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f'Missing normal-reference artifact: {key}; rerun periodic_template_matcher')
        resolved = (manifest.parent / name).resolve()
        if not resolved.is_relative_to(manifest.parent):
            raise ValueError(f'Artifact must stay inside the result directory: {key}')
        paths[key] = resolved
        arrays[key] = read_image(resolved, cv2.IMREAD_UNCHANGED)
    model, semantic_labels, semantic_valid = None, None, None
    if normal_config is not None:
        from .normal_baseline import build_normal_baseline
        name = metadata['artifacts'].get('prediction_mask')
        if not isinstance(name, str) or not name:
            raise ValueError('Normal calibration requires the saved semantic prediction mask')
        label_path = (manifest.parent / name).resolve()
        if not label_path.is_relative_to(manifest.parent):
            raise ValueError('Prediction mask must stay inside the result directory')
        paths['prediction_mask'] = label_path
        semantic_labels = read_image(label_path, cv2.IMREAD_UNCHANGED)
        name = metadata['artifacts'].get('prediction_valid')
        if not isinstance(name, str) or not name:
            raise ValueError('Normal calibration requires saved semantic prediction validity')
        valid_path = (manifest.parent / name).resolve()
        if not valid_path.is_relative_to(manifest.parent):
            raise ValueError('Prediction validity must stay inside the result directory')
        paths['prediction_valid'] = valid_path
        semantic_valid = read_image(valid_path, cv2.IMREAD_UNCHANGED)
        model = build_normal_baseline(normal_config)
    result = analyze_pair(arrays['input'], arrays['supposed_to_be_good'], arrays['reference_valid'],
                          photometric_valid=arrays['reference_photometric_valid'],
                          source_map=arrays['reference_source_map'], config=config,
                          normal_baseline=model, semantic_labels=semantic_labels, semantic_valid=semantic_valid)
    source = dict(manifest=str(manifest), artifacts={k: str(v) for k, v in paths.items()},
                  normal_reference=metadata.get('normal_reference'))
    return save_analysis(result, arrays['input'], arrays['supposed_to_be_good'],
                         output_dir if output_dir is not None else manifest.parent / 'anomaly',
                         source_metadata=source, protected_paths=(*paths.values(), manifest,
                             *(model.protected_paths if model is not None else ())))
