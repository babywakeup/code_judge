"""Post-hoc evaluation of frozen graph-cut or SAM3 predictions, never prediction code.

All non-ignored labels are unioned. Ignored shapes have ``flags.ignore=true``
or label ``ignore``, ``_ignore_`` or ``__ignore__`` (case-insensitive). Even
ignored shapes must be valid polygons/rectangles. Coordinates are pixel centers,
rounded with NumPy rint; OpenCV fills inclusive boundaries. Out-of-image and
degenerate shapes are rejected rather than silently clipped or discarded.

Metrics cover the *entire* image, including missing reference coverage. Both
empty masks score one; an empty denominator otherwise scores zero. Empty GT
has coverage one (no uncovered GT). A verified external image takes precedence
over an embedded copy; differences are recorded, not silently accepted as equal.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


_REQUIRED = {'mask', 'valid', 'input', 'overlay', 'seeds', 'contours'}
_OUTPUTS = ('gt_mask.png', 'comparison.png', 'metrics.json')


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes, description: str) -> dict:
    try:
        value = json.loads(data.decode('utf-8-sig'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid {description} JSON') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{description} must be a JSON object')
    return value


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _decode(data: bytes, description: str) -> np.ndarray:
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError(f'Cannot decode {description}') from exc
    if image is None:
        raise ValueError(f'Cannot decode {description}')
    return image


def _frozen_prediction(path: Path):
    """Read and hash *every* artifact before any caller may read GT."""
    manifest_path = (path / 'prediction.json' if path.is_dir() else path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = _json(manifest_bytes, 'prediction manifest')
    if (type(manifest.get('format_version')) is not int or manifest['format_version'] != 1
            or manifest.get('kind') not in ('box_graphcut_prediction', 'sam3_prediction')):
        raise ValueError('Unsupported prediction manifest format_version or kind')
    shape = manifest.get('image_shape')
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError('image_shape must be [H, W] with positive integers')
    if not isinstance(manifest.get('config'), dict) or not isinstance(manifest.get('boxes'), list):
        raise ValueError('Prediction config must be an object and boxes must be a list')
    artifacts, hashes = manifest.get('artifacts'), manifest.get('artifact_sha256')
    if (not isinstance(artifacts, dict) or not _REQUIRED.issubset(artifacts)
            or not isinstance(hashes, dict) or set(hashes) != set(artifacts)):
        raise ValueError('Every prediction artifact must have an artifact_sha256 hash')
    sources = manifest.get('source')
    if not isinstance(sources, dict) or 'target' not in sources:
        raise ValueError('Prediction source.target is required')
    protected = {manifest_path}
    for source in sources.values():
        if (not isinstance(source, dict) or not isinstance(source.get('path'), str)
                or not Path(source['path']).is_absolute() or not _digest(source.get('sha256'))):
            raise ValueError('Each source requires an absolute path and sha256 digest')
        protected.add(Path(source['path']).resolve())
    frozen = {}
    for key, name in artifacts.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError('Artifact paths must be relative to the prediction directory')
        artifact = (manifest_path.parent / name).resolve()
        if not _within(artifact, manifest_path.parent) or artifact == manifest_path:
            raise ValueError('Artifact path escapes or aliases the prediction manifest')
        if not _digest(hashes[key]):
            raise ValueError(f'Invalid sha256 hash for artifact {key}')
        data = artifact.read_bytes()
        if _sha256(data) != hashes[key].lower():
            raise ValueError(f'Frozen artifact sha256 hash mismatch: {key}')
        frozen[key] = data
        protected.add(artifact)
    image = _decode(frozen['input'], 'prediction input')
    if image.shape != (*shape, 3) or image.dtype != np.uint8:
        raise ValueError('Prediction input must be full-image uint8 BGR')
    masks = []
    for key in ('mask', 'valid'):
        mask = _decode(frozen[key], key)
        allowed = (0, 255) if key == 'valid' or mask.max() > 1 else (0, 1)
        if (mask.shape != tuple(shape) or mask.dtype != np.uint8
                or not np.isin(mask, allowed).all()):
            raise ValueError(f'{key} must be a full-image uint8 binary mask')
        masks.append(mask != 0)
    return manifest_path, manifest, _sha256(manifest_bytes), image, masks, protected


def pixel_metrics(prediction: np.ndarray, gt: np.ndarray) -> dict:
    """Return integer confusion counts and finite full-image binary scores."""
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError('Prediction and GT must have the same two-dimensional shape')
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(pred.size - tp - fp - fn)
    empty = tp + fp + fn == 0

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float(empty)

    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Precision=ratio(tp, tp + fp), Recall=ratio(tp, tp + fn),
                IoU=ratio(tp, tp + fp + fn), Dice=ratio(2 * tp, 2 * tp + fp + fn))


def rasterize_labelme(annotation: dict, image_shape: tuple[int, int]) -> np.ndarray:
    """Validate and union LabelMe polygons/rectangles into a boolean GT mask."""
    height, width = image_shape
    for key, expected in (('imageHeight', height), ('imageWidth', width)):
        if type(annotation.get(key)) is not int or annotation[key] != expected:
            raise ValueError(f'GT {key} does not match prediction image dimensions')
    shapes = annotation.get('shapes')
    if not isinstance(shapes, list):
        raise ValueError('GT shapes must be a list')
    result = np.zeros((height, width), np.uint8)
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f'GT shape {index} must be an object')
        kind, label = shape.get('shape_type', 'polygon'), shape.get('label')
        if kind not in ('polygon', 'rectangle'):
            raise ValueError(f'Unsupported GT shape_type: {kind!r}')
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f'GT shape {index} needs a nonempty label')
        flags = shape.get('flags', {})
        if not isinstance(flags, dict) or type(flags.get('ignore', False)) is not bool:
            raise ValueError(f'GT shape {index} has invalid ignore flags')
        try:
            points = np.asarray(shape.get('points'))
            if points.dtype.kind not in 'iuf':
                raise ValueError('Points must be numeric, not strings or booleans')
            points = points.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'Invalid GT points for shape {index}') from exc
        if (points.ndim != 2 or points.shape[1] != 2
                or (len(points) != 2 if kind == 'rectangle' else len(points) < 3)
                or not np.isfinite(points).all()):
            raise ValueError(f'Invalid GT point count or coordinates for shape {index}')
        if (np.any(points < 0) or np.any(points[:, 0] > width - 1)
                or np.any(points[:, 1] > height - 1)):
            raise ValueError(f'GT shape {index} has out-of-image points')
        points = np.rint(points).astype(np.int32)
        if kind == 'rectangle':
            (x0, y0), (x1, y1) = points
            points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
        if cv2.contourArea(points) <= 0:
            raise ValueError(f'Degenerate GT shape {index}')
        if flags.get('ignore', False) or label.casefold() in {'ignore', '_ignore_', '__ignore__'}:
            continue
        cv2.fillPoly(result, [points], 255)
    return result != 0


def _output_directory(out: Path, prediction_dir: Path, protected: set[Path]) -> None:
    if _within(out, prediction_dir):
        raise ValueError('Evaluation output must be outside the frozen prediction directory')
    if out.exists() or out.is_symlink():
        raise ValueError('Evaluation output must be a new independent directory')
    if any(_within(path, out) for path in protected):
        raise ValueError('Evaluation output overlaps a protected source, GT or frozen artifact')
    if any((out / name).resolve() in protected for name in _OUTPUTS):
        raise ValueError('Evaluation output name overlaps a protected artifact')


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'Cannot encode evaluation image: {path.name}')
    with path.open('xb') as stream:
        stream.write(encoded.tobytes())


def evaluate_prediction(prediction: str | Path, gt: str | Path,
                        out: str | Path | None = None) -> dict:
    """Evaluate a frozen manifest, writing only to a new independent directory.

    The saved input is authoritative. Original source files need not still exist;
    their recorded paths are protected, not reread. If the GT-referenced image
    exists, its decoded BGR pixels must equal the frozen input exactly.
    """
    manifest_path, manifest, manifest_hash, image, masks, protected = _frozen_prediction(Path(prediction))
    prediction_dir = manifest_path.parent
    gt_path = Path(gt).resolve()
    protected.add(gt_path)
    output = (Path(out) if out is not None else
              prediction_dir.with_name(prediction_dir.name + '_evaluation')).resolve()
    _output_directory(output, prediction_dir, protected)

    # No GT bytes are read until every frozen artifact above has passed its hash.
    gt_bytes = gt_path.read_bytes()
    annotation = _json(gt_bytes, 'GT')
    name = annotation.get('imagePath')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('GT imagePath must identify the original target filename')
    target_name = Path(manifest['source']['target']['path']).name
    if name.replace('\\', '/').rsplit('/', 1)[-1].casefold() != target_name.casefold():
        raise ValueError('GT imagePath filename does not match the original target')
    truth = rasterize_labelme(annotation, image.shape[:2])
    gt_image = (gt_path.parent / name.replace('\\', '/')).resolve()
    protected.add(gt_image)
    _output_directory(output, prediction_dir, protected)
    image_match = 'name_dimensions_only'
    embedded_matches = None
    embedded_data = annotation.get('imageData')
    if embedded_data not in (None, ''):
        if not isinstance(embedded_data, str):
            raise ValueError('Embedded GT imageData must be base64 text')
        try:
            embedded = _decode(base64.b64decode(embedded_data, validate=True), 'embedded GT image')
        except (ValueError, binascii.Error) as exc:
            raise ValueError('Invalid embedded GT imageData') from exc
        if embedded.shape != image.shape or embedded.dtype != image.dtype:
            raise ValueError('Embedded GT image dimensions/channels do not match frozen input')
        embedded_matches = bool(np.array_equal(embedded, image))
    if gt_image.exists():
        decoded = _decode(gt_image.read_bytes(), 'GT referenced image')
        if decoded.dtype != image.dtype or not np.array_equal(decoded, image):
            raise ValueError('GT referenced image pixels do not match frozen prediction input')
        image_match = 'decoded_pixels'
    elif embedded_matches is not None:
        if not embedded_matches:
            raise ValueError('Embedded GT image pixels do not match frozen input; no external image to verify')
        image_match = 'embedded_decoded_pixels'
    pred, valid = masks
    result = pixel_metrics(pred, truth)
    comparable = int(np.count_nonzero(truth & valid))
    total = int(np.count_nonzero(truth))
    result.update(
        gt_comparable_pixels=comparable, gt_uncovered_pixels=total - comparable,
        gt_coverage_fraction=comparable / total if total else 1.,
        prediction_manifest=str(manifest_path), prediction_manifest_sha256=manifest_hash,
        gt=str(gt_path), gt_sha256=_sha256(gt_bytes), image_shape=list(image.shape[:2]),
        image_match=image_match, embedded_image_matches_input=embedded_matches,
        warnings=(['embedded_copy_differs_external_image_verified'] if embedded_matches is False else []),
        evaluation_domain='full_image',
        comparison_colors_bgr={'TP': [0, 255, 0], 'FP': [0, 0, 255], 'FN': [255, 0, 0]},
        artifacts={'gt_mask': 'gt_mask.png', 'comparison': 'comparison.png'},
    )
    comparison = np.zeros_like(image)
    comparison[pred & truth] = (0, 255, 0)
    comparison[pred & ~truth] = (0, 0, 255)
    comparison[~pred & truth] = (255, 0, 0)
    # exist_ok=False plus exclusive file creation also protect against accidental reruns.
    output.mkdir(parents=True, exist_ok=False)
    _write_png(output / 'gt_mask.png', truth.astype(np.uint8) * 255)
    _write_png(output / 'comparison.png', comparison)
    with (output / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_prediction(args.prediction, args.gt, args.out)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Evaluation failed: {exc}\n')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())