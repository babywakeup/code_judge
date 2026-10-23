"""Read class-ID masks or explicitly class-mapped LabelMe segmentation shapes."""
import json

import cv2
import numpy as np

from ..utils import read_image


def load_labels(spec, resolve, shape):
    """None means absent, never a generated annotation. Last shape wins overlaps.

    LabelMe unpainted pixels are background (0). For partial annotations supply
    annotation_valid separately; it is NOT inferred from foreground pixels.
    """
    if spec.get('labels') and spec.get('labelme'):
        raise ValueError('Choose labels OR labelme per template')
    if spec.get('labels'):
        labels = read_image(resolve(spec['labels']), cv2.IMREAD_UNCHANGED)
        if labels.shape != tuple(shape) or labels.dtype not in (np.uint8, np.uint16):
            raise ValueError('labels must be a same-size uint8/uint16 single-channel mask')
        return labels.astype(np.uint16)
    if not spec.get('labelme'):
        return None
    data = json.loads(resolve(spec['labelme']).read_text(encoding='utf-8-sig'))
    if (data.get('imageHeight'), data.get('imageWidth')) != tuple(shape):
        raise ValueError('LabelMe image dimensions must match the original template')
    classes = spec.get('class_map', {})
    if not classes or any(type(v) is not int or not 1 <= v <= 65535 for v in classes.values()):
        raise ValueError('LabelMe requires an explicit class_map: name -> integer ID 1..65535')
    if len(set(classes.values())) != len(classes):
        raise ValueError('class_map IDs must be unique')
    labels = np.zeros(shape, np.uint16)
    for item in data.get('shapes', []):
        name = item['label']
        if name not in classes:
            raise ValueError(f'Unmapped LabelMe class: {name}')
        pts = np.asarray(item['points'], np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            raise ValueError('LabelMe points must be finite [x, y] pairs')
        kind = item.get('shape_type', 'polygon')
        if kind == 'polygon' and len(pts) >= 3:
            cv2.fillPoly(labels, [np.rint(pts).astype(np.int32)], classes[name])
        elif kind == 'rectangle' and len(pts) == 2:
            lo, hi = np.rint(pts.min(axis=0)).astype(int), np.rint(pts.max(axis=0)).astype(int)
            cv2.rectangle(labels, tuple(lo), tuple(hi), classes[name], -1)
        elif kind == 'circle' and len(pts) == 2:
            cv2.circle(labels, tuple(np.rint(pts[0]).astype(int)),
                       round(float(np.linalg.norm(pts[1] - pts[0]))), classes[name], -1)
        else:
            raise ValueError(f'Unsupported/invalid segmentation shape: {kind}')
    return labels


def demo_labels(shape):
    """Artificial shapes for transport illustration, NOT product segmentation."""
    h, w = shape
    labels = np.zeros(shape, np.uint16)
    cv2.circle(labels, (round(w * .33), round(h * .32)), round(min(h, w) * .05), 1, -1)
    cv2.rectangle(labels, (round(w * .46), round(h * .40)),
                  (round(w * .59), round(h * .45)), 2, -1)
    points = np.array([[.62, .61], [.70, .59], [.74, .68], [.65, .71]])
    cv2.fillPoly(labels, [np.rint(points * [w, h]).astype(np.int32)], 3)
    return labels