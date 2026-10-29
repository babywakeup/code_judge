"""Read class-ID masks or explicitly class-mapped LabelMe segmentation shapes."""

from __future__ import annotations

import json

import cv2
import numpy as np

from ..utils import read_image
from ..utils.types import Diagnostics, LabelMap, Resolver, Shape


def _draw_shape(labels, item, class_id):
    pts = np.asarray(item['points'], np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError('LabelMe points must be finite [x, y] pairs')
    kind = item.get('shape_type', 'polygon')
    if kind == 'polygon' and len(pts) >= 3:
        cv2.fillPoly(labels, [np.rint(pts).astype(np.int32)], class_id)
    elif kind == 'rectangle' and len(pts) == 2:
        lo, hi = np.rint(pts.min(axis=0)).astype(int), np.rint(pts.max(axis=0)).astype(int)
        cv2.rectangle(labels, tuple(lo), tuple(hi), class_id, -1)
    elif kind == 'circle' and len(pts) == 2:
        cv2.circle(labels, tuple(np.rint(pts[0]).astype(int)),
                   round(float(np.linalg.norm(pts[1]-pts[0]))), class_id, -1)
    else:
        raise ValueError(f'Unsupported/invalid segmentation shape: {kind}')


def load_objects(spec, resolve, shape, class_id, cycle):
    """Preserve each original polygon, ordered by x rather than JSON draw order."""
    order = spec.get('object_order')
    if order is None:
        return []
    if (not isinstance(order, (list, tuple)) or len(order) != len(cycle)
            or any(not isinstance(name, str) for name in order) or set(order) != set(cycle)):
        raise ValueError('object_order must contain each configured identity exactly once')
    if not spec.get('labelme'):
        raise ValueError('object_order requires original LabelMe shapes')
    data = json.loads(resolve(spec['labelme']).read_text(encoding='utf-8-sig'))
    if (data.get('imageHeight'), data.get('imageWidth')) != tuple(shape[:2]):
        raise ValueError('LabelMe image dimensions must match original object image')
    objects = []
    for index, item in enumerate(data.get('shapes', [])):
        if spec.get('class_map', {}).get(item['label']) != class_id:
            continue
        mask = np.zeros(shape[:2], np.uint8)
        _draw_shape(mask, item, 1)
        yy, xx = np.where(mask > 0)
        if not len(xx):
            raise ValueError('Object reference shape is outside its source image')
        objects.append(dict(shape_index=index, mask=mask, center_x=float((xx.min()+xx.max())/2)))
    if len(objects) != len(order):
        raise ValueError('object_order length must equal number of ordered-class LabelMe shapes')
    objects.sort(key=lambda item: item['center_x'])
    return [dict(obj, identity=name) for obj, name in zip(objects, order)]


def load_labels(spec: Diagnostics, resolve: Resolver, shape: Shape) -> LabelMap | None:
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
        _draw_shape(labels, item, classes[name])
    return labels
