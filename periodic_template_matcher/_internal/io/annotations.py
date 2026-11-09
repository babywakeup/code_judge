"""Read class-ID masks or explicitly class-mapped LabelMe segmentation shapes."""

from __future__ import annotations

import json

import cv2
import numpy as np

from ...utils import read_image
from ...utils.types import Diagnostics, LabelMap, Resolver, Shape


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


def layout_axis(specs):
    """Templates share one layout axis; legacy lists mean left-to-right."""
    axes = set()
    for spec in specs:
        if 'instance_layouts' in spec:
            layouts = spec['instance_layouts']
            if not isinstance(layouts, dict) or not layouts:
                raise ValueError('instance_layouts must be a nonempty object')
            for layout in layouts.values():
                if not isinstance(layout, dict) or layout.get('order_axis') not in ('x', 'y'):
                    raise ValueError('instance_layouts order_axis must be "x" or "y"')
                axes.add(layout['order_axis'])
        elif spec.get('object_order') is not None:
            axes.add('x')
    if len(axes) > 1:
        raise ValueError('All templates must use the same instance order_axis')
    return next(iter(axes), None)


def object_layout(spec, class_id):
    """Read an explicitly class-bound layout, retaining legacy object_order."""
    if 'instance_layouts' not in spec:
        order = spec.get('object_order')
        return None if order is None else dict(order_axis='x', order=order)
    if 'object_order' in spec:
        raise ValueError('Do not mix instance_layouts with legacy object_order')
    layouts = spec['instance_layouts']
    if class_id is None:
        raise ValueError('instance_layouts requires an enabled class with instance_matching')
    if (not isinstance(layouts, dict) or len(layouts) != 1
            or {str(k) for k in layouts} != {str(class_id)}):
        raise ValueError('instance_layouts must contain exactly the configured instance_matching class ID')
    layout = next(iter(layouts.values()))
    if (not isinstance(layout, dict) or not {'order_axis', 'order'} <= set(layout)
            or set(layout) - {'order_axis', 'order', 'description'}):
        raise ValueError('Each instance_layouts entry requires order_axis and order')
    if layout['order_axis'] not in ('x', 'y'):
        raise ValueError('order_axis must be "x" or "y"')
    return dict(order_axis=layout['order_axis'], order=layout['order'])


def load_objects(spec, resolve, shape, class_id, cycle):
    """Preserve each polygon; bind class, axis and order independently of drawing."""
    layout = object_layout(spec, class_id)
    if layout is None:
        return []
    order = layout['order']
    if (not isinstance(order, (list, tuple)) or not 1 <= len(order) <= len(cycle)
            or any(not isinstance(name, str) for name in order)
            or len(set(order)) != len(order) or not set(order) <= set(cycle)):
        raise ValueError('Instance order must contain unique identities from the configured cycle')
    canonical = list(cycle)
    if list(order) not in [(canonical[i:]+canonical[:i])[:len(order)] for i in range(len(canonical))]:
        raise ValueError('Instance order must be a cyclic rotation or contiguous window of the configured identity cycle')
    if not spec.get('labelme'):
        raise ValueError('Instance order requires original LabelMe shapes')
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
        coords = xx if layout['order_axis'] == 'x' else yy
        objects.append(dict(shape_index=index, mask=mask,
                    **{f"center_{layout['order_axis']}": float((coords.min()+coords.max())/2)}))
    if len(objects) != len(order):
        raise ValueError('Instance order length must equal number of matching-class LabelMe shapes')
    objects.sort(key=lambda item: item[f"center_{layout['order_axis']}"])
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
