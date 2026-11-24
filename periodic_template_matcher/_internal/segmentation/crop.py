"""Restore rectangular working crops to original image coordinates."""

from dataclasses import fields

import numpy as np

from ..models import MatchResult, PreparedInputs


def crop_origin(region):
    bounds = region.get('valid_bounds', {})
    return bounds.get('x_start', 0), bounds.get('y_start', 0)


def _translation(x, y):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def rebase_matrix(matrix, target, source):
    if matrix is None or (target == (0, 0) and source == (0, 0)):
        return matrix
    x, y = source
    return (_translation(*target) @ np.asarray(matrix) @ _translation(-x, -y)).tolist()


def rebase_diagnostics(registration, target, sources, primary):
    to_target = _translation(*target)
    to_crops = [_translation(-x, -y) for x, y in sources]
    objects = registration.get('ordered_objects', {}).get('objects', [])
    owner = {obj['identity']: obj['template_index'] for obj in objects}

    def matrix(value, source):
        if value is None or (target == (0, 0) and sources[source] == (0, 0)):
            return value
        return (to_target @ np.asarray(value) @ to_crops[source]).tolist()

    def visit(value, source):
        if isinstance(value, list):
            return [visit(item, source) for item in value]
        if not isinstance(value, dict):
            return value
        source = value.get('template_index', owner.get(value.get('identity'), source))
        restored = {}
        for key, item in value.items():
            if key in ('matrix_template_to_target', 'corrected_matrix'):
                item = matrix(item, source)
            elif key == 'supporting_transforms':
                references = value.get('supporting_references', [])
                item = [matrix(H, references[i]['template_index'] if i < len(references) else source)
                        for i, H in enumerate(item)]
            elif key == 'center_x':
                item += target[0]
            elif key == 'center_y':
                item += target[1]
            elif key in ('target_bbox', 'source_target_bbox'):
                item = [item[0]+target[0], item[1]+target[1],
                        item[2]+target[0], item[3]+target[1]]
            elif key == 'source_bbox':
                sx, sy = sources[source]
                item = [item[0]+sx, item[1]+sy, item[2]+sx, item[3]+sy]
            else:
                item = visit(item, source)
            restored[key] = item
        return restored

    return visit(registration, primary)


def embed_crop(value, shape, origin):
    """Place an array at its crop origin on a zero-filled canvas."""
    x, y = origin
    height, width = value.shape[:2]
    canvas = np.zeros((*shape[:2], *value.shape[2:]), dtype=value.dtype)
    canvas[y:y+height, x:x+width] = value
    return canvas


def restore_crop(result: MatchResult, inputs: PreparedInputs) -> MatchResult:
    """Place target arrays back in their original canvas and rebase transforms."""
    primary = result.registration['ranking'][0]
    if inputs.original_templates is not None:
        result.template_image = inputs.original_templates[primary]
    x, y = crop_origin(inputs.regions['target'])
    if inputs.original_target is not None:
        target = inputs.original_target
        for field in fields(MatchResult):
            if field.name in ('registration', 'input_image', 'template_image'):
                continue
            value = getattr(result, field.name)
            if value is not None:
                setattr(result, field.name, embed_crop(value, target.shape, (x, y)))
        result.input_image = target
    sources = [crop_origin(region) for region in inputs.regions['templates']]
    if (x, y) != (0, 0) or any(origin != (0, 0) for origin in sources):
        result.registration = rebase_diagnostics(result.registration, (x, y), sources, primary)
    return result
