"""Instance coordinate adaptation for the common identity-layout solver.

Transpose, rather than relabel identities or rotate masks heuristically. The
same solver handles x and y; all exported transforms remain in input coordinates.
"""
from dataclasses import replace

import numpy as np

from ..models import ReferencePiece


SWAP = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])


def transpose_image(image):
    return np.ascontiguousarray(np.swapaxes(image, 0, 1))


def transpose_matrix(matrix):
    return None if matrix is None else (SWAP @ np.asarray(matrix) @ SWAP).tolist()


def transpose_config(cfg):
    swap = {'x': 'y', 'y': 'x'}
    return replace(cfg, object_order_axis=swap[cfg.object_order_axis],
                   class_periodicity={c: [swap[a] for a in axes] for c, axes in cfg.class_periodicity.items()},
                   period_vectors=[(v[1], v[0]) for v in cfg.period_vectors])


def transpose_diagnostics(value):
    """An involution, including nested corrected/supporting transforms."""
    if isinstance(value, list):
        return [transpose_diagnostics(item) for item in value]
    if not isinstance(value, dict):
        return value
    rename = {'center_x': 'center_y', 'center_y': 'center_x',
              'horizontal_fraction': 'vertical_fraction', 'vertical_fraction': 'horizontal_fraction',
              'left': 'top', 'top': 'left', 'right': 'bottom', 'bottom': 'right'}
    result = {}
    for key, item in value.items():
        if key in ('matrix_template_to_target', 'corrected_matrix'):
            item = transpose_matrix(item)
        elif key == 'supporting_transforms':
            item = [transpose_matrix(m) for m in item]
        elif key == 'target_vector':
            item = [item[1], item[0]]
        elif key == 'repeat_axes':
            item = ['y' if a == 'x' else 'x' for a in item]
        elif key == 'order_axis':
            item = 'y' if item == 'x' else 'x'
        else:
            item = transpose_diagnostics(item)
        result[rename.get(key, key)] = item
    return result


def transpose_prediction(prediction):
    return type(prediction)(transpose_image(prediction.labels), transpose_image(prediction.identities),
                            transpose_image(prediction.valid), transpose_diagnostics(prediction.diagnostics),
                            [ReferencePiece(p.template_index, np.asarray(transpose_matrix(p.matrix)),
                                            transpose_image(p.mask), p.kind)
                             for p in prediction.reference_pieces])
