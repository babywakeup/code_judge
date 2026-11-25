"""Record accepted periodic predictions separately from their merge footprint."""

import cv2
import numpy as np


def record_periodic_prediction(result, labels, accepted, *, region=None):
    """Keep the candidate's labels with deterministic first-accepted precedence."""
    if not accepted.any():
        return
    if result.completion_valid is None:
        result.completion_valid = np.zeros(result.mapped_labels.shape, bool)
        result.completion_labels = np.zeros_like(result.mapped_labels)
    if region is None:
        region = np.s_[:, :]
    valid = result.completion_valid[region]
    layer = result.completion_labels[region]
    write = accepted & (~valid | ((layer == 0) & (labels > 0)))
    layer[write] = labels[write]
    valid |= accepted


def projected_components(labels, support, seeds):
    """Expand approved fragments to their own projected foreground contours."""
    prediction = seeds.copy()
    for class_id in np.unique(labels[seeds]):
        if not class_id:
            continue
        foreground = (labels == class_id) & support
        count, components = cv2.connectedComponents(foreground.astype(np.uint8))
        selected = np.zeros(count, bool)
        selected[components[seeds & foreground]] = True
        selected[0] = False
        prediction |= selected[components]
    return prediction
