"""Resolve crop-sensitive identity ambiguity using the original masked canvas."""

import cv2
import numpy as np

from ..segmentation.crop import crop_origin, embed_crop, rebase_diagnostics, rebase_matrix
from .matching import ObjectPrediction


def _has_sampled_support(mask, scale):
    h, w = mask.shape
    size = round(w*scale), round(h*scale)
    weights = (mask > 0).astype(np.float32)
    nearest = cv2.resize(weights, size, interpolation=cv2.INTER_NEAREST) > 0
    if not nearest.any():
        return False
    area = cv2.resize(weights, size, interpolation=cv2.INTER_AREA) > 1e-6
    return bool((nearest & area).any())


def retry_cropped_identity(inputs, match, located, config, locator):
    previous = located.prediction
    images = inputs.original_templates if inputs.original_templates is not None else inputs.templates
    target = inputs.original_target if inputs.original_target is not None else inputs.target
    shapes = [image.shape for image in (*images, target)]
    scale = config.matching_scale(*shapes)
    if any(round(min(shape[:2])*scale) < 32 for shape in shapes):
        return previous
    source_origins = [crop_origin(region) for region in inputs.regions['templates']]
    destination = crop_origin(inputs.regions['target'])
    masks, references = [], []
    axis = 0 if config.object_order_axis == 'x' else 1
    for image, valid, refs, origin in zip(images, located.source_valids, inputs.objects, source_origins):
        masks.append(embed_crop(valid, image.shape, origin))
        start = origin[axis]
        bounds = start, start + valid.shape[1 - axis]
        references.append([dict(ref, mask=embed_crop(ref['mask'], image.shape, origin),
                                _order_axis_bounds=bounds) for ref in refs])
    target_mask = embed_crop(inputs.target_valid, target.shape, destination)
    pair_scales = [config.matching_scale(image.shape, target.shape) for image in images]
    # Mask samples can disappear when the original canvas changes the resize grid.
    if not all(_has_sampled_support(target_mask, s) for s in {scale, *pair_scales}):
        return previous
    if not all(_has_sampled_support(mask, s)
               for mask, pair in zip(masks, pair_scales) for s in {scale, pair}):
        return previous
    matrices = [
        rebase_matrix(candidate.get('matrix_template_to_target'), destination, origin)
        if candidate.get('geometry_accepted', match.result.accepted) else None
        for candidate, origin in zip(match.result.registration['matches'], source_origins)]
    retried = locator(images, references, masks, target, target_mask, matrices, config)
    info = retried.diagnostics
    order = previous.diagnostics.get('visible_identities')
    used = bool(order) and info['status'] == 'accepted' and info.get('visible_identities') == order
    retry = dict(used=used, crop_order_margin=previous.diagnostics.get('order_margin'),
                 canvas_order_margin=info.get('order_margin'), canvas_status=info['status'],
                 canvas_order=info.get('visible_order'))
    if not used:
        previous.diagnostics['identity_retry'] = retry
        return previous

    x, y = destination
    h, w = inputs.target.shape[:2]
    region = np.s_[y:y+h, x:x+w]
    diagnostics = rebase_diagnostics(info, (-x, -y),
                                    [(-sx, -sy) for sx, sy in source_origins], match.template_index)
    diagnostics['identity_retry'] = retry
    return ObjectPrediction(retried.labels[region].copy(), retried.identities[region].copy(),
                            retried.valid[region].copy(), diagnostics)
