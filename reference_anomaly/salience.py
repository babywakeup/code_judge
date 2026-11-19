"""Experimental texture-conditioned residual prominence, separate from raw evidence."""
from __future__ import annotations

import numpy as np


def _background(values: np.ndarray) -> tuple[float, float]:
    if len(values) > 100000:
        values = values[np.linspace(0, len(values)-1, 100000, dtype=int)]
    if not len(values):
        return 0., .5
    median = float(np.median(values))
    sigma = float(1.4826*np.median(np.abs(values-median)))
    return median, max(.5, sigma)


def compute_salience(strict: np.ndarray, tolerant: np.ndarray, texture: np.ndarray,
                     valid: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rank excess over common residuals in comparable reference texture groups.

    This deliberately emphasizes local outliers. Broad changes may be suppressed;
    callers must retain strict/tolerant evidence as independent outputs.
    """
    trusted = valid & (confidence == 255)
    fit = trusted if trusted.sum() >= 64 else valid
    values = texture[fit]
    if len(values) > 100000:
        values = values[np.linspace(0, len(values)-1, 100000, dtype=int)]
    cuts = np.unique(np.quantile(values, [.25, .5, .75])) if len(values) else np.array([])
    groups = np.digitize(texture, cuts, right=True)
    global_median, global_scale = _background(strict[fit])
    excess = np.zeros(strict.shape, np.float32)
    records = []
    for group in range(len(cuts)+1):
        region = valid & (groups == group)
        if not region.any():
            continue
        evidence = fit & region
        samples = int(evidence.sum())
        if samples >= 64:
            median, scale = _background(strict[evidence])
            status = 'fitted'
        else:
            median, scale = global_median, global_scale
            status = 'global_fallback'
        # Require half a background scale of excess before displaying prominence.
        excess[region] = np.maximum(0., (strict[region]-median)/scale-.5)
        records.append(dict(group=group, pixels=int(region.sum()), samples=samples,
                            status=status, background_median=median, background_scale=scale))
    ratio = np.divide(tolerant, strict, out=np.ones_like(strict), where=strict > 1e-6)
    weight = (.25+.75*np.clip(ratio, 0., 1.)).astype(np.float32)
    weight *= np.where(confidence == 255, 1., .5).astype(np.float32)
    weight[~valid] = 0
    score = excess*weight
    score[~valid] = 0
    return score, weight, dict(
        method='texture_conditioned_excess_with_displacement_downweighting',
        experimental=True, texture_measure='smoothed_reference_gradient',
        texture_cutpoints=cuts.tolist(), texture_groups=records,
        shift_retention=.25, seam_retention=.5, background_scale_floor=.5,
        excess_offset=.5, candidate_high_threshold=3., candidate_low_threshold=1.5,
        limitation='Common or broad changes can be suppressed; inspect strict evidence as well.')
