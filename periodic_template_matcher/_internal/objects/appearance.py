"""Masked object appearance search, including partially visible image edges."""

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils.types import Array, Mask

def _correlation_peaks(patch: Array, mask: Mask, target: Array, valid: Mask,
                       cfg: MatcherConfig) -> list[tuple[float, int, int]]:
    """Masked zero-mean NCC on each actual intersection, including image edges."""
    ph, pw = patch.shape
    patch = np.ascontiguousarray(patch*mask, np.float32)
    mask = np.ascontiguousarray(mask, np.float32)
    valid = np.ascontiguousarray(valid, np.float32)
    target = np.ascontiguousarray(target*valid, np.float32)

    def corr(a, b):
        return cv2.matchTemplate(cv2.copyMakeBorder(a, ph-1, ph-1, pw-1, pw-1,
                                  cv2.BORDER_CONSTANT, value=0), b, cv2.TM_CCORR)

    count = np.maximum(corr(valid, mask), 0)
    st, ss = corr(target, mask), corr(valid, patch)
    et, es = corr(target*target, mask), corr(valid, patch*patch)
    cross = corr(target, patch)
    denominator = np.maximum(count, 1)
    numerator = cross - st*ss/denominator
    target_energy = np.maximum(et-st*st/denominator, 0)
    source_energy = np.maximum(es-ss*ss/denominator, 0)
    energy = target_energy*source_energy
    score = numerator/np.sqrt(np.maximum(energy, 1e-8))
    # FFT/convolution roundoff can give almost-flat intersections a spurious NCC
    # of 1. Require actual contrast in BOTH observed patches before normalization.
    eligible = ((count >= max(64, cfg.object_min_visible*float(mask.sum())))
                & (target_energy/denominator > .0025) & (source_energy/denominator > .0025))
    score = np.where(eligible & np.isfinite(score), np.clip(score, -1, 1), -1)
    peaks = []
    partial_cycle = cfg.object_visible_count is not None
    locations = []
    budget = cfg.object_candidates * (2 if partial_cycle else 1)
    for _ in range(budget):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        value = float(score[y, x])
        if value < cfg.object_min_score:
            break
        peaks.append((value, x-pw+1, y-ph+1))
        rx, ry = max(6, pw//3), max(6, ph//8)
        score[max(0, y-ry):y+ry+1, max(0, x-rx):x+rx+1] = -1
        if partial_cycle:
            # Vertical aliases of a strong interior match must not exclude a
            # weaker clipped occurrence at another horizontal position.
            location = next((loc for loc in locations if abs(loc[0]-x) <= rx), None)
            if location is None:
                location = [x, 0]
                locations.append(location)
            location[1] += 1
            if location[1] >= 2:
                score[:, max(0, location[0]-rx):location[0]+rx+1] = -1
            if len(locations) >= cfg.object_candidates:
                break
    return peaks

