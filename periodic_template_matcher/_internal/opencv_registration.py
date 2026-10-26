"""OpenCV coarse translation search with masked, partial-overlap normalization."""
import cv2
import numpy as np


def translation_peaks(source, target, sm, tm, cfg):
    """Return (merit, dx, dy) peaks mapping source into target coordinates.

    Zero padding exposes every signed shift with at least one overlapping pixel.
    Separate mask correlations prevent padding and invalid pixels from contributing
    to the feature energies or overlap. TM_CCORR supplies the dot products;
    normalization is performed on each shift's actual valid intersection.
    """
    sh, sw = source.shape
    th, tw = target.shape
    sm = np.ascontiguousarray(sm > 0, dtype=np.float32)
    tm = np.ascontiguousarray(tm > 0, dtype=np.float32)
    source = np.ascontiguousarray(source * sm, dtype=np.float32)
    target = np.ascontiguousarray(target * tm, dtype=np.float32)
    if not sm.any() or not tm.any():
        return []

    def correlate(image, kernel):
        padded = cv2.copyMakeBorder(image, sh - 1, sh - 1, sw - 1, sw - 1,
                                    cv2.BORDER_CONSTANT, value=0)
        return cv2.matchTemplate(padded, kernel, cv2.TM_CCORR)

    overlap = np.rint(np.maximum(correlate(tm, sm), 0))
    numerator = correlate(target, source)
    energy_target = np.maximum(correlate(target * target, sm), 0)
    energy_source = np.maximum(correlate(tm, source * source), 0)
    energy = energy_target * energy_source
    coverage = overlap / float(tm.sum())
    eligible = ((coverage >= cfg.min_overlap)
                & (overlap >= cfg.min_overlap * sm.sum()) & (energy > 1e-8))
    merit = np.clip(numerator / np.sqrt(np.maximum(energy, 1e-8)), 0, 1)
    merit *= np.sqrt(np.minimum(coverage, 1))
    merit = np.where(eligible & np.isfinite(merit), merit, -1)
    radius = max(8, min(sh, sw, th, tw) // 20)
    peaks = []
    for _ in range(cfg.peaks_per_hypothesis):
        y, x = np.unravel_index(np.argmax(merit), merit.shape)
        if merit[y, x] < 0:
            break
        peaks.append((float(merit[y, x]), float(x - sw + 1), float(y - sh + 1)))
        merit[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1
    return peaks
