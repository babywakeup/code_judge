"""FFT correlation kernels for translation search and lattice diagnostics."""

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils.types import Array, Diagnostics, Mask, Shape

def _corr(a: Array, b: Array, fft_shape: Shape) -> Array:
    fa = np.fft.rfft2(a, s=fft_shape)
    fb = np.fft.rfft2(b, s=fft_shape)
    return np.fft.irfft2(fa * np.conj(fb), s=fft_shape).real


def _translation_peaks(source: Array, target: Array, sm: Mask, tm: Mask,
                       cfg: MatcherConfig) -> list[tuple[float, float, float]]:
    """Linear (not circular) normalized correlation, multiple displacement peaks."""
    sh, sw = source.shape
    th, tw = target.shape
    shape = (cv2.getOptimalDFTSize(sh + th - 1), cv2.getOptimalDFTSize(sw + tw - 1))
    ys, xs = np.arange(-sh + 1, th), np.arange(-sw + 1, tw)
    index = np.ix_(ys % shape[0], xs % shape[1])
    overlap = _corr(tm, sm, shape)[index]
    num = _corr(target, source, shape)[index]
    den1 = _corr(target * target, sm, shape)[index]
    den2 = _corr(tm, source * source, shape)[index]
    corr = num / np.sqrt(np.maximum(den1 * den2, 1e-8))
    cov = np.maximum(overlap, 0) / max(float(tm.sum()), 1)
    eligible = (cov >= cfg.min_overlap) & (overlap >= cfg.min_overlap * sm.sum())
    corr = np.where(eligible, np.clip(corr, 0, 1) * np.sqrt(np.minimum(cov, 1)), -1)
    peaks = []
    radius = max(8, min(sh, sw, th, tw) // 20)
    for _ in range(cfg.peaks_per_hypothesis):
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        if corr[y, x] < 0:
            break
        peaks.append((float(corr[y, x]), float(xs[x]), float(ys[y])))
        corr[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1
    return peaks


def _periods(feature: Array, mask: Mask) -> list[Diagnostics]:
    """Diagnostic lattice candidates; harmonics are not unique cell identities."""
    h, w = feature.shape
    shape = (cv2.getOptimalDFTSize(2*h-1), cv2.getOptimalDFTSize(2*w-1))
    centered = (feature - feature[mask > 0].mean()) * mask
    corr = _corr(centered, centered, shape)
    energy = np.sqrt(np.maximum(_corr(centered*centered, mask, shape) * _corr(mask, centered*centered, shape), 1e-9))
    overlap = _corr(mask, mask, shape)
    result = []
    for axis, limit in ((0, h), (1, w)):
        stop = int(limit * 0.7)
        x = np.arange(max(12, limit//18), stop)
        vals = corr[x, 0]/energy[x, 0] if axis == 0 else corr[0, x]/energy[0, x]
        ovs = overlap[x, 0] if axis == 0 else overlap[0, x]
        candidates = [(float(vals[i]), int(x[i])) for i in range(1, len(x)-1)
                      if vals[i] > 0.65 and vals[i] >= vals[i-1] and vals[i] >= vals[i+1]
                      and ovs[i] > mask.sum()*0.3]
        for strength, shift in sorted(candidates, reverse=True)[:2]:
            result.append({'vector_work_px': [shift if axis == 1 else 0, shift if axis == 0 else 0],
                           'autocorrelation': strength})
    return result

