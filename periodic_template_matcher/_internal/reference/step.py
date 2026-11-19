"""Build normal appearance from verified template pixels, never target pixels."""
from __future__ import annotations

import cv2
import numpy as np

from ...utils import to_gray
from ..models import MatchResult, PreparedInputs


def _channels(image: np.ndarray) -> np.ndarray:
    return image[..., None] if image.ndim == 2 else image[..., :3]


def fit_photometry(source: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    """Deterministic robust source-to-target affine fit with bounded sampling.

    Exclude saturation, mask boundaries and strong edges before residual clipping.
    Seed with two-point hypotheses so a bright defect cannot bias ordinary LS.
    Insufficient contrast/evidence is explicit, not a successful identity fit.
    """
    a, b = _channels(source), _channels(target)
    channels = a.shape[2]
    fallback = dict(status='insufficient_evidence', gain=[1.] * channels,
                    offset=[0.] * channels, samples=0, inlier_fraction=0., residual_mad=None)
    eligible = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    eligible &= ((a > 5) & (a < 250) & (b > 5) & (b < 250)).all(axis=2)
    for image in (source, target):
        gray = to_gray(image).astype(np.float32)
        edge = np.maximum(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)),
                          np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))) / 4
        eligible &= cv2.dilate((edge > 32).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    indices = np.flatnonzero(eligible)
    if len(indices) > 30000:
        indices = indices[np.linspace(0, len(indices)-1, 30000, dtype=int)]
    x = a.reshape(-1, channels)[indices].astype(np.float64)
    y = b.reshape(-1, channels)[indices].astype(np.float64)
    fallback['samples'] = len(x)
    if len(x) < 64:
        return fallback
    if (np.std(x, axis=0) < 2).any() or (np.std(y, axis=0) < 2).any():
        fallback.update(status='insufficient_contrast', offset=np.median(y-x, axis=0).tolist())
        return fallback
    rng = np.random.default_rng(0)
    gain, offset = [], []
    for channel in range(channels):
        xc, yc = x[:, channel], y[:, channel]
        # Evaluate hypotheses on a bounded subsample; refit on all eligible pixels.
        stride = max(1, len(xc)//1500)
        sx, sy = xc[::stride], yc[::stride]
        best = (float('inf'), 1., float(np.median(yc-xc)))
        for i, j in rng.integers(0, len(xc), (96, 2)):
            if abs(xc[i]-xc[j]) < 8:
                continue
            g = (yc[i]-yc[j])/(xc[i]-xc[j])
            o = yc[i]-g*xc[i]
            if .25 <= g <= 4 and abs(o) <= 200:
                error = float(np.median(np.abs(sy-(g*sx+o))))
                if error < best[0]:
                    best = error, float(g), float(o)
        gain.append(best[1])
        offset.append(best[2])
    gain, offset = np.asarray(gain), np.asarray(offset)
    inliers = np.ones(len(x), bool)
    for _ in range(5):
        residual = y-(x*gain+offset)
        center = np.median(residual, axis=0)
        mad = 1.4826*np.median(np.abs(residual-center), axis=0)
        inliers = (np.abs(residual-center) <= np.maximum(3., 3*mad)).all(axis=1)
        if inliers.sum() < max(64, .5*len(x)) or (np.std(x[inliers], axis=0) < 2).any():
            return dict(fallback, status='unreliable_fit')
        for channel in range(channels):
            design = np.column_stack((x[inliers, channel], np.ones(int(inliers.sum()))))
            gain[channel], offset[channel] = np.linalg.lstsq(design, y[inliers, channel], rcond=None)[0]
    error = float(np.median(np.abs(y[inliers]-(x[inliers]*gain+offset))))
    if (gain < .25).any() or (gain > 4).any() or (np.abs(offset) > 200).any() or error > 12:
        return dict(fallback, status='unreliable_fit')
    return dict(status='fitted', gain=gain.tolist(), offset=offset.tolist(), samples=len(x),
                inlier_fraction=float(inliers.mean()), residual_mad=error)


def _appearance(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Convert source channels to target format without copying target appearance."""
    if target.ndim == 2:
        return to_gray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if target.shape[2] == 3:
        return image[..., :3]
    return image if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)


class NormalReferenceStep:
    """Compose accepted pieces with explicit geometry, photometry and provenance."""

    def run(self, inputs: PreparedInputs, result: MatchResult) -> None:
        if not result.accepted or not result.reference_pieces:
            return
        target = inputs.target
        h, w = target.shape[:2]
        source_map = np.zeros((h, w), np.uint16)
        pieces = result.reference_pieces
        if len(pieces) > np.iinfo(np.uint16).max:
            raise ValueError('Too many normal reference fragments for uint16 source map')
        # Last accepted fragment wins, matching label/object precedence.
        for index, piece in enumerate(pieces, 1):
            # Use the same bilinear support as image warping. Nearest-neighbor
            # geometry alone can admit a mix of valid pixels and excluded text.
            support = cv2.warpPerspective(inputs.template_valids[piece.template_index].astype(np.float32),
                                          piece.matrix, (w, h))
            source_map[piece.mask & inputs.target_roi & (support >= 1.-1e-6)] = index
        reference = np.zeros_like(target)
        records = []
        # Warp one fragment at a time to avoid keeping N full RGB canvases alive.
        for index, piece in enumerate(pieces, 1):
            mask = source_map == index
            if not mask.any():
                continue
            image = _appearance(inputs.templates[piece.template_index], target)
            warped = cv2.warpPerspective(image, piece.matrix, (w, h))
            reference[mask] = warped[mask]
            records.append(dict(source_id=index, template_index=piece.template_index, kind=piece.kind,
                                matrix_template_to_target=np.asarray(piece.matrix).tolist(),
                                pixels=int(mask.sum()), photometry=fit_photometry(warped, target, mask)))
        valid = source_map > 0
        global_fit = fit_photometry(reference, target, valid)
        photometric_valid = np.zeros((h, w), bool)
        for record in records:
            mask = source_map == record['source_id']
            parameters = record['photometry']
            reliable = parameters['status'] == 'fitted'
            if not reliable and global_fit['status'] == 'fitted':
                record['photometry'] = dict(global_fit, status='global_fallback',
                                             local_failure=parameters['status'])
                parameters = record['photometry']
            elif not reliable:
                parameters = dict(parameters, status='uncorrected', local_failure=parameters['status'],
                                  gain=[1.] * len(parameters['gain']), offset=[0.] * len(parameters['offset']))
                record['photometry'] = parameters
            # Fallback is explicitly not locally reliable, even when global fit exists.
            photometric_valid[mask] = reliable
            values = reference[mask].astype(np.float64)
            if target.ndim == 2:
                values = values*parameters['gain'][0] + parameters['offset'][0]
            else:
                values[:, :3] = values[:, :3]*parameters['gain'] + parameters['offset']
            reference[mask] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        result.supposed_to_be_good = reference
        result.reference_valid = valid
        result.reference_photometric_valid = photometric_valid
        result.reference_source_map = source_map
        result.registration['normal_reference'] = dict(
            status='available' if valid.any() else 'no_verified_pixels',
            model='per_fragment_robust_affine', target='input_brightness_and_contrast',
            valid_pixels=int(valid.sum()), photometric_valid_pixels=int(photometric_valid.sum()),
            global_photometry=global_fit, pieces=records)
        # Masks are temporary; public provenance is the compact source map + records.
        result.reference_pieces = []
