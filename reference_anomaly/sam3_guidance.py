"""Array-only, fixed reference-evidence policy for interactive SAM3.

No image/annotation IO, model imports, GT metrics or output-mask modifications.
Unknown and intermediate-confidence reference pixels contribute no evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


GUIDANCE_CONFIG = dict(
    policy='reference_anomaly_core_near_normal_v1',
    foreground_threshold=3., background_threshold=1., reliable_confidence=255,
    connectivity=8, max_components=16, interior_fraction=.5,
    positive_points_per_component=3, inside_negative_points=8, outside_negative_points=8,
    negative_safety_distance=3., negative_radius_fraction=.1, min_negative_radius=8,
    min_core_coverage=.5, max_normal_fraction=.5,
    candidate_order='core_coverage_minus_normal_fraction_then_model_quality_then_index',
    prompt_grouping='one_joint_prompt_per_detection',
    unknown_reference_role='neutral_not_background',
)


@dataclass(frozen=True)
class GuidanceEvidence:
    core: np.ndarray
    normal: np.ndarray
    near_normal: np.ndarray
    components: np.ndarray
    component_ids: list[int]
    component_count: int


def guidance_evidence(score: np.ndarray, reliable: np.ndarray,
                      inside: np.ndarray) -> GuidanceEvidence:
    """Derive crop-local evidence from already validated score/reliability arrays.

    `inside` denotes the ORIGINAL detector, not the padded crop. Core IDs use
    OpenCV's 8-connected labels. Normal evidence is not restricted to the box.
    """
    if (score.ndim != 2 or score.shape != reliable.shape or score.shape != inside.shape
            or reliable.dtype != np.bool_ or inside.dtype != np.bool_):
        raise ValueError('Guidance requires same-size score, bool reliable and bool inside arrays')
    high = reliable & (score >= GUIDANCE_CONFIG['foreground_threshold'])
    count, components = cv2.connectedComponents(
        (high & inside).astype(np.uint8), connectivity=GUIDANCE_CONFIG['connectivity'])
    peaks = np.zeros(count, dtype=score.dtype)
    np.maximum.at(peaks, components.ravel(), score.ravel())
    groups = sorted(range(1, count), key=lambda k: (-peaks[k].item(), k))
    selected = groups[:GUIDANCE_CONFIG['max_components']]
    core = np.zeros(score.shape, bool)
    for group in selected:
        depth = cv2.distanceTransform(np.pad((components == group).astype(np.uint8), 1),
                                      cv2.DIST_L2, 5)[1:-1, 1:-1]
        cutoff = max(1., float(depth.max()) * GUIDANCE_CONFIG['interior_fraction'])
        core |= (components == group) & (depth >= cutoff)
    # Protect ALL anomaly regions, including outside this detection and those
    # beyond the positive-component budget, from being used as negative points.
    distance = cv2.distanceTransform((~high).astype(np.uint8), cv2.DIST_L2, 5)
    normal = (reliable & (score <= GUIDANCE_CONFIG['background_threshold'])
              & (distance >= GUIDANCE_CONFIG['negative_safety_distance']))
    _, _, box_width, box_height = cv2.boundingRect(inside.astype(np.uint8))
    radius = max(GUIDANCE_CONFIG['min_negative_radius'], math.ceil(
        min(box_width, box_height) * GUIDANCE_CONFIG['negative_radius_fraction']))
    return GuidanceEvidence(core=core, normal=normal, near_normal=normal & (distance <= radius),
                            components=components, component_ids=selected, component_count=len(groups))


def choose_guided_candidate(masks: np.ndarray, quality: np.ndarray, records: list[dict],
                            evidence: GuidanceEvidence):
    """Score validated SAM alternatives; select or abstain, NEVER edit a mask.

    `records` come from the normal binary/finite/point-consistency validation.
    The denominator excludes unknown, intermediate and safety-zone pixels, so
    adding such pixels cannot improve OR worsen an evidence score.
    """
    core_pixels = int(evidence.core.sum())
    accepted = []
    for index, (mask, record) in enumerate(zip(masks, records)):
        binary = np.asarray(mask, bool)
        core_overlap = int((binary & evidence.core).sum())
        normal_overlap = int((binary & evidence.normal).sum())
        coverage = core_overlap / core_pixels if core_pixels else 0.
        domain_pixels = core_overlap + normal_overlap
        normal_fraction = normal_overlap / domain_pixels if domain_pixels else 0.
        reasons = []
        if not record['nonempty']:
            reasons.append('empty_mask')
        if not record['prompt_consistent']:
            reasons.append('point_conflict')
        if not core_pixels or coverage < GUIDANCE_CONFIG['min_core_coverage']:
            reasons.append('insufficient_core_coverage')
        if normal_fraction > GUIDANCE_CONFIG['max_normal_fraction']:
            reasons.append('excess_reliable_normal')
        record.update(core_pixels=core_pixels, covered_core_pixels=core_overlap,
                      reliable_normal_pixels=normal_overlap, core_coverage=coverage,
                      normal_fraction=normal_fraction, evidence_score=coverage-normal_fraction,
                      evidence_accepted=not reasons, rejection_reasons=reasons)
        if not reasons:
            accepted.append(index)
    selected = max(accepted, key=lambda i: (records[i]['evidence_score'], float(quality[i]), -i)) if accepted else None
    return (np.asarray(masks[selected], bool) if selected is not None else None), selected, records