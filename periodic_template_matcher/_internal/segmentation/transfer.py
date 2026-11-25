"""Steps 4–5: direct semantic transfer, then verified periodic completion."""

from dataclasses import replace

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, to_gray, warp_labels
from ..models import PreparedInputs, RegisteredMatch, TransferredLabels
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _score
from .periodic import complete_periodic
from .regions import RegionResize, component_regions


class LabelTransferStep:
    """Transfer known labels, excluding the independently located instance class."""

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            config: MatcherConfig) -> TransferredLabels:
        index = match.template_index
        assert match.matrix is not None and match.result.accepted
        transferred = self._source(inputs, index, config)
        mapped, valid = warp_labels(transferred.labels, match.matrix, inputs.target.shape,
                                    transferred.annotation_valid)
        valid &= inputs.target_roi
        mapped[~valid] = 0
        if config.label_mode == 'semantic':
            background = self._fill_direct_gaps(inputs, match, config, mapped, valid)
            transferred = replace(transferred, supplemental_background=background)
        match.result.mapped_labels, match.result.label_valid = mapped, valid
        match.result.direct_labels, match.result.direct_valid = mapped.copy(), valid.copy()
        return transferred

    @staticmethod
    def _source(inputs, index, config):
        labels = inputs.labels[index]
        assert labels is not None
        source_roi = inputs.template_valids[index]
        if inputs.annotation_valids[index] is not None:
            source_roi = source_roi & (inputs.annotation_valids[index] > 0)
        if inputs.objects is not None:
            labels = labels.copy()
            labels[labels == config.ordered_object_class] = 0
        return TransferredLabels(labels, source_roi)

    def _fill_direct_gaps(self, inputs, match, config, labels, known):
        result = match.result
        records = []
        target_features = {}
        background = None
        for index in result.registration['ranking']:
            missing = inputs.target_roi & ~known
            if not missing.any():
                break
            entry = result.registration['matches'][index]
            pose = entry.get('matrix_template_to_target')
            if (index == match.template_index or inputs.labels[index] is None
                    or not entry.get('geometry_accepted') or pose is None):
                continue
            matrix = np.asarray(pose, float)
            source = self._source(inputs, index, config)
            mapped, valid = warp_labels(source.labels, matrix, inputs.target.shape, source.annotation_valid)
            gap = valid & missing
            if not gap.any():
                continue
            fill = _verify_direct_gaps(inputs, index, matrix, gap, config, target_features,
                                       low_frequency=entry.get('evidence_mode') == 'low_frequency')
            if not fill.any():
                continue
            direct = fill
            if config.periodic_completion:
                # A secondary phase can miss a foreground annotation. Resolve
                # positive periodic evidence before accepting its background.
                negative = fill & (mapped == 0)
                if background is None:
                    background = negative
                else:
                    background |= negative
                direct = fill & (mapped > 0)
            labels[direct] = mapped[direct]
            known |= direct
            if config.periodic_completion:
                if result.completed_template is None:
                    result.completed_template = result.warped_template.copy()
                    result.completed_valid = result.geometric_valid.copy()
                image_fill = fill & ~result.completed_valid
                if image_fill.any():
                    image = cv2.warpPerspective(inputs.templates[index], matrix, gap.shape[::-1])
                    result.completed_template[image_fill] = image[image_fill]
                    result.completed_valid |= image_fill
            records.append(dict(template_index=index, added_label_pixels=int(direct.sum())))
        if records:
            result.registration['direct_sources'] = records
        return background


def _verify_direct_gaps(inputs, index, matrix, gap, cfg, target_features, *, low_frequency):
    """Use the registered pose and visible context around each uncovered region."""
    source, target = inputs.templates[index], inputs.target
    h, w = target.shape[:2]
    sh, sw = source.shape[:2]
    scale = cfg.matching_scale(source.shape, target.shape)
    ss, ts = (round(sw*scale), round(sh*scale)), (round(w*scale), round(h*scale))
    if min(*ss, *ts) < 32:
        return np.zeros_like(gap)
    Da, Db = np.diag([ss[0]/sw, ss[1]/sh, 1.]), np.diag([ts[0]/w, ts[1]/h, 1.])
    src = _features(*ValidRegion.resize_for_matching(to_gray(source), inputs.template_valids[index], ss),
                    low_frequency=low_frequency, cfg=cfg)
    key = ts, low_frequency
    if key not in target_features:
        target_features[key] = _features(
            *ValidRegion.resize_for_matching(to_gray(target), inputs.target_roi, ts),
            low_frequency=low_frequency, cfg=cfg)
    dst = target_features[key]
    work = Db @ matrix @ np.linalg.inv(Da)
    warp = ScoreWarp(src, work, ts)
    resize = RegionResize((h, w), ts)
    local_cfg = replace(cfg, min_overlap=cfg.local_min_overlap)
    ncc = max(cfg.periodic_min_ncc, cfg.periodic_low_frequency_min_ncc) if low_frequency else cfg.periodic_min_ncc
    edge = min(cfg.periodic_min_edge, cfg.periodic_low_frequency_edge_cap) if low_frequency else cfg.periodic_min_edge
    approved = np.zeros_like(gap)
    context = np.ones((cfg.local_context_size,)*2, np.uint8)
    for region in component_regions(gap):
        roi = cv2.dilate(resize(region), context) & dst[3]
        metrics = _score(src, (*dst[:3], roi), work, local_cfg, warped=warp, min_structure_ncc=ncc)
        if metrics['structure_ncc'] >= ncc and metrics['edge_alignment'] >= edge:
            approved[region.slices] |= region.mask
    return approved


class PeriodicCompletionStep:
    """Complete only accepted semantic transfers using verified source periods.

    Period inference, per-class repeat axes, ECC and local copy validation are
    owned by the existing ``complete_periodic`` numerical kernel.
    """

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            transferred: TransferredLabels, config: MatcherConfig) -> None:
        if config.periodic_completion and config.label_mode == 'semantic':
            complete_periodic(match.result, transferred.labels, transferred.annotation_valid,
                              inputs.template_valids[match.template_index], inputs.target_roi, config)
