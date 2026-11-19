"""Steps 4–5: direct semantic transfer, then verified periodic completion."""

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import warp_labels, warp_valid_mask
from ..models import MatchResult, PreparedInputs, ReferencePiece, RegisteredMatch, TransferredLabels
from .periodic import complete_periodic


class LabelTransferStep:
    """Transfer known labels, excluding the independently located instance class."""

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            config: MatcherConfig) -> TransferredLabels:
        index = match.template_index
        labels = inputs.labels[index]
        assert match.matrix is not None and match.result.accepted and labels is not None
        source_roi = inputs.template_valids[index].copy()
        if inputs.annotation_valids[index] is not None:
            source_roi &= (inputs.annotation_valids[index] > 0).astype(np.uint8)
        if inputs.objects is not None:
            labels = labels.copy()
            labels[labels == config.ordered_object_class] = 0
        mapped, valid = warp_labels(labels, match.matrix, inputs.target.shape, source_roi)
        valid &= inputs.target_roi
        mapped[~valid] = 0
        match.result.mapped_labels, match.result.label_valid = mapped, valid
        match.result.reference_pieces.append(ReferencePiece(index, match.matrix.copy(),
            match.result.geometric_valid.copy() & inputs.target_roi, 'direct'))
        return TransferredLabels(labels, source_roi)


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
            self._complete_from_other_sources(inputs, match, config)

    def _complete_from_other_sources(self, inputs, match, config):
        result = match.result
        records = []
        for index in result.registration['ranking']:
            if result.label_valid[inputs.target_roi].all():
                break
            candidate = result.registration['matches'][index]
            if (index == match.template_index or inputs.labels[index] is None
                    or not candidate.get('geometry_accepted', False)
                    or candidate.get('matrix_template_to_target') is None):
                continue
            matrix = np.asarray(candidate['matrix_template_to_target'], float)
            source = inputs.templates[index]
            size = (inputs.target.shape[1], inputs.target.shape[0])
            registration = dict(result.registration, ranking=[index], best_template_index=index)
            registration.pop('periodic_completion', None)
            other = MatchResult(registration, inputs.target, source,
                cv2.warpPerspective(source, matrix, size),
                warp_valid_mask(inputs.template_valids[index], matrix, inputs.target.shape) & inputs.target_roi)
            alternate = RegisteredMatch(other, index, matrix)
            transferred = LabelTransferStep().run(inputs, alternate, config)
            complete_periodic(other, transferred.labels, transferred.annotation_valid,
                              inputs.template_valids[index], inputs.target_roi, config)
            fill = other.label_valid & inputs.target_roi & ~result.label_valid
            for piece in other.reference_pieces:
                piece.mask &= fill
                if piece.mask.any():
                    result.reference_pieces.append(piece)
            result.mapped_labels[fill] = other.mapped_labels[fill]
            result.label_valid |= fill
            result.label_periodic_valid |= other.label_periodic_valid & fill
            result.completed_template[fill] = other.completed_template[fill]
            result.completed_valid |= fill & other.completed_valid
            result.periodic_valid |= fill & other.periodic_valid
            records.append(dict(template_index=index, added_label_pixels=int(fill.sum()),
                                periodic_completion=other.registration['periodic_completion']))
        if records:
            missing = int((inputs.target_roi & ~result.label_valid).sum())
            result.registration['periodic_completion'].update(
                supplemental_templates=records, remaining_unknown_pixels=missing,
                label_coverage_roi=float(result.label_valid.sum()/inputs.target_roi.sum()),
                status='complete' if missing == 0 else 'incomplete')
