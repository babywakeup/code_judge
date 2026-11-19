"""Shared result containers and per-call step contracts, in original pixels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..utils.types import BoolMask, Diagnostics, Image, LabelMap, Mask, Matrix


@dataclass
class ReferencePiece:
    """Exact accepted source geometry and target pixels, before photometry."""

    template_index: int
    matrix: Matrix
    mask: BoolMask
    kind: str


@dataclass
class MatchResult:
    """Registration metadata plus target-sized arrays, with no file IO.

    mapped_labels/label_valid exist only for an accepted, annotated match.
    Saving a rejected result retains its input and status without a prediction.
    """

    registration: Diagnostics
    input_image: Image
    template_image: Image
    warped_template: Image | None
    geometric_valid: BoolMask
    mapped_labels: LabelMap | None = None
    label_valid: BoolMask | None = None
    completed_template: Image | None = None
    completed_valid: BoolMask | None = None
    periodic_valid: BoolMask | None = None
    label_periodic_valid: BoolMask | None = None
    object_identities: LabelMap | None = None
    object_valid: BoolMask | None = None
    base_labels: LabelMap | None = None
    base_valid: BoolMask | None = None
    supposed_to_be_good: Image | None = None
    reference_valid: BoolMask | None = None
    reference_photometric_valid: BoolMask | None = None
    reference_source_map: LabelMap | None = None
    reference_pieces: list[ReferencePiece] = field(default_factory=list, repr=False)

    @property
    def accepted(self) -> bool:
        return self.registration['accepted']

    @property
    def best_template_index(self) -> int | None:
        return self.registration['best_template_index']

    @property
    def match_score(self) -> float:
        return self.registration['matches'][self.registration['ranking'][0]]['match_score']

    @property
    def matrix_template_to_target(self) -> Matrix | None:
        best = self.registration['matches'][self.registration['ranking'][0]]
        matrix = best.get('matrix_template_to_target')
        return None if matrix is None else np.asarray(matrix, np.float64)


@dataclass
class EvaluationResult:
    match: MatchResult
    metrics: Diagnostics
    ground_truth: LabelMap
    ground_truth_valid: BoolMask

    @property
    def score(self) -> float | None:
        return self.metrics['score']


@dataclass
class PreparedInputs:
    """Validated arrays; image ROI and annotation knowledge remain independent."""

    templates: Sequence[Image]
    labels: list[LabelMap | None]
    target: Image
    template_valids: list[Mask]
    annotation_valids: Sequence[Mask | None]
    target_valid: Mask
    regions: Diagnostics
    objects: list[list[Diagnostics]] | None

    @property
    def target_roi(self) -> BoolMask:
        return self.target_valid > 0


@dataclass
class RegisteredMatch:
    """Selected source and direct geometric result, even if selection rejects it."""

    result: MatchResult
    template_index: int
    matrix: Matrix | None


@dataclass
class TransferredLabels:
    """Source labels/knowledge used by both direct and periodic transfer.

    The independently matched instance class is removed from these labels, not
    from caller-owned arrays. ``annotation_valid`` already includes image ROI.
    """

    labels: LabelMap
    annotation_valid: Mask
