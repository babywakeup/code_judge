"""Result containers shared by the matcher and presentation layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.types import BoolMask, Diagnostics, Image, LabelMap, Matrix


@dataclass
class MatchResult:
    """Registration metadata plus target-sized arrays, with no file IO.

    mapped_labels/label_valid exist only for an accepted, annotated match.
    The top-ranked template can still be visualized when rejected.
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
