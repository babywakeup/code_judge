"""Result containers shared by the matcher and presentation layer."""
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    """Registration metadata plus target-sized arrays, with no file IO.

    mapped_labels/label_valid exist only for an accepted, annotated match.
    The top-ranked template can still be visualized when rejected.
    """

    registration: dict
    input_image: np.ndarray
    template_image: np.ndarray
    warped_template: np.ndarray | None
    geometric_valid: np.ndarray
    mapped_labels: np.ndarray | None = None
    label_valid: np.ndarray | None = None

    @property
    def accepted(self):
        return self.registration['accepted']

    @property
    def best_template_index(self):
        return self.registration['best_template_index']

    @property
    def match_score(self):
        return self.registration['matches'][self.registration['ranking'][0]]['match_score']

    @property
    def matrix_template_to_target(self):
        best = self.registration['matches'][self.registration['ranking'][0]]
        matrix = best.get('matrix_template_to_target')
        return None if matrix is None else np.asarray(matrix, np.float64)


@dataclass
class EvaluationResult:
    match: MatchResult
    metrics: dict
    ground_truth: np.ndarray
    ground_truth_valid: np.ndarray

    @property
    def score(self):
        return self.metrics['score']