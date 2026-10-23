"""Matcher configuration and validation, without matching or file IO logic."""
from dataclasses import dataclass
from math import isfinite


@dataclass
class MatcherConfig:
    """Registration parameters, identity policy and segmentation evaluation."""

    angles: tuple = (-2.0, 0.0, 2.0)
    scales: tuple = (0.97, 1.0, 1.03)
    max_side: int = 800
    peaks_per_hypothesis: int = 4
    refine_candidates: int = 5
    min_overlap: float = 0.45
    min_score: float = 60.0
    min_template_gap: float = 3.0
    min_candidate_gap: float = 2.0
    bottom_ignore_fraction: float = 0.0
    edge_tolerance_px: float = 2.0  # Working-resolution pixels, not native pixels.
    ecc_iterations: int = 70
    seed: int = 7
    label_mode: str = 'semantic'
    include_background: bool = False

    def validate(self):
        if self.max_side < 96 or not self.angles or not self.scales:
            raise ValueError('max_side >= 96 and nonempty angles/scales required')
        if any(s <= 0 for s in self.scales):
            raise ValueError('scales must be positive')
        if not 0 < self.min_overlap <= 1 or not 0 <= self.bottom_ignore_fraction < 0.5:
            raise ValueError('Invalid overlap or bottom-ignore fraction')
        if self.edge_tolerance_px <= 0 or min(self.peaks_per_hypothesis, self.refine_candidates, self.ecc_iterations) < 1:
            raise ValueError('Tolerance and iteration/candidate counts must be positive')
        if self.label_mode not in ('semantic', 'instance'):
            raise ValueError('label_mode must be semantic or instance')
        if not isinstance(self.include_background, bool):
            raise ValueError('include_background must be bool')
        if not all(isfinite(v) for v in (*self.angles, *self.scales)):
            raise ValueError('angles and scales must be finite')
        if not 0 <= self.min_score <= 100:
            raise ValueError('min_score must be in 0..100')
        if not 0 <= self.min_template_gap <= 100 or not 0 <= self.min_candidate_gap <= 100:
            raise ValueError('score gaps must be in 0..100')