"""Matcher configuration and validation, without matching or file IO logic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isfinite
from typing import Any


OPTION_GROUPS = {
    'step1': {'scale', 'max_side', 'bottom_ignore_fraction', 'valid_height'},
    'step2': {'backend', 'angles', 'scales', 'peaks_per_hypothesis', 'seed'},
    'step3': {'refine_candidates', 'ecc_iterations', 'edge_tolerance_px',
              'min_overlap', 'min_score', 'min_template_gap', 'min_candidate_gap'},
    'periodic': {'periodic_completion', 'period_vectors', 'periodic_anchor_class',
                 'periodic_max_steps', 'periodic_min_ncc', 'periodic_min_edge'},
    'evaluation': {'label_mode', 'include_background'},
}


@dataclass
class MatcherConfig:
    """Registration parameters, identity policy and segmentation evaluation."""

    angles: Sequence[float] = (-2.0, 0.0, 2.0)
    scales: Sequence[float] = (0.97, 1.0, 1.03)
    max_side: int = 800
    peaks_per_hypothesis: int = 4
    refine_candidates: int = 5
    min_overlap: float = 0.45
    min_score: float = 60.0
    min_template_gap: float = 3.0
    min_candidate_gap: float = 2.0
    bottom_ignore_fraction: float = 0.0
    valid_height: int | None = None  # Default original-pixel height for all images.
    edge_tolerance_px: float = 2.0  # Working-resolution pixels, not native pixels.
    ecc_iterations: int = 70
    seed: int = 7
    label_mode: str = 'semantic'
    include_background: bool = False
    backend: str = 'fft'
    periodic_completion: bool = False
    period_vectors: Sequence[Sequence[float]] = ()  # Optional source-original pixel lattice vectors.
    periodic_anchor_class: int = 1
    periodic_max_steps: int = 2
    periodic_min_ncc: float = .75
    periodic_min_edge: float = .6
    scale: float | None = None  # Shared working resolution; None uses max_side.

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> MatcherConfig:
        """Read grouped JSON options while accepting legacy flat parameters."""
        if not isinstance(options, dict):
            raise ValueError('options must be an object')
        flat = {}
        names = {field.name for field in fields(cls)}
        for key, value in options.items():
            if key in OPTION_GROUPS:
                if not isinstance(value, dict):
                    raise ValueError(f'options.{key} must be an object')
                entries = value.items()
            elif key in names:
                entries = [(key, value)]
            else:
                raise ValueError(f'Unknown options key: {key}')
            for name, setting in entries:
                if key in OPTION_GROUPS and name == 'description':
                    continue  # Human-readable group documentation, not a matcher parameter.
                if key in OPTION_GROUPS and name not in OPTION_GROUPS[key]:
                    raise ValueError(f'Parameter {name} does not belong in options.{key}')
                if name in flat:
                    raise ValueError(f'Duplicate options parameter: {name}')
                flat[name] = setting
        cfg = cls(**flat)
        cfg.validate()
        return cfg

    def matching_scale(self, *shapes: Sequence[int]) -> float:
        """Choose one nominal scale for all template and input image shapes."""
        return self.scale if self.scale is not None else min(
            1.0, self.max_side / max(max(shape[:2]) for shape in shapes))

    def validate(self) -> None:
        if self.scale is not None and (type(self.scale) not in (int, float)
                                      or not isfinite(self.scale) or not 0 < self.scale <= 1):
            raise ValueError('scale must be finite in (0, 1] or None')
        if self.valid_height is not None and (type(self.valid_height) is not int or self.valid_height < 1):
            raise ValueError('valid_height must be a positive integer or None')
        if not isinstance(self.periodic_completion, bool):
            raise ValueError('periodic_completion must be bool')
        if type(self.periodic_max_steps) is not int or not 1 <= self.periodic_max_steps <= 5:
            raise ValueError('periodic_max_steps must be an integer in 1..5')
        if type(self.periodic_anchor_class) is not int or not 1 <= self.periodic_anchor_class <= 65535:
            raise ValueError('periodic_anchor_class must be in 1..65535')
        if not 0 <= self.periodic_min_ncc <= 1 or not 0 <= self.periodic_min_edge <= 1:
            raise ValueError('periodic similarity thresholds must be in 0..1')
        if not isinstance(self.period_vectors, (tuple, list)) or len(self.period_vectors) > 2:
            raise ValueError('period_vectors must contain at most two source-pixel vectors')
        for vector in self.period_vectors:
            if (not isinstance(vector, (tuple, list)) or len(vector) != 2
                    or not all(isinstance(v, (int, float)) and isfinite(v) for v in vector)
                    or sum(v*v for v in vector) < 1):
                raise ValueError('period vectors must be finite nonzero [dx, dy] pairs')
        if len(self.period_vectors) == 2:
            a, b = self.period_vectors
            if abs(a[0]*b[1]-a[1]*b[0]) < .05 * sum(v*v for v in a)**.5 * sum(v*v for v in b)**.5:
                raise ValueError('period vectors must be independent')
        if self.backend not in ('fft', 'opencv'):
            raise ValueError('backend must be fft or opencv')
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
