"""Matcher configuration and validation, without matching or file IO logic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from math import isfinite
from typing import Any


OPTION_GROUPS = {
    'objects': {'ordered_object_class', 'object_cycle', 'object_min_score', 'object_min_gap',
                'object_min_visible', 'object_candidates', 'object_spacing_weight', 'object_order_axis',
                'object_visible_count'},
    'step1': {'scale', 'max_side', 'valid_height',
              'valid_x_start', 'valid_x_end', 'valid_y_start', 'valid_y_end'},
    'step2': {'backend', 'angles', 'scales', 'peaks_per_hypothesis', 'seed', 'template_workers'},
    'step3': {'refine_candidates', 'ecc_iterations', 'ecc_roi', 'edge_tolerance_px',
              'min_overlap', 'min_score', 'min_template_gap', 'min_candidate_gap'},
    'periodic': {'periodic_completion', 'period_vectors', 'periodic_anchor_class',
                 'periodic_max_steps', 'periodic_min_ncc', 'periodic_min_edge', 'class_periodicity',
                 'shared_periods'},
    'evaluation': {'label_mode', 'include_background'},
}


def _class_options(classes: Any) -> dict[str, Any]:
    """Normalize readable per-class JSON rules to the existing runtime API."""
    if not isinstance(classes, dict) or not classes:
        raise ValueError('options.classes must be a nonempty object keyed by class ID')
    policies, matching = {}, {}
    parameters = {'cycle': 'object_cycle', 'visible_count': 'object_visible_count', 'min_score': 'object_min_score',
                  'min_gap': 'object_min_gap', 'min_visible': 'object_min_visible',
                  'candidates': 'object_candidates', 'spacing_weight': 'object_spacing_weight'}
    for key, rule in classes.items():
        if (type(key) not in (str, int) or not str(key).isascii() or not str(key).isdigit()
                or str(int(key)) != str(key) or not 1 <= int(key) <= 65535):
            raise ValueError('options.classes keys must be canonical class IDs in 1..65535')
        class_id = str(key)
        if class_id in policies:
            raise ValueError('options.classes contains duplicate class IDs')
        if (not isinstance(rule, dict) or 'repeat_axes' not in rule
                or set(rule) - {'repeat_axes', 'instance_matching', 'description'}):
            raise ValueError('Each class rule requires repeat_axes and may contain instance_matching/description')
        policies[class_id] = rule['repeat_axes']
        if 'instance_matching' not in rule:
            continue
        settings = rule['instance_matching']
        if (not isinstance(settings, dict) or 'cycle' not in settings
                or set(settings) - (set(parameters) | {'description'})):
            raise ValueError('instance_matching requires cycle and only supported matching parameters')
        if matching:
            raise ValueError('Only one class with instance_matching is currently supported')
        matching['ordered_object_class'] = int(key)
        matching.update({parameters[k]: v for k, v in settings.items() if k != 'description'})
    return dict(class_periodicity=policies, **matching)


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
    valid_height: int | None = None  # Default original-pixel height for all images.
    edge_tolerance_px: float = 2.0  # Working-resolution pixels, not native pixels.
    ecc_iterations: int = 70
    ecc_roi: bool = False  # Opt-in padded local ECC; floating-point results may differ.
    seed: int = 7
    label_mode: str = 'semantic'
    include_background: bool = False
    backend: str = 'fft'
    template_workers: int = 0  # 0: auto, up to four templates; 1: serial registration.
    periodic_completion: bool = False
    shared_periods: bool = False  # Jointly verified translation lattice; no per-copy affine drift.
    period_vectors: Sequence[Sequence[float]] = ()  # Optional source-original pixel lattice vectors.
    periodic_anchor_class: int = 1
    periodic_max_steps: int = 2
    periodic_min_ncc: float = .75
    periodic_min_edge: float = .6
    # Empty mapping keeps legacy defaults; otherwise missing classes cannot repeat.
    # Axes describe source-image translation symmetry, NOT counts or pixel periods.
    class_periodicity: dict[str | int, Sequence[str]] = field(default_factory=dict)
    scale: float | None = None  # Shared working resolution; None uses max_side.
    ordered_object_class: int | None = None
    object_order_axis: str = 'x'
    object_cycle: Sequence[str] = ('A', 'B', 'C')
    object_visible_count: int | None = None  # None preserves complete-cycle matching.
    object_min_score: float = .7
    object_min_gap: float = .03
    object_min_visible: float = .15
    object_candidates: int = 8
    object_spacing_weight: float = .3
    valid_x_start: int = 0
    valid_x_end: int | None = None
    valid_y_start: int = 0
    valid_y_end: int | None = None

    @classmethod
    def from_options(cls, options: dict[str, Any], *, object_order_axis: str | None = None) -> MatcherConfig:
        """Read grouped JSON options while accepting legacy flat parameters."""
        if not isinstance(options, dict):
            raise ValueError('options must be an object')
        flat = {}
        names = {field.name for field in fields(cls)}
        for key, value in options.items():
            if key == 'classes':
                continue  # Validated and normalized after legacy fields are collected.
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
        if 'classes' in options:
            if set(flat) & (OPTION_GROUPS['objects'] | {'class_periodicity'}):
                raise ValueError('Do not mix options.classes with legacy objects/class_periodicity parameters')
            flat.update(_class_options(options['classes']))
        if object_order_axis is not None:
            if 'object_order_axis' in flat and flat['object_order_axis'] != object_order_axis:
                raise ValueError('Configured object_order_axis conflicts with template layouts')
            flat['object_order_axis'] = object_order_axis
        cfg = cls(**flat)
        cfg.validate()
        return cfg

    def matching_scale(self, *shapes: Sequence[int]) -> float:
        """Choose one nominal scale for all template and input image shapes."""
        return self.scale if self.scale is not None else min(
            1.0, self.max_side / max(max(shape[:2]) for shape in shapes))

    def without_valid_region(self) -> MatcherConfig:
        """The registration stage receives inputs whose ROI has already been resolved."""
        return replace(self, valid_height=None, valid_x_start=0, valid_x_end=None,
                       valid_y_start=0, valid_y_end=None)

    def periodic_axes(self, class_id: int) -> tuple[str, ...]:
        """Allowed source axes; an explicit empty list means no periodic copies."""
        if not self.class_periodicity:
            return (self.object_repeat_axis,) if class_id == self.ordered_object_class else ('x', 'y')
        axes = self.class_periodicity.get(str(class_id), self.class_periodicity.get(class_id, ()))
        return tuple(axis for axis in ('x', 'y') if axis in axes)

    @property
    def object_repeat_axis(self) -> str:
        """Support can repeat perpendicular to the ordered identity layout."""
        return 'y' if self.object_order_axis == 'x' else 'x'

    def allows_periodic_shift(self, class_id: int, vectors: Sequence[Sequence[float]]) -> bool:
        """Check every active lattice generator, allowing small registration skew.

        A sum of forbidden diagonal generators cannot manufacture an allowed axis.
        Two-dimensional policies also support explicitly configured oblique lattices.
        """
        axes = self.periodic_axes(class_id)
        if not axes:
            return False
        if len(axes) == 2:
            return True
        axis = 0 if axes[0] == 'x' else 1
        return all(abs(v[axis]) > 0 and abs(v[1-axis]) <= .15*abs(v[axis]) for v in vectors)

    def validate(self) -> None:
        if type(self.ecc_roi) is not bool:
            raise ValueError('ecc_roi must be bool')
        if type(self.shared_periods) is not bool:
            raise ValueError('shared_periods must be bool')
        for axis, start, end in (('x', self.valid_x_start, self.valid_x_end),
                                 ('y', self.valid_y_start, self.valid_y_end)):
            if type(start) is not int or start < 0:
                raise ValueError(f'valid_{axis}_start must be a nonnegative integer')
            if end is not None and (type(end) is not int or end <= start):
                raise ValueError(f'valid_{axis}_end must be an integer greater than valid_{axis}_start')
        if type(self.template_workers) is not int or self.template_workers < 0:
            raise ValueError('template_workers must be a nonnegative integer (0=auto, 1=serial)')
        if self.object_order_axis not in ('x', 'y'):
            raise ValueError('object_order_axis must be "x" or "y"')
        if not isinstance(self.class_periodicity, dict):
            raise ValueError('class_periodicity must be an object mapping class IDs to axis lists')
        seen = set()
        for key, axes in self.class_periodicity.items():
            if type(key) is int:
                class_id = key
            elif isinstance(key, str) and key.isascii() and key.isdigit() and str(int(key)) == key:
                class_id = int(key)
            else:
                raise ValueError('class_periodicity keys must be canonical class IDs in 1..65535')
            if not 1 <= class_id <= 65535 or class_id in seen:
                raise ValueError('class_periodicity keys must be unique class IDs in 1..65535')
            seen.add(class_id)
            if (not isinstance(axes, (list, tuple)) or any(not isinstance(a, str) or a not in ('x', 'y') for a in axes)
                    or len(set(axes)) != len(axes)):
                raise ValueError('class_periodicity axes must be [], ["x"], ["y"], or ["x", "y"]')
        if self.ordered_object_class is not None and self.label_mode != 'semantic':
            raise ValueError('Ordered object matching requires semantic label mode')
        if self.ordered_object_class is not None and (type(self.ordered_object_class) is not int
                                                      or not 1 <= self.ordered_object_class <= 65535):
            raise ValueError('ordered_object_class must be a class ID or None')
        if (self.ordered_object_class is not None
                and self.object_order_axis in self.periodic_axes(self.ordered_object_class)):
            direction = 'horizontal' if self.object_order_axis == 'x' else 'vertical'
            raise ValueError(f'ordered object matching does not support {direction} identity repetition along order_axis')
        if (not isinstance(self.object_cycle, (list, tuple)) or not 1 <= len(self.object_cycle) <= 65535
                or any(not isinstance(v, str) or not v for v in self.object_cycle)
                or len(set(self.object_cycle)) != len(self.object_cycle)):
            raise ValueError('object_cycle must contain 1..65535 unique nonempty identity names')
        if self.object_visible_count is not None and (
                type(self.object_visible_count) is not int
                or not 1 <= self.object_visible_count <= len(self.object_cycle)):
            raise ValueError('object_visible_count must be an integer in 1..len(object_cycle) or None')
        if not isfinite(self.object_spacing_weight) or not 0 <= self.object_spacing_weight <= 1:
            raise ValueError('object_spacing_weight must be finite in [0, 1]')
        if (not 0 < self.object_min_visible <= 1 or not 0 <= self.object_min_score <= 1
                or not 0 <= self.object_min_gap <= 1):
            raise ValueError('invalid object score, margin or visibility threshold')
        if type(self.object_candidates) is not int or not 1 <= self.object_candidates <= 16:
            raise ValueError('object_candidates must be an integer in 1..16')
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
        if not 0 < self.min_overlap <= 1:
            raise ValueError('min_overlap must be in (0, 1]')
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
