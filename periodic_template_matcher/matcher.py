"""Public array API: select templates, transfer labels, evaluate and inspect."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from ._internal.evaluation import evaluate_segmentation
from ._internal.periodic_completion import complete_periodic
from ._internal.registration import select_templates
from ._internal.results import EvaluationResult, MatchResult
from ._internal.result_io import save_result
from .config import MatcherConfig
from .utils import ValidRegion, normalize_masks, validate_image, warp_labels, warp_valid_mask
from .utils.types import Diagnostics, Image, LabelInput, Mask, PathInput


class PeriodicTemplateMatcher:
    """Array-based facade for registration, mask transfer and inspection.

    templates: nonempty list of uint8 gray/BGR/BGRA photographs.
    masks: one entry per template: HxW integer IDs, disjoint binary class masks
           (classes 1..K), or None. Label knowledge is separate from image ROI.
    input: a single target image. Duplicate array candidates are not collapsed.
    """

    def __init__(self, config: MatcherConfig | None = None) -> None:
        if config is not None and not isinstance(config, MatcherConfig):
            raise TypeError('config must be MatcherConfig')
        self.config = deepcopy(config) if config is not None else MatcherConfig()
        self.config.validate()

    @staticmethod
    def _valids(values: Sequence[Mask | None] | None, images: Sequence[Image],
                name: str) -> Sequence[Mask | None]:
        """Check optional per-template masks; None entries mean no extra restriction."""
        if values is None:
            return [None] * len(images)
        if not isinstance(values, Sequence) or len(values) != len(images):
            raise ValueError(f'{name} must have one entry per template')
        for mask, image in zip(values, images):
            if mask is not None and (not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]):
                raise ValueError(f'{name} entries must match image height and width')
        return values

    def match(self, templates: Sequence[Image], masks: Sequence[LabelInput | None],
              input: Image, *, template_valids: Sequence[Mask | None] | None = None,
              annotation_valids: Sequence[Mask | None] | None = None,
              target_valid: Mask | None = None,
              template_valid_heights: Sequence[int | None] | None = None,
              target_valid_height: int | None = None) -> MatchResult:
        """Select a template and transfer its labels only on accepted matches.

        template_valids/target_valid limit image ROI. annotation_valids mark
        known template labels; zero is otherwise known background. Explicit valid
        heights are original-pixel top-row counts, overriding default exclusions.
        """
        self.config.validate()
        if not isinstance(templates, Sequence) or not templates:
            raise ValueError('templates must be a nonempty list of images')
        for image in [*templates, input]:
            if not isinstance(image, np.ndarray):
                raise ValueError('Images must be numpy arrays; use read_image for file paths')
            validate_image(image)
        if not isinstance(masks, Sequence) or len(masks) != len(templates):
            raise ValueError('masks must have one entry per template (None is allowed)')
        labels = [None if mask is None else normalize_masks(mask, image.shape[:2], f'masks[{i}]')
                  for i, (mask, image) in enumerate(zip(masks, templates))]
        valids = self._valids(template_valids, templates, 'template_valids')
        annotations = self._valids(annotation_valids, templates, 'annotation_valids')
        # Region construction also checks the target mask's shape.
        valids, target_mask, regions = self._prepare_regions(
            templates, input, valids, target_valid, template_valid_heights, target_valid_height)
        # Heights and masks are already resolved. Do not exclude rows a second time.
        registration_config = replace(self.config, valid_height=None, bottom_ignore_fraction=0.)
        registration = select_templates(templates, input, registration_config, valids,
                                        target_mask, self.config.label_mode)
        registration.update(config=asdict(self.config), valid_regions=regions)
        index = registration['ranking'][0]
        source = templates[index]
        best = registration['matches'][index]
        H = best.get('matrix_template_to_target')
        result = MatchResult(registration, input, source, None, np.zeros(input.shape[:2], bool))
        if H is None:
            return result
        H = np.asarray(H, np.float64)
        target_roi = target_mask > 0
        source_roi = valids[index].copy()
        result.warped_template = cv2.warpPerspective(source, H, (input.shape[1], input.shape[0]))
        result.geometric_valid = warp_valid_mask(source_roi, H, input.shape)
        result.geometric_valid &= target_roi
        if result.accepted and labels[index] is not None:
            if annotations[index] is not None:
                source_roi &= (annotations[index] > 0).astype(np.uint8)
            mapped, valid = warp_labels(labels[index], H, input.shape, source_roi)
            valid &= target_roi
            mapped[~valid] = 0
            result.mapped_labels, result.label_valid = mapped, valid
            if self.config.periodic_completion and self.config.label_mode == 'semantic':
                complete_periodic(result, labels[index], source_roi,
                                  valids[index],
                                  target_roi, self.config)
        return result

    def _prepare_regions(self, templates: Sequence[Image], input: Image,
                         valids: Sequence[Mask | None], target_valid: Mask | None,
                         template_valid_heights: Sequence[int | None] | None,
                         target_valid_height: int | None) -> tuple[list[Mask], Mask, Diagnostics]:
        """Resolve per-image policy before feature extraction or template selection."""
        heights = template_valid_heights
        if heights is None:
            heights = [None]*len(templates)
        if not isinstance(heights, Sequence) or len(heights) != len(templates):
            raise ValueError('template_valid_heights must have one entry per template')
        policies = [ValidRegion.from_config(self.config, height) for height in heights]
        masks = [policy.build_mask(image, mask) for policy, image, mask in zip(policies, templates, valids)]
        target_policy = ValidRegion.from_config(self.config, target_valid_height)
        target_mask = target_policy.build_mask(input, target_valid)
        info = dict(templates=[policy.describe(image, mask) for policy, image, mask in zip(policies, templates, masks)],
                    target=target_policy.describe(input, target_mask))
        return masks, target_mask, info

    def evaluate(self, templates: Sequence[Image], masks: Sequence[LabelInput | None],
                 input: Image, input_masks: LabelInput, *, input_valid: Mask | None = None,
                 template_valids: Sequence[Mask | None] | None = None,
                 annotation_valids: Sequence[Mask | None] | None = None,
                 target_valid: Mask | None = None,
                 template_valid_heights: Sequence[int | None] | None = None,
                 target_valid_height: int | None = None) -> EvaluationResult:
        """Match once and compare to GT, which never guides registration.

        input_masks is HxW integer IDs or binary class masks 1..K. input_valid
        excludes unknown GT. Score=100*mIoU on full GT ROI, including uncovered
        foreground as FN. Rejected/unannotated matches or empty ROI score None.
        """
        if not isinstance(input, np.ndarray):
            raise ValueError('input must be a numpy image array')
        validate_image(input)
        truth = normalize_masks(input_masks, input.shape[:2], 'input_masks')
        self._valids([input_valid], [input], 'input_valid')
        roi = ValidRegion.from_config(self.config, target_valid_height).build_mask(input, target_valid) > 0
        if input_valid is not None:
            roi &= input_valid > 0
        matched = self.match(templates, masks, input, template_valids=template_valids,
                             annotation_valids=annotation_valids, target_valid=target_valid,
                             template_valid_heights=template_valid_heights, target_valid_height=target_valid_height)
        if matched.mapped_labels is None:
            metrics = dict(status='match_rejected' if not matched.accepted else 'no_template_annotations',
                           score=None, mean_iou=None, mean_dice=None, pixel_accuracy=None,
                           prediction_coverage=0.0 if roi.any() else None,
                           evaluated_pixels=0, per_class=[], overlap_only=None)
        else:
            metrics = evaluate_segmentation(matched.mapped_labels, truth, matched.label_valid,
                                            roi, include_background=self.config.include_background)
        return EvaluationResult(matched, metrics, truth, roi)

    def save_result(self, result: MatchResult | EvaluationResult,
                    output_dir: PathInput) -> Path:
        """Stage 1: save arrays and metadata; open them with the stage-2 app."""
        return save_result(result, output_dir)
