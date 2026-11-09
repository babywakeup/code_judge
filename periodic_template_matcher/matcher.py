"""Public array API: select templates, transfer labels, evaluate and inspect."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from ._internal.objects.matching import match_ordered_objects
from ._internal.objects.completion import complete_object_support
from ._internal.pipeline import MatchPipeline
from ._internal.registration.core import select_templates
from ._internal.models import EvaluationResult, MatchResult
from ._internal.io.export import save_result
from ._internal.segmentation.evaluation import SegmentationEvaluationStep
from ._internal.objects.steps import ObjectSupportCompletionStep, OrderedObjectStep
from ._internal.segmentation.preparation import InputPreparationStep
from ._internal.registration.step import RegistrationStep
from .config import MatcherConfig
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
        """Compatibility delegate; validation belongs to InputPreparationStep."""
        return InputPreparationStep.validate_masks(values, images, name)

    def match(self, templates: Sequence[Image], masks: Sequence[LabelInput | None],
              input: Image, *, template_valids: Sequence[Mask | None] | None = None,
              annotation_valids: Sequence[Mask | None] | None = None,
              target_valid: Mask | None = None,
              template_valid_heights: Sequence[int | None] | None = None,
              target_valid_height: int | None = None,
              template_objects: Sequence[Sequence[Diagnostics]] | None = None) -> MatchResult:
        """Select a template and transfer its labels only on accepted matches.

        template_valids/target_valid limit image ROI. annotation_valids mark
        known template labels; zero is otherwise known background. Explicit valid
        heights are original-pixel top-row counts, overriding default exclusions.
        """
        prepared = InputPreparationStep(self.config).run(
            templates, masks, input, template_valids=template_valids,
            annotation_valids=annotation_valids, target_valid=target_valid,
            template_valid_heights=template_valid_heights, target_valid_height=target_valid_height,
            template_objects=template_objects)
        # Keep existing algorithm injection points while steps own their policies.
        pipeline = MatchPipeline(
            self.config, registration=RegistrationStep(select_templates),
            ordered_objects=OrderedObjectStep(match_ordered_objects),
            object_support=ObjectSupportCompletionStep(complete_object_support))
        return pipeline.run(prepared)

    def _prepare_regions(self, templates: Sequence[Image], input: Image,
                         valids: Sequence[Mask | None], target_valid: Mask | None,
                         template_valid_heights: Sequence[int | None] | None,
                         target_valid_height: int | None) -> tuple[list[Mask], Mask, Diagnostics]:
        """Compatibility delegate; ROI construction belongs to InputPreparationStep."""
        return InputPreparationStep(self.config).prepare_regions(
            templates, input, valids, target_valid, template_valid_heights, target_valid_height)

    def evaluate(self, templates: Sequence[Image], masks: Sequence[LabelInput | None],
                 input: Image, input_masks: LabelInput, *, input_valid: Mask | None = None,
                 template_valids: Sequence[Mask | None] | None = None,
                 annotation_valids: Sequence[Mask | None] | None = None,
                 target_valid: Mask | None = None,
                 template_valid_heights: Sequence[int | None] | None = None,
                 target_valid_height: int | None = None,
                 template_objects: Sequence[Sequence[Diagnostics]] | None = None) -> EvaluationResult:
        """Match once and compare to GT, which never guides registration.

        input_masks is HxW integer IDs or binary class masks 1..K. input_valid
        excludes unknown GT. Score=100*mIoU on full GT ROI, including uncovered
        foreground as FN. Rejected/unannotated matches or empty ROI score None.
        """
        evaluation = SegmentationEvaluationStep(self.config)
        truth = evaluation.prepare(input, input_masks, input_valid, target_valid, target_valid_height)
        matched = self.match(templates, masks, input, template_valids=template_valids,
                             annotation_valids=annotation_valids, target_valid=target_valid,
                             template_valid_heights=template_valid_heights, target_valid_height=target_valid_height,
                             template_objects=template_objects)
        return evaluation.run(matched, truth)

    def save_result(self, result: MatchResult | EvaluationResult,
                    output_dir: PathInput) -> Path:
        """Stage 1: save arrays and metadata; open them with the stage-2 app."""
        return save_result(result, output_dir)
