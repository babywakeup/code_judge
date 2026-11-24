"""Public matcher for configured files and in-memory arrays."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import partial
import json
from os import PathLike
from pathlib import Path
import time

from ._internal.io.workflow import InferenceFiles
from ._internal.objects.matching import match_ordered_objects
from ._internal.objects.completion import complete_object_support
from ._internal.pipeline import MatchPipeline
from ._internal.registration.core import (
    PreparedImage, prepare_registration_sources, prepare_registration_target, select_templates,
)
from ._internal.models import EvaluationResult, MatchResult, PreparedTarget
from ._internal.io import export
from ._internal.segmentation.evaluation import SegmentationEvaluationStep
from ._internal.segmentation.crop import restore_crop
from ._internal.objects.steps import ObjectSupportCompletionStep, OrderedObjectStep
from ._internal.segmentation.preparation import InputPreparationStep
from ._internal.registration.step import RegistrationStep
from .config import MatcherConfig
from .utils.regions import ValidRegion
from .utils.types import Diagnostics, Image, LabelInput, Mask, PathInput


class PeriodicTemplateMatcher:
    """Run registration, label transfer and inspection on files or arrays.

    Pass a JSON configuration path to preload file inputs for ``infer()``.
    A MatcherConfig can configure direct array calls to ``match()``.

    templates: nonempty list of uint8 gray/BGR/BGRA photographs.
    masks: one entry per template: HxW integer IDs, disjoint binary class masks
           (classes 1..K), or None. Label knowledge is separate from image ROI.
    input: a single target image. Duplicate array candidates are not collapsed.
    """

    def __init__(self, config: MatcherConfig | PathInput | None = None) -> None:
        self._files = None
        if isinstance(config, (str, PathLike)):
            self._files = InferenceFiles(config)
            self.config = deepcopy(self._files.config)
            self._initialize_file_inputs()
        elif config is None or isinstance(config, MatcherConfig):
            self.config = deepcopy(config) if config is not None else MatcherConfig()
        else:
            raise TypeError('config must be MatcherConfig or a configuration file path')
        self.config.validate()

    def _initialize_file_inputs(self) -> None:
        files = self._files
        assert files is not None
        self._file_preparation = InputPreparationStep(files.config)
        templates = files.read_templates()
        self._templates = self._file_preparation.prepare_templates(
            templates.images, templates.labels, template_valids=templates.valids,
            annotation_valids=templates.annotations, template_valid_heights=templates.heights,
            template_objects=templates.objects)
        self._default_target = (self._prepare_file_target(files, files.default_target)
                                if files.default_target is not None else None)
        shapes = [image.shape for image in self._templates.images]
        if self._default_target is not None:
            shapes.append(self._default_target.image.shape)
        self._source_scale = files.config.matching_scale(*shapes)
        config = files.config.without_valid_region()
        self._registration_sources = prepare_registration_sources(
            self._templates.images, self._templates.valids, config, self._source_scale)
        self._alternate_sources: tuple[float, tuple[PreparedImage, ...]] | None = None
        self._registration_target = (
            prepare_registration_target(self._default_target.image, self._default_target.valid,
                                        config, self._source_scale)
            if self._default_target is not None else None)

    def _configured_files(self) -> InferenceFiles:
        if self._files is None:
            raise ValueError('File inference requires a configuration path at initialization')
        return self._files

    def infer(self, target: PathInput | None = None) -> MatchResult:
        """Infer a configured target, or a path relative to the configuration file."""
        files = self._configured_files()
        return self._infer_file(files, files.target_path(target))

    def get_valid(self, image: Image, *, valid_height: int | None = None) -> Image:
        """Return a copy of the configured valid rectangle in original pixels."""
        config = self._files.config if self._files is not None else self.config
        return ValidRegion.from_config(config, valid_height).crop(image)

    def _prepare_file_target(self, files: InferenceFiles, path: Path) -> PreparedTarget:
        image, valid = files.read_target(path)
        return self._file_preparation.prepare_target(image, valid, files.target_height)

    def _infer_file(self, files: InferenceFiles, path: Path) -> MatchResult:
        use_default = path == files.default_target
        target = (self._default_target if use_default
                  else self._prepare_file_target(files, path))
        assert target is not None
        inputs = self._file_preparation.combine(self._templates, target)
        scale = files.config.matching_scale(*(image.shape for image in (*inputs.templates, inputs.target)))
        selector = partial(select_templates, _prepared_sources=self._sources_for_scale(files.config, scale),
                           _prepared_target=self._registration_target if use_default else None)
        result = restore_crop(self._pipeline(files.config, selector).run(inputs), inputs)
        # MatchResult exposes writable images. Keep the next inference's assets private.
        result.template_image = result.template_image.copy()
        if use_default:
            result.input_image = result.input_image.copy()
        files.annotate(result, path, self._templates.labels)
        return result

    def _sources_for_scale(self, config: MatcherConfig, scale: float) -> tuple[PreparedImage, ...]:
        if scale == self._source_scale:
            return self._registration_sources
        alternate = self._alternate_sources
        if alternate is not None and alternate[0] == scale:
            return alternate[1]
        sources = prepare_registration_sources(self._templates.images, self._templates.valids,
                                               config.without_valid_region(), scale)
        self._alternate_sources = (scale, sources)
        return sources

    def infer_to(self, output_dir: PathInput, *, target: PathInput | None = None,
                 timings: dict[str, float] | None = None) -> Diagnostics:
        """Run inference and export. Timings can report seconds excluding PNG export."""
        started = time.perf_counter() if timings is not None else None
        files = self._configured_files()
        path = files.target_path(target)
        out = Path(output_dir).resolve()
        files.validate_output(out, path)
        result = self._infer_file(files, path)
        out = out.resolve()
        files.validate_output(out, path)
        manifest = (self.save_result(result, out) if timings is None
                    else self.save_result(result, out, timings=timings))
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if started is not None:
            timings['seconds'] = time.perf_counter() - started - timings['image_save_seconds']
        return metadata

    @staticmethod
    def _pipeline(config: MatcherConfig, selector: Callable[..., Diagnostics]) -> MatchPipeline:
        return MatchPipeline(config, registration=RegistrationStep(selector),
                             ordered_objects=OrderedObjectStep(match_ordered_objects),
                             object_support=ObjectSupportCompletionStep(complete_object_support,
                                                                       locator=match_ordered_objects))

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
        return restore_crop(self._pipeline(self.config, select_templates).run(prepared), prepared)

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

    def _export_path(self, result: MatchResult | EvaluationResult, output_dir: PathInput,
                     names: Sequence[str]) -> Path:
        out = Path(output_dir).resolve()
        if self._files is not None:
            match = result.match if isinstance(result, EvaluationResult) else result
            target = match.registration.get('target') if isinstance(match, MatchResult) else None
            self._files.validate_output(out, Path(target).resolve() if isinstance(target, str) else None, names)
        return out

    def save_images(self, result: MatchResult | EvaluationResult, output_dir: PathInput) -> dict[str, str]:
        """Save result PNGs and return their relative paths by artifact name."""
        out = self._export_path(result, output_dir, export.IMAGE_ARTIFACTS)
        return export.save_images(result, out)

    def save_json(self, result: MatchResult | EvaluationResult, output_dir: PathInput, *,
                  artifacts: Mapping[str, str] | None = None) -> Path:
        """Save only result.json, optionally linking images in the same directory."""
        out = self._export_path(result, output_dir, ('result.json',))
        return export.save_json(result, out, artifacts=artifacts)

    def save_result(self, result: MatchResult | EvaluationResult, output_dir: PathInput, *,
                    timings: dict[str, float] | None = None) -> Path:
        """Save the viewer bundle. Optionally record image_save_seconds separately."""
        out = self._export_path(result, output_dir, export.ARTIFACTS)
        return export.save_result(result, out, timings=timings)
