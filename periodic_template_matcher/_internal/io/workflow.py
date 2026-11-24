"""Configuration, duplicate handling and stage-1 prediction persistence."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import cv2

from ...config import MatcherConfig
from ...utils import read_image
from ...utils.types import Diagnostics, Image, LabelInput, Mask, PathInput, Resolver
from ..models import MatchResult
from .annotations import load_labels, load_objects, object_layout, layout_axis
from .export import ARTIFACTS


_TEMPLATE_PATH_FIELDS = ('image', 'labels', 'labelme', 'annotation_valid', 'valid_mask')


@dataclass
class TemplateFiles:
    images: list[Image]
    labels: list[LabelInput | None]
    valids: list[Mask | None]
    annotations: list[Mask | None]
    heights: list[int | None]
    objects: list[list[Diagnostics]] | None


def unique_templates(
    specs: Sequence[Diagnostics], resolve: Resolver,
) -> tuple[list[Diagnostics], list[list[Diagnostics]]]:
    """Collapse identical image files; conflicting labels/ROIs must be resolved."""
    unique, aliases, seen = [], [], {}
    for index, item in enumerate(specs):
        digest = hashlib.sha256(resolve(item['image']).read_bytes()).hexdigest()
        signature = {key: str(resolve(item[key])) if item.get(key) else None
                     for key in ('labels', 'labelme', 'annotation_valid', 'valid_mask')}
        signature['class_map'] = item.get('class_map', {})
        signature['valid_height'] = item.get('valid_height')
        signature['object_order'] = item.get('object_order')
        signature['instance_layouts'] = item.get('instance_layouts')
        name = item.get('name', f'template_{index + 1}')
        if digest in seen:
            previous, original_signature = seen[digest]
            if signature != original_signature:
                raise ValueError('Duplicate template image has conflicting annotations or valid masks')
            aliases[previous].append({'input_index': index, 'name': name})
        else:
            seen[digest] = (len(unique), signature)
            unique.append(dict(item, name=name))
            aliases.append([{'input_index': index, 'name': name}])
    return unique, aliases


class InferenceFiles:
    """Read configured assets and attach file provenance to an array result."""

    def __init__(self, config_path: PathInput) -> None:
        self.path = Path(config_path).resolve()
        spec = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.options = dict(spec.get('options', {}))
        self.config = MatcherConfig.from_options(self.options, object_order_axis=layout_axis(spec['templates']))
        self.config.label_mode = spec.get('label_mode', self.config.label_mode)
        self.config.validate()
        self.specs, self.aliases = unique_templates(spec['templates'], self.resolve)
        for template in self.specs:
            for key in _TEMPLATE_PATH_FIELDS:
                if template.get(key):
                    template[key] = str(self.resolve(template[key]))
        self.input_template_count = len(spec['templates'])
        self.layouts = [
            object_layout(s, self.config.ordered_object_class)
            if self.config.ordered_object_class is not None or 'instance_layouts' in s else None
            for s in self.specs]
        self.default_target = self.resolve(spec['target']) if spec.get('target') else None
        self.target_mask = self.resolve(spec['target_valid_mask']) if spec.get('target_valid_mask') else None
        self.target_height = spec.get('target_valid_height')
        self.input_paths = {self.path, self.default_target, self.target_mask}
        for template in spec['templates']:
            self.input_paths.update(self.resolve(template[key]) for key in
                                    _TEMPLATE_PATH_FIELDS
                                    if template.get(key))

    def resolve(self, value: PathInput) -> Path:
        return (self.path.parent / value).resolve()

    def read_templates(self) -> TemplateFiles:
        specs = self.specs
        images = [read_image(self.resolve(s['image'])) for s in specs]
        labels = [load_labels(s, self.resolve, im.shape[:2]) for s, im in zip(specs, images)]
        cfg = self.config
        objects = ([load_objects(s, self.resolve, im.shape[:2], cfg.ordered_object_class, cfg.object_cycle)
                    for s, im in zip(specs, images)] if cfg.ordered_object_class is not None else None)
        valids = [self._mask(s, 'valid_mask') for s in specs]
        annotations = [self._mask(s, 'annotation_valid') for s in specs]
        return TemplateFiles(images, labels, valids, annotations,
                             [s.get('valid_height') for s in specs], objects)

    def _mask(self, spec: Diagnostics, key: str) -> Mask | None:
        return read_image(self.resolve(spec[key]), cv2.IMREAD_GRAYSCALE) if spec.get(key) else None

    def target_path(self, target: PathInput | None) -> Path:
        path = self.default_target if target is None else self.resolve(target)
        if path is None:
            raise ValueError('Provide a target image or configure a default target')
        if self.target_mask is not None and path != self.default_target:
            raise ValueError('The configured target_valid_mask applies only to the default target')
        return path

    def read_target(self, path: Path) -> tuple[Image, Mask | None]:
        valid = read_image(self.target_mask, cv2.IMREAD_GRAYSCALE) if self.target_mask is not None else None
        return read_image(path), valid

    def annotate(self, match: MatchResult, target: Path, labels: Sequence[LabelInput | None]) -> None:
        result = match.registration
        result.update(template_names=[s['name'] for s in self.specs], target=str(target),
                      input_template_count=self.input_template_count, unique_template_count=len(self.specs),
                      template_aliases=deepcopy(self.aliases), warnings=[])
        if 'classes' in self.options:
            result['class_rules'] = deepcopy(self.options['classes'])
        result['template_instance_layouts'] = [
            {str(self.config.ordered_object_class): deepcopy(layout)} if layout is not None else {}
            for layout in self.layouts]
        for candidate, template in zip(result['matches'], self.specs):
            candidate.update(template_name=template['name'], template_image=template['image'],
                             annotation_available=labels[candidate['template_index']] is not None)
        selected = self.specs[result['ranking'][0]]
        result['class_names'] = {str(value): name for name, value in selected.get('class_map', {}).items()}

    def validate_output(self, out: Path, target: Path | None, names: Sequence[str] = ARTIFACTS) -> None:
        inputs = self.input_paths | {target}
        if any((out / name).resolve() in inputs for name in names):
            raise ValueError('Output artifacts must not overwrite input files')


def run_config(config_path: PathInput, output_dir: PathInput) -> Diagnostics:
    from ...matcher import PeriodicTemplateMatcher

    return PeriodicTemplateMatcher(config_path).infer_to(output_dir)
