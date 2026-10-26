"""Configuration, duplicate handling and stage-1 prediction persistence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import hashlib
import json

import cv2

from ..config import MatcherConfig
from ..matcher import PeriodicTemplateMatcher
from ..utils import read_image
from ..utils.types import Diagnostics, PathInput, Resolver
from .annotations import load_labels
from .result_io import ARTIFACTS


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


def run_config(config_path: PathInput, output_dir: PathInput) -> Diagnostics:
    config_path, out = Path(config_path).resolve(), Path(output_dir).resolve()
    spec = json.loads(config_path.read_text(encoding='utf-8-sig'))

    def resolve(value: str) -> Path:
        return (config_path.parent / value).resolve()

    options = dict(spec.get('options', {}))
    cfg = MatcherConfig.from_options(options)
    cfg.label_mode = spec.get('label_mode', cfg.label_mode)
    cfg.validate()
    specs, aliases = unique_templates(spec['templates'], resolve)
    images = [read_image(resolve(s['image'])) for s in specs]
    target = read_image(resolve(spec['target']))
    valids = [read_image(resolve(s['valid_mask']), cv2.IMREAD_GRAYSCALE)
              if s.get('valid_mask') else None for s in specs]
    tv = read_image(resolve(spec['target_valid_mask']), cv2.IMREAD_GRAYSCALE) if spec.get('target_valid_mask') else None
    # Validate even non-selected annotations, so a bad template cannot hide silently.
    labels = [load_labels(s, resolve, im.shape[:2]) for s, im in zip(specs, images)]
    # Annotation knowledge and image ROI are intersected once inside match().
    annotation_valids = [read_image(resolve(s['annotation_valid']), cv2.IMREAD_GRAYSCALE)
                        if s.get('annotation_valid') else None for s in specs]
    matcher = PeriodicTemplateMatcher(cfg)
    matched = matcher.match(images, labels, target,
                            template_valids=valids,
                            annotation_valids=annotation_valids,
                            target_valid=tv,
                            template_valid_heights=[s.get('valid_height') for s in specs],
                            target_valid_height=spec.get('target_valid_height'))
    result = matched.registration
    result.update(template_names=[s['name'] for s in specs], target=str(resolve(spec['target'])),
                  input_template_count=len(spec['templates']), unique_template_count=len(specs),
                  template_aliases=aliases, warnings=[])
    for match, template in zip(result['matches'], specs):
        match.update(template_name=template['name'], template_image=str(resolve(template['image'])),
                     annotation_available=labels[match['template_index']] is not None)
    selected = specs[result['ranking'][0]]
    result['class_names'] = {str(value): name for name, value in selected.get('class_map', {}).items()}
    # The exporter replaces owned artifacts; none may be configured input files.
    inputs = {config_path, resolve(spec['target'])}
    if spec.get('target_valid_mask'):
        inputs.add(resolve(spec['target_valid_mask']))
    for template in spec['templates']:
        inputs.update(resolve(template[key]) for key in
                      ('image', 'labels', 'labelme', 'valid_mask', 'annotation_valid')
                      if template.get(key))
    if any((out / name).resolve() in inputs for name in ARTIFACTS):
        raise ValueError('Output artifacts must not overwrite input files')
    manifest = matcher.save_result(matched, out)
    return json.loads(manifest.read_text(encoding='utf-8'))
