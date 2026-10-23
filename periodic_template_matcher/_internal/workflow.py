"""Configuration, duplicate handling, annotation transfer and report artifacts."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .annotations import demo_labels, load_labels
from ..config import MatcherConfig
from ..matcher import PeriodicTemplateMatcher
from ..utils import (build_valid_mask, overlay_labels, read_image, to_gray,
                     warp_labels, write_image)
from .visual_report import build_report


ARTIFACTS = (
    'result.json', 'report.html', 'overview.jpg', 'best_alignment.jpg',
    'geometric_valid.png', 'matrix_template_to_target.npy', 'mapped_labels.png',
    'label_valid.png', 'label_overlay.jpg', 'demo_source_labels.png',
    'demo_mapped_labels.png', 'demo_label_valid.png', 'demo_label_overlay.jpg',
)


def unique_templates(specs, resolve):
    """Collapse identical image files; conflicting labels/ROIs must be resolved."""
    unique, aliases, seen = [], [], {}
    for index, item in enumerate(specs):
        digest = hashlib.sha256(resolve(item['image']).read_bytes()).hexdigest()
        signature = {key: str(resolve(item[key])) if item.get(key) else None
                     for key in ('labels', 'labelme', 'annotation_valid', 'valid_mask')}
        signature['class_map'] = item.get('class_map', {})
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


def run_config(config_path, output_dir):
    config_path, out = Path(config_path).resolve(), Path(output_dir).resolve()
    spec = json.loads(config_path.read_text(encoding='utf-8-sig'))

    def resolve(value):
        return (config_path.parent / value).resolve()

    options = dict(spec.get('options', {}))
    options['label_mode'] = spec.get('label_mode', 'semantic')
    cfg = MatcherConfig(**options)
    cfg.validate()
    specs, aliases = unique_templates(spec['templates'], resolve)
    images = [read_image(resolve(s['image'])) for s in specs]
    target = read_image(resolve(spec['target']))
    valids = [read_image(resolve(s['valid_mask']), cv2.IMREAD_GRAYSCALE)
              if s.get('valid_mask') else None for s in specs]
    tv = read_image(resolve(spec['target_valid_mask']), cv2.IMREAD_GRAYSCALE) if spec.get('target_valid_mask') else None
    # Validate even non-selected annotations, so a bad template cannot hide silently.
    labels = [load_labels(s, resolve, im.shape[:2]) for s, im in zip(specs, images)]
    annotation_valids = []
    for s, im, valid in zip(specs, images, valids):
        vm = build_valid_mask(to_gray(im), valid, cfg)
        if s.get('annotation_valid'):
            ann = read_image(resolve(s['annotation_valid']), cv2.IMREAD_GRAYSCALE)
            if ann.shape != vm.shape:
                raise ValueError('annotation_valid must match the template')
            vm &= (ann > 0).astype(np.uint8)
        annotation_valids.append(vm)
    matched = PeriodicTemplateMatcher(cfg).match(images, labels, target,
                                                template_valids=valids,
                                                annotation_valids=annotation_valids,
                                                target_valid=tv)
    result = matched.registration
    result.update(template_names=[s['name'] for s in specs], target=str(resolve(spec['target'])),
                  input_template_count=len(spec['templates']), unique_template_count=len(specs),
                  template_aliases=aliases, labels_exported=False, demo_labels_exported=False,
                  annotation_status='not_available', warnings=[])
    if len(specs) < len(spec['templates']):
        result['warnings'].append('重复模板已合并，不将同一张图作为两个竞争模板。')
    if len(specs) == 1:
        result['warnings'].append('当前仅有一个唯一模板：已验证配准，但未验证真实多模板选择。')
    for r, s in zip(result['matches'], specs):
        r.update(template_name=s['name'], template_image=str(resolve(s['image'])),
                 annotation_available=labels[r['template_index']] is not None)
    # Only remove owned artifacts, and never overwrite a configured input.
    inputs = {config_path, resolve(spec['target'])}
    if spec.get('target_valid_mask'):
        inputs.add(resolve(spec['target_valid_mask']))
    for s in specs:
        inputs.update(resolve(s[k]) for k in ('image', 'labels', 'labelme', 'valid_mask', 'annotation_valid') if s.get(k))
    if any(out / name in inputs for name in ARTIFACTS):
        raise ValueError('Output artifacts must not overwrite input files')
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        (out / name).unlink(missing_ok=True)
    # Always show the top candidate for diagnosis, including rejected registrations.
    idx = result['ranking'][0]
    best, image = result['matches'][idx], images[idx]
    if labels[idx] is not None:
        result['annotation_status'] = 'blocked_by_match'
    H = best.get('matrix_template_to_target')
    views = {}
    if H is not None:
        H = np.asarray(H)
        warped = matched.warped_template
        vm = build_valid_mask(to_gray(image), valids[idx], cfg)
        geometric_valid = matched.geometric_valid
        tm = build_valid_mask(to_gray(target), tv, cfg) > 0
        geometric_valid &= tm
        blend = target.copy()
        blend[geometric_valid] = cv2.addWeighted(target, .5, warped, .5, 0)[geometric_valid]
        views.update(warped=warped, valid=geometric_valid, blend=blend)
        write_image(out / 'best_alignment.jpg', blend)
        write_image(out / 'geometric_valid.png', geometric_valid.astype(np.uint8) * 255)
        np.save(out / 'matrix_template_to_target.npy', H)
        result['geometric_coverage_full_image'] = float(geometric_valid.mean())
        if labels[idx] is not None:
            result['annotation_status'] = 'blocked_by_match' if not result['accepted'] else 'exported'
            if result['accepted']:
                mapped, valid = matched.mapped_labels, matched.label_valid
                write_image(out / 'mapped_labels.png', mapped)
                write_image(out / 'label_valid.png', valid.astype(np.uint8) * 255)
                overlay = overlay_labels(target, mapped, valid)
                write_image(out / 'label_overlay.jpg', overlay)
                views.update(label_source=overlay_labels(image, labels[idx], annotation_valids[idx] > 0),
                             label_target=overlay)
                result.update(labels_exported=True, label_coverage_full_image=float(valid.mean()),
                              mapped_class_ids=np.unique(mapped[valid]).tolist())
        else:
            result['warnings'].append('所选模板没有真实分割标注；未生成正式 mapped_labels / label_valid。')
            if spec.get('demo_annotations', False):
                source = demo_labels(image.shape[:2])
                mapped, valid = warp_labels(source, H, target.shape, vm)
                valid &= tm
                mapped[~valid] = 0
                write_image(out / 'demo_source_labels.png', source)
                write_image(out / 'demo_mapped_labels.png', mapped)
                write_image(out / 'demo_label_valid.png', valid.astype(np.uint8) * 255)
                overlay = overlay_labels(target, mapped, valid)
                write_image(out / 'demo_label_overlay.jpg', overlay)
                views.update(label_source=overlay_labels(image, source, vm > 0), label_target=overlay)
                result['demo_labels_exported'] = True
                result['warnings'].append('demo_* 仅为人工几何示意，不是真实标注、预测或正式迁移结果。')
    else:
        result['warnings'].append('没有可用变换矩阵；仅展示原始输入与拒绝原因。')
    if best.get('periodic_detected'):
        result['warnings'].append('检测到周期结构，绝对周期编号未验证；不可据此迁移唯一实例编号。')
    result['report'] = 'report.html'
    build_report(out, result, images, target, views)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result