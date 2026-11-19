"""Normal-only residual baselines; never learn these statistics from the defect input."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np


TEXTURE_CUTS = [1.5, 3., 6.]
FLOORS = np.array([4., 2., 4., .03], np.float32)


@dataclass
class NormalBaseline:
    groups: dict
    metadata: dict
    protected_paths: tuple = ()

    def score(self, responses, texture, valid, labels, weight, structure_weight):
        score = np.zeros(valid.shape, np.float32)
        known = np.zeros(valid.shape, bool)
        if labels is None:
            return score, known, dict(self.metadata, status='missing_semantic_labels', valid_pixels=0)
        if labels.shape != valid.shape or labels.dtype.kind not in 'ui':
            raise ValueError('Normal baseline semantic labels must be an input-sized integer map')
        bins = np.digitize(texture, TEXTURE_CUTS)
        for key, statistics in self.groups.items():
            class_id, group = map(int, key.split(':'))
            region = valid & (labels == class_id) & (bins == group)
            if not region.any():
                continue
            local = np.zeros(int(region.sum()), np.float32)
            for index, response in enumerate(responses):
                z = np.maximum(0., (response[region]-statistics['median'][index])/statistics['scale'][index]-.5)
                if index == 1:
                    z *= structure_weight
                np.maximum(local, z, out=local)
            score[region] = local*weight[region]
            known |= region
        return score, known, dict(self.metadata, valid_pixels=int(known.sum()))


def fit_normal_baseline(samples: list[dict], *, metadata: dict | None = None) -> NormalBaseline:
    collected = {}
    for sample in samples:
        bins = np.digitize(sample['texture'], TEXTURE_CUTS)
        for class_id in np.unique(sample['labels'][sample['valid']]):
            for group in range(4):
                mask = sample['valid'] & (sample['labels'] == class_id) & (bins == group)
                indices = np.flatnonzero(mask)
                if len(indices) < 128:
                    continue
                if len(indices) > 50000:
                    indices = indices[np.linspace(0, len(indices)-1, 50000, dtype=int)]
                values = np.column_stack([r.ravel()[indices] for r in sample['responses']])
                collected.setdefault(f'{int(class_id)}:{group}', []).append(values)
    groups = {}
    for key, entries in collected.items():
        values = np.concatenate(entries)
        median = np.median(values, axis=0)
        scale = np.maximum(FLOORS, 1.4826*np.median(np.abs(values-median), axis=0))
        groups[key] = dict(samples=len(values), median=median.tolist(), scale=scale.tolist())
    info = dict(metadata or {}, status='available' if groups else 'insufficient_normal_samples',
                training_source='normal_templates_only', texture_cutpoints=TEXTURE_CUTS,
                signal_order=['color', 'structure', 'multiscale', 'ssim'],
                groups=groups, normal_scale_floors=FLOORS.tolist(),
                limitation='Few normal templates do not cover all normal variation; photometric fitting can absorb broad changes.')
    return NormalBaseline(groups, info)


def build_normal_baseline(config_path: str | Path) -> NormalBaseline:
    """Hold each normal template out and register other normal templates to it.

    Only accepted geometry, fitted photometry, matching semantic labels and full
    valid neighborhoods train the baseline. Ordered instances are conservatively
    excluded because a shared semantic class does not establish shared identity.
    """
    from periodic_template_matcher import MatcherConfig
    from periodic_template_matcher._internal.io.annotations import load_labels, layout_axis
    from periodic_template_matcher._internal.io.workflow import unique_templates
    from periodic_template_matcher._internal.registration.core import select_templates
    from periodic_template_matcher._internal.reference.step import fit_photometry
    from periodic_template_matcher.utils import read_image, ValidRegion, warp_labels, to_gray
    from .core import analyze_pair, AnomalyConfig

    path = Path(config_path).resolve()
    spec = json.loads(path.read_text(encoding='utf-8-sig'))
    resolve = lambda name: (path.parent / name).resolve()
    cfg = MatcherConfig.from_options(spec.get('options', {}), object_order_axis=layout_axis(spec['templates']))
    cfg.validate()
    templates, _ = unique_templates(spec['templates'], resolve)
    protected = {path}
    for template in spec['templates']:
        protected.update(resolve(template[k]) for k in ('image', 'labelme', 'labels', 'annotation_valid', 'valid_mask') if template.get(k))
    images = [read_image(resolve(t['image'])) for t in templates]
    labels = [load_labels(t, resolve, im.shape[:2]) for t, im in zip(templates, images)]
    valids = []
    for template, image in zip(templates, images):
        extra = read_image(resolve(template['valid_mask']), 0) if template.get('valid_mask') else None
        valid = ValidRegion.from_config(cfg, template.get('valid_height')).build_mask(image, extra)
        if template.get('annotation_valid'):
            valid &= (read_image(resolve(template['annotation_valid']), 0) > 0).astype(np.uint8)
        valids.append(valid)
    samples, records = [], []
    for target_index, target in enumerate(images):
        others = [i for i in range(len(images)) if i != target_index]
        if not others:
            break
        registration = select_templates([images[i] for i in others], target,
            replace(cfg, valid_height=None), [valids[i] for i in others], valids[target_index], 'semantic')
        match = registration['matches'][registration['ranking'][0]]
        source_index = others[registration['ranking'][0]]
        record = dict(target_index=target_index, source_index=source_index,
                      accepted=False, registration_status=registration['status'], match_score=match.get('match_score'))
        records.append(record)
        if not registration['accepted'] or not match.get('geometry_accepted', False):
            continue
        if labels[source_index] is None or labels[target_index] is None:
            record['reason'] = 'normal_semantic_annotations_required'
            continue
        matrix = np.asarray(match['matrix_template_to_target'], float)
        size = (target.shape[1], target.shape[0])
        support = cv2.warpPerspective(valids[source_index].astype(np.float32), matrix, size) >= 1-1e-6
        warped_labels, _ = warp_labels(labels[source_index], matrix, target.shape, valids[source_index])
        valid = support & (valids[target_index] > 0) & (warped_labels == labels[target_index])
        if cfg.ordered_object_class is not None:
            valid &= labels[target_index] != cfg.ordered_object_class
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        warped = cv2.warpPerspective(images[source_index], matrix, size)
        photo = fit_photometry(warped, target, valid)
        record['photometry'] = photo
        if photo['status'] != 'fitted':
            record['reason'] = 'normal_photometry_unreliable'
            continue
        corrected = np.clip(np.rint(warped.astype(np.float32)*photo['gain']+photo['offset']), 0, 255).astype(np.uint8)
        analysis = analyze_pair(target, corrected, valid, config=AnomalyConfig(tolerance_radius=0))
        gray = to_gray(corrected).astype(np.float32)
        texture = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0)/8, cv2.Sobel(gray, cv2.CV_32F, 0, 1)/8)
        texture[cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0] = 0
        texture = cv2.GaussianBlur(texture, (5, 5), 1.)
        eligible = analysis.ssim_valid & (analysis.confidence == 255)
        # Keep bounded normal-only samples, not N full-resolution analysis canvases.
        indices = np.flatnonzero(eligible)
        if len(indices) > 100000:
            indices = indices[np.linspace(0, len(indices)-1, 100000, dtype=int)]
        responses = (analysis.color_response, analysis.structure_response, analysis.multiscale_response, analysis.ssim_response)
        samples.append(dict(responses=tuple(r.ravel()[indices] for r in responses),
                            texture=texture.ravel()[indices], labels=labels[target_index].ravel()[indices],
                            valid=np.ones(len(indices), bool)))
        record.update(accepted=True, samples=len(indices), matrix_template_to_target=matrix.tolist())
    model = fit_normal_baseline(samples, metadata=dict(config_path=str(path), normal_pairs=records,
        excluded_ordered_class=cfg.ordered_object_class,
        template_images=[str(resolve(t['image'])) for t in templates]))
    model.protected_paths = tuple(protected)
    return model
