"""Public array API: select templates, transfer labels, evaluate and inspect."""
from copy import deepcopy

import cv2
import numpy as np

from .config import MatcherConfig
from ._internal.evaluation import evaluate_segmentation
from ._internal.registration import select_templates
from ._internal.results import EvaluationResult, MatchResult
from .utils import build_valid_mask, normalize_masks, to_gray, warp_labels
from ._internal.visual_report import visualize_match


class PeriodicTemplateMatcher:
    """Array-based facade for registration, mask transfer and inspection.

    templates: nonempty list of uint8 gray/BGR/BGRA photographs.
    masks: one entry per template: HxW integer IDs, disjoint binary class masks
           (classes 1..K), or None. Label knowledge is separate from image ROI.
    input: a single target image. Duplicate array candidates are not collapsed.
    """

    def __init__(self, config: MatcherConfig | None = None):
        if config is not None and not isinstance(config, MatcherConfig):
            raise TypeError('config must be MatcherConfig')
        self.config = deepcopy(config) if config is not None else MatcherConfig()
        self.config.validate()

    @staticmethod
    def _valids(values, images, name):
        if values is None:
            return [None] * len(images)
        if not isinstance(values, (list, tuple)) or len(values) != len(images):
            raise ValueError(f'{name} must have one entry per template')
        for mask, image in zip(values, images):
            if mask is not None and (not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]):
                raise ValueError(f'{name} entries must match image height and width')
        return values

    def match(self, templates, masks, input, *, template_valids=None,
              annotation_valids=None, target_valid=None) -> MatchResult:
        """Select a template and transfer its labels only on accepted matches.

        template_valids/target_valid limit image ROI. annotation_valids mark
        known template labels; zero is otherwise known background.
        """
        self.config.validate()
        if not isinstance(templates, (list, tuple)) or not templates:
            raise ValueError('templates must be a nonempty list of images')
        for image in [*templates, input]:
            if not isinstance(image, np.ndarray):
                raise ValueError('Images must be numpy arrays; use read_image for file paths')
            to_gray(image)
        if not isinstance(masks, (list, tuple)) or len(masks) != len(templates):
            raise ValueError('masks must have one entry per template (None is allowed)')
        labels = [None if mask is None else normalize_masks(mask, image.shape[:2], f'masks[{i}]')
                  for i, (mask, image) in enumerate(zip(masks, templates))]
        valids = self._valids(template_valids, templates, 'template_valids')
        annotations = self._valids(annotation_valids, templates, 'annotation_valids')
        self._valids([target_valid], [input], 'target_valid')
        registration = select_templates(templates, input, self.config, valids,
                                        target_valid, self.config.label_mode)
        index = registration['ranking'][0]
        source = templates[index]
        best = registration['matches'][index]
        H = best.get('matrix_template_to_target')
        result = MatchResult(registration, input, source, None, np.zeros(input.shape[:2], bool))
        if H is None:
            return result
        H = np.asarray(H, np.float64)
        target_roi = build_valid_mask(to_gray(input), target_valid, self.config) > 0
        source_roi = build_valid_mask(to_gray(source), valids[index], self.config)
        result.warped_template = cv2.warpPerspective(source, H, (input.shape[1], input.shape[0]))
        _, result.geometric_valid = warp_labels(np.zeros(source.shape[:2], np.uint8), H, input.shape, source_roi)
        result.geometric_valid &= target_roi
        if result.accepted and labels[index] is not None:
            if annotations[index] is not None:
                source_roi &= (annotations[index] > 0).astype(np.uint8)
            mapped, valid = warp_labels(labels[index], H, input.shape, source_roi)
            valid &= target_roi
            mapped[~valid] = 0
            result.mapped_labels, result.label_valid = mapped, valid
        return result

    def evaluate(self, templates, masks, input, input_masks, *, input_valid=None,
                 template_valids=None, annotation_valids=None, target_valid=None) -> EvaluationResult:
        """Match once and compare to GT, which never guides registration.

        input_masks is HxW integer IDs or binary class masks 1..K. input_valid
        excludes unknown GT. Score=100*mIoU on full GT ROI, including uncovered
        foreground as FN. Rejected/unannotated matches or empty ROI score None.
        """
        if not isinstance(input, np.ndarray):
            raise ValueError('input must be a numpy image array')
        to_gray(input)
        truth = normalize_masks(input_masks, input.shape[:2], 'input_masks')
        self._valids([input_valid], [input], 'input_valid')
        self._valids([target_valid], [input], 'target_valid')
        roi = build_valid_mask(to_gray(input), target_valid, self.config) > 0
        if input_valid is not None:
            roi &= input_valid > 0
        matched = self.match(templates, masks, input, template_valids=template_valids,
                             annotation_valids=annotation_valids, target_valid=target_valid)
        if matched.mapped_labels is None:
            metrics = dict(status='match_rejected' if not matched.accepted else 'no_template_annotations',
                           score=None, mean_iou=None, mean_dice=None, pixel_accuracy=None,
                           prediction_coverage=0.0 if roi.any() else None,
                           evaluated_pixels=0, per_class=[], overlap_only=None)
        else:
            metrics = evaluate_segmentation(matched.mapped_labels, truth, matched.label_valid,
                                            roi, include_background=self.config.include_background)
        return EvaluationResult(matched, metrics, truth, roi)

    def visualize(self, result: MatchResult | EvaluationResult, output_path=None):
        """Return a BGR inspection canvas and optionally save it, without a GUI."""
        return visualize_match(result, output_path)