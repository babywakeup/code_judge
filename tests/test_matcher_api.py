"""Class API, segmentation metric arithmetic and visualization regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.evaluation import evaluate_segmentation
from periodic_template_matcher.utils import read_image, warp_labels
from periodic_template_matcher.utils.masks import normalize_masks
from periodic_template_matcher._internal.visual_report import visualize_match
from tests.helpers import texture


def selection(accepted=True, matrix=None):
    return dict(accepted=accepted, best_template_index=0, ranking=[0],
                status='accepted' if accepted else 'absolute_cell_unverified',
                matches=[dict(template_index=0, match_score=90.,
                              matrix_template_to_target=np.eye(3).tolist() if matrix is None else matrix)])


class MatcherApiTests(unittest.TestCase):
    def setUp(self):
        self.config = MatcherConfig(max_side=200, angles=(0.,), scales=(1.,),
                                    refine_candidates=3, ecc_iterations=35)
        self.matcher = PeriodicTemplateMatcher(self.config)
        self.image = texture()
        self.labels = np.zeros(self.image.shape[:2], np.uint16)
        self.labels[35:70, 40:90] = 1001

    def test_real_matching_translation_evaluation_and_direction(self):
        H = np.array([[1., 0, 6], [0, 1., -4], [0, 0, 1.]])
        target = cv2.warpPerspective(self.image, H, (200, 160))
        truth, _ = warp_labels(self.labels, H, target.shape)
        evaluation = self.matcher.evaluate([texture(99), self.image],
                                           [np.zeros_like(self.labels), self.labels], target, truth)
        result = evaluation.match
        self.assertTrue(result.accepted)
        self.assertEqual(result.best_template_index, 1)
        self.assertEqual(result.mapped_labels.dtype, np.uint16)
        self.assertEqual(result.mapped_labels[40, 55], 1001)
        self.assertGreater(evaluation.score, 95.)
        self.assertGreater(evaluation.metrics['mean_dice'], .97)
        self.assertFalse(result.label_valid[0, 0])
        json.dumps(evaluation.metrics, allow_nan=False)

    def test_binary_mask_lists_and_visualization(self):
        first = self.labels > 0
        second = np.zeros_like(first)
        second[100:120, 130:150] = True
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()):
            evaluated = self.matcher.evaluate([self.image], [[first, second]], self.image,
                                               [first.astype(np.uint8) * 255, second])
        self.assertEqual(evaluated.score, 100.)
        self.assertEqual(set(np.unique(evaluated.match.mapped_labels)), {0, 1, 2})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evaluation.png'
            canvas = self.matcher.visualize(evaluated, path)
            self.assertEqual(canvas.dtype, np.uint8)
            self.assertEqual(canvas.shape[2], 3)
            np.testing.assert_array_equal(read_image(path), canvas)
        self.assertEqual(visualize_match(evaluated.match).shape, canvas.shape)

    def test_grayscale_and_bgra_visualizations(self):
        for image in (cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)):
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()):
                result = self.matcher.match([image], [self.labels], image)
            self.assertEqual(self.matcher.visualize(result).shape[2], 3)

    def test_validity_is_separate_from_background_and_does_not_mutate_inputs(self):
        known = np.zeros_like(self.labels)
        known[20:90, 20:100] = 1
        before = self.labels.copy()
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()):
            result = self.matcher.match([self.image], [self.labels], self.image,
                                        annotation_valids=[known])
        self.assertFalse(result.label_valid[0, 0])
        self.assertTrue(result.label_valid[25, 25])
        self.assertEqual(result.mapped_labels[25, 25], 0)
        np.testing.assert_array_equal(before, self.labels)
        self.assertTrue(result.geometric_valid[0, 0])

    def test_rejected_and_missing_annotations_return_no_score_or_labels(self):
        for accepted, masks, status in ((False, [self.labels], 'match_rejected'),
                                         (True, [None], 'no_template_annotations')):
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(accepted)):
                evaluation = self.matcher.evaluate([self.image], masks, self.image, self.labels)
            self.assertIsNone(evaluation.score)
            self.assertEqual(evaluation.metrics['status'], status)
            self.assertIsNone(evaluation.match.mapped_labels)
            self.assertIsNone(evaluation.match.label_valid)
            self.assertIsNotNone(evaluation.match.warped_template)
            self.matcher.visualize(evaluation)

    def test_blank_match_has_no_matrix_or_prediction(self):
        blank = np.zeros_like(self.image)
        result = self.matcher.match([blank], [self.labels], blank)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.matrix_template_to_target)
        self.assertIsNone(result.warped_template)
        self.assertFalse(result.geometric_valid.any())
        self.matcher.visualize(result)

    def test_ground_truth_validity_not_passed_to_registration(self):
        known = np.zeros_like(self.labels)
        known[40:50, 50:60] = 1
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()) as selector:
            result = self.matcher.evaluate([self.image], [self.labels], self.image,
                                           self.labels, input_valid=known)
        self.assertIsNone(selector.call_args.args[4])  # target_valid, not input_valid
        self.assertEqual(result.metrics['evaluated_pixels'], 100)
        self.assertEqual(result.score, 100.)

    def test_empty_template_annotation_coverage_cannot_get_perfect_score(self):
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()):
            evaluated = self.matcher.evaluate([self.image], [self.labels], self.image,
                                               self.labels, annotation_valids=[np.zeros_like(self.labels)])
        self.assertTrue(evaluated.match.accepted)
        self.assertFalse(evaluated.match.label_valid.any())
        self.assertEqual(evaluated.score, 0.)
        self.assertEqual(evaluated.metrics['prediction_coverage'], 0.)
        self.assertEqual(evaluated.metrics['pixel_accuracy'], 0.)
        self.assertIsNone(evaluated.metrics['overlap_only']['mean_iou'])

    def test_target_roi_and_bottom_exclusion_are_respected(self):
        self.matcher.config.bottom_ignore_fraction = .08
        roi = np.ones_like(self.labels)
        roi[:20] = 0
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()):
            evaluated = self.matcher.evaluate([self.image], [self.labels], self.image,
                                               self.labels, target_valid=roi)
        self.assertEqual(evaluated.score, 100.)
        self.assertFalse(evaluated.ground_truth_valid[:20].any())
        self.assertFalse(evaluated.ground_truth_valid[147:].any())
        self.assertFalse(evaluated.match.label_valid[147:].any())
        self.assertEqual(evaluated.metrics['evaluated_pixels'], (147 - 20) * 200)

    def test_empty_gt_roi_and_background_only_are_not_perfect_foreground_score(self):
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection()):
            evaluated = self.matcher.evaluate([self.image], [self.labels], self.image,
                                               self.labels, input_valid=np.zeros_like(self.labels))
        self.assertIsNone(evaluated.score)
        self.assertEqual(evaluated.metrics['status'], 'no_valid_ground_truth')
        zero = np.zeros_like(self.labels)
        known = np.ones_like(zero, bool)
        score = evaluate_segmentation(zero, zero, known, known)
        self.assertIsNone(score['score'])
        self.assertEqual(evaluate_segmentation(zero, zero, known, known,
                                               include_background=True)['score'], 100.)

    def test_validation_and_config_copy(self):
        self.config.min_score = 100
        self.assertEqual(self.matcher.config.min_score, 60)
        with self.assertRaises(ValueError):
            PeriodicTemplateMatcher(MatcherConfig(label_mode='wrong'))
        with self.assertRaises(ValueError):
            PeriodicTemplateMatcher(MatcherConfig(min_score=101))
        with self.assertRaises(TypeError):
            PeriodicTemplateMatcher({})
        for templates, masks in (([], []), ([self.image], []), ([self.image], [np.zeros((2, 2), np.uint8)])):
            with self.assertRaises(ValueError):
                self.matcher.match(templates, masks, self.image)
        with self.assertRaises(ValueError):
            self.matcher.match([self.image], [self.labels], self.image,
                               annotation_valids=[np.zeros((2, 2))])
        with self.assertRaises(ValueError):
            self.matcher.evaluate([self.image], [self.labels], self.image, np.zeros((2, 2)))


class SegmentationMetricTests(unittest.TestCase):
    def test_hand_calculated_iou_dice_and_accuracy(self):
        truth = np.array([[1, 1], [0, 0]], np.uint16)
        pred = np.array([[1, 0], [1, 0]], np.uint16)
        valid = np.ones_like(truth, bool)
        metrics = evaluate_segmentation(pred, truth, valid, valid)
        self.assertAlmostEqual(metrics['mean_iou'], 1 / 3)
        self.assertAlmostEqual(metrics['mean_dice'], .5)
        self.assertAlmostEqual(metrics['pixel_accuracy'], .5)
        self.assertAlmostEqual(metrics['score'], 100 / 3)
        self.assertEqual(metrics['per_class'][0]['tp'], 1)
        self.assertEqual(metrics['per_class'][0]['fp'], 1)
        self.assertEqual(metrics['per_class'][0]['fn'], 1)

    def test_unknown_foreground_penalized_but_not_converted_to_background(self):
        truth = np.array([[1, 1], [0, 0]], np.uint16)
        pred = truth.copy()
        known = np.array([[1, 0], [1, 1]], bool)
        metrics = evaluate_segmentation(pred, truth, known, np.ones_like(known))
        self.assertEqual(metrics['mean_iou'], .5)
        self.assertEqual(metrics['overlap_only']['mean_iou'], 1.)
        self.assertEqual(metrics['prediction_coverage'], .75)
        self.assertEqual(metrics['uncovered_pixels'], 1)
        self.assertEqual(metrics['per_class'][0]['fn'], 1)
        self.assertEqual(metrics['per_class'][0]['fp'], 0)

    def test_spurious_classes_included_and_gt_unknown_ignored(self):
        truth = np.array([[1, 1], [0, 0]], np.uint16)
        pred = np.array([[1, 2], [0, 4000]], np.uint16)
        known = np.ones_like(truth, bool)
        roi = np.array([[1, 1], [1, 0]], bool)
        metrics = evaluate_segmentation(pred, truth, known, roi)
        self.assertEqual([c['class_id'] for c in metrics['per_class']], [1, 2])
        self.assertEqual(metrics['mean_iou'], .25)
        self.assertEqual(metrics['evaluated_pixels'], 3)

    def test_mask_normalization_rejects_ambiguous_or_lossy_inputs(self):
        mask = np.array([[0, 255], [0, 0]], np.uint8)
        self.assertEqual(normalize_masks(mask, (2, 2))[0, 1], 255)
        self.assertEqual(normalize_masks([mask], (2, 2))[0, 1], 1)
        self.assertFalse(normalize_masks([], (2, 2)).any())
        for invalid in ([mask, mask], np.zeros((2, 2, 3), np.uint8),
                        np.ones((2, 2), np.float32), np.full((2, 2), -1),
                        np.full((2, 2), 65536), [np.full((2, 2), 2)]):
            with self.assertRaises(ValueError):
                normalize_masks(invalid, (2, 2))


if __name__ == '__main__':
    unittest.main()