"""Periodic completion transfers real source labels into uncovered target ROI."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher, run_config
from periodic_template_matcher.utils import write_image, read_image


def selection(accepted, matrix):
    return dict(accepted=accepted, best_template_index=0, ranking=[0],
                status='accepted' if accepted else 'rejected',
                matches=[dict(template_index=0, match_score=90.,
                              matrix_template_to_target=matrix.tolist())])


def fixture():
    rng = np.random.default_rng(29)
    tile = rng.integers(25, 230, (80, 80), dtype=np.uint8)
    tile = cv2.GaussianBlur(tile, (5, 5), 1)
    source = np.tile(tile, (3, 3))
    labels = np.zeros((80, 80), np.uint16)
    labels[20:42, 15:35] = 1
    labels[48:65, 45:65] = 1001
    labels = np.tile(labels, (3, 3))
    # Infinite periodic target with negative source-to-target translation.
    target = np.roll(source, (-9, -12), axis=(0, 1))
    truth = np.roll(labels, (-9, -12), axis=(0, 1))
    H = np.array([[1., 0, -12], [0, 1., -9], [0, 0, 1]])
    return source, labels, target, truth, H


class PeriodicCompletionTests(unittest.TestCase):
    def cfg(self, **kwargs):
        options = dict(max_side=240, angles=(0.,), scales=(1.,), ecc_iterations=25,
                       periodic_completion=True, period_vectors=((80., 0.), (0., 80.)))
        options.update(kwargs)
        return MatcherConfig(**options)

    def matched(self, annotations=None, cfg=None, target=None, accepted=True):
        source, labels, image, truth, H = fixture()
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(accepted, H)):
            result = PeriodicTemplateMatcher(cfg or self.cfg()).match(
                [source], [labels], image if target is None else target,
                annotation_valids=[annotations])
        return result, truth

    def test_completes_border_corner_and_preserves_direct_labels(self):
        result, truth = self.matched()
        self.assertTrue(result.label_valid.all())
        self.assertTrue(result.completed_valid.all())
        self.assertFalse(result.geometric_valid[-1, -1])
        self.assertTrue(result.periodic_valid[-1, -1])
        np.testing.assert_array_equal(result.mapped_labels, truth)
        self.assertEqual(result.registration['periodic_completion']['status'], 'complete')

    def test_infers_period_from_anchor_components(self):
        result, truth = self.matched(cfg=self.cfg(period_vectors=()))
        self.assertTrue(result.label_valid.all())
        np.testing.assert_array_equal(result.mapped_labels, truth)
        self.assertEqual(result.registration['periodic_completion']['period_source'], 'anchor_class')

    def test_partial_annotations_not_promoted_to_background(self):
        ann = np.zeros((240, 240), np.uint8)
        ann[20:42, 15:35] = 1
        result, _ = self.matched(annotations=ann)
        self.assertFalse(result.label_valid.all())
        self.assertTrue(np.all(result.mapped_labels[result.label_valid] == 1))
        self.assertEqual(result.registration['periodic_completion']['status'], 'incomplete')

    def test_invalid_roi_and_rejection_are_respected(self):
        result, _ = self.matched(cfg=self.cfg(bottom_ignore_fraction=.08))
        end = int(240*.92)
        self.assertTrue(result.label_valid[:end].all())
        self.assertFalse(result.label_valid[end:].any())
        result, _ = self.matched(accepted=False)
        self.assertIsNone(result.mapped_labels)
        self.assertIsNone(result.completed_template)

    def test_explicit_heights_control_periodic_completion_roi(self):
        source, labels, target, _, H = fixture()
        mask = np.ones(target.shape, np.uint8)
        mask[80:95, 80:95] = 0
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)):
            result = PeriodicTemplateMatcher(self.cfg(bottom_ignore_fraction=.2)).match(
                [source], [labels], target, template_valid_heights=[210],
                target_valid_height=220, target_valid=mask)
        self.assertFalse(result.label_valid[220:].any())
        self.assertFalse(result.periodic_valid[220:].any())
        self.assertFalse(result.label_valid[80:95, 80:95].any())
        self.assertTrue(result.label_valid[210:220].all())
        self.assertEqual(result.registration['periodic_completion']['status'], 'complete')

    def test_mismatched_image_does_not_fill(self):
        image = np.random.default_rng(53).integers(0, 255, (240, 240), dtype=np.uint8)
        result, _ = self.matched(target=image)
        self.assertFalse(result.periodic_valid.any())
        self.assertFalse(result.label_valid.all())

    def test_mismatched_missing_border_is_not_hidden_by_matching_interior(self):
        _, _, target, _, _ = fixture()
        target[:, -12:] = np.random.default_rng(59).integers(0, 255, (240, 12), dtype=np.uint8)
        result, _ = self.matched(target=target)
        self.assertFalse(result.label_valid[:, -12:].any())
        self.assertEqual(result.registration['periodic_completion']['status'], 'incomplete')

    def test_no_period_and_invalid_options(self):
        for kwargs in (dict(periodic_completion='yes'), dict(period_vectors=((0, 0),)),
                       dict(periodic_max_steps=0), dict(periodic_min_ncc=2)):
            with self.assertRaises(ValueError):
                self.cfg(**kwargs).validate()
        from tests.helpers import texture
        image = texture()
        H = np.array([[1., 0, -8], [0, 1., -6], [0, 0, 1]])
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)):
            result = PeriodicTemplateMatcher(self.cfg(period_vectors=())).match(
                [image], [np.zeros(image.shape[:2], np.uint16)], image)
        self.assertEqual(result.registration['periodic_completion']['status'], 'no_period')

    def test_public_backends_and_exports(self):
        source, labels, target, truth, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, image in [('source.png', source), ('target.png', target), ('labels.png', labels)]:
                write_image(root/name, image)
            for backend in ('fft', 'opencv'):
                with self.subTest(backend=backend):
                    cfg = self.cfg(backend=backend, refine_candidates=3)
                    from dataclasses import asdict
                    spec = dict(target='target.png', templates=[dict(image='source.png', labels='labels.png')],
                                options=asdict(cfg))
                    path = root/'config.json'
                    path.write_text(json.dumps(spec), encoding='utf-8')
                    result = run_config(path, root/backend)
                    self.assertTrue(result['accepted'])
                    self.assertEqual(result['periodic_completion']['status'], 'complete')
                    np.testing.assert_array_equal(read_image(root/backend/'label_valid.png', 0), 255)
                    self.assertTrue((root/backend/'periodic_overlay.html').exists())
                    self.assertTrue((root/backend/'label_periodic_valid.png').exists())
                    spec['options']['periodic_completion'] = False
                    path.write_text(json.dumps(spec), encoding='utf-8')
                    run_config(path, root/backend)
                    self.assertFalse((root/backend/'periodic_overlay.html').exists())


if __name__ == '__main__':
    unittest.main()
