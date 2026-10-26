"""Deterministic registration, annotation IO and output safety regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, run_config
from periodic_template_matcher._internal.registration import register, select_templates
from periodic_template_matcher._internal.annotations import load_labels
from periodic_template_matcher._internal.workflow import unique_templates
from periodic_template_matcher.utils import read_image, warp_labels, write_image
from tests.helpers import texture


class RegistrationTests(unittest.TestCase):
    def test_valid_mask_projection_preserves_holes_and_target_bounds(self):
        from periodic_template_matcher.utils import warp_valid_mask
        mask = np.ones((3, 4), np.uint8)
        mask[1, 1] = 0
        H = np.array([[1., 0, 2], [0, 1., 1], [0, 0, 1.]])
        actual = warp_valid_mask(mask, H, (5, 5))
        expected = np.zeros((5, 5), bool)
        expected[1:4, 2:5] = True
        expected[2, 3] = False
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(mask, [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]])

    def test_multiple_templates_share_target_features_and_period_detection(self):
        from periodic_template_matcher._internal import registration
        source = texture()
        with patch.object(registration, '_features', wraps=registration._features) as features:
            with patch.object(registration, '_periods', wraps=registration._periods) as periods:
                result = select_templates([source, texture(99)], source, self.cfg)
        self.assertEqual(result['ranking'][0], 0)
        self.assertEqual(features.call_count, 3)  # Two templates and one target.
        self.assertEqual(periods.call_count, 1)

    def test_shared_preprocessing_matches_standalone_with_mixed_sizes(self):
        templates = [texture(), texture(99)[:140, :180]]
        target = texture()
        cfg = MatcherConfig(max_side=150, angles=(0.,), scales=(1.,))
        ranked = select_templates(templates, target, cfg)
        for index, template in enumerate(templates):
            direct = register(template, target, cfg, working_scale=.75)
            match = ranked['matches'][index]
            self.assertAlmostEqual(match['match_score'], direct['match_score'], places=7)
            self.assertEqual(match['working_target_size'], [150, 120])
            np.testing.assert_allclose(match['matrix_template_to_target'],
                                       direct['matrix_template_to_target'], atol=1e-7)

    def test_explicit_scale_overrides_max_side_for_both_backends(self):
        source = texture()
        for backend in ('fft', 'opencv'):
            for scale, size in ((1., [200, 160]), (.5, [100, 80])):
                with self.subTest(backend=backend, scale=scale):
                    cfg = MatcherConfig(scale=scale, max_side=96, backend=backend,
                                        angles=(0.,), scales=(1.,))
                    direct = register(source, source, cfg)
                    ranked = select_templates([source], source, cfg)['matches'][0]
                    for result in (direct, ranked):
                        self.assertEqual(result['working_scale'], scale)
                        self.assertEqual(result['working_target_size'], size)
                        self.assertTrue(result['geometry_accepted'])
                        np.testing.assert_allclose(result['matrix_template_to_target'],
                                                   np.eye(3), atol=.1)

    def test_scale_validation(self):
        for scale in (0, -.5, 1.1, float('nan'), float('inf'), True, '0.5'):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'scale'):
                MatcherConfig(scale=scale).validate()

    def setUp(self):
        self.cfg = MatcherConfig(max_side=200, angles=(0.,), scales=(1.,),
                          refine_candidates=3, peaks_per_hypothesis=3, ecc_iterations=35)

    def test_translation_and_rotation_direction(self):
        source = texture()
        H = np.vstack([cv2.getRotationMatrix2D((100, 80), .8, 1.01), [0, 0, 1]])
        H[:2, 2] += [5, -4]
        target = cv2.warpPerspective(source, H, (200, 160))
        result = register(source, target, self.cfg)
        actual = np.asarray(result['matrix_template_to_target'])
        points = np.array([[[40., 40.], [150., 110.], [100., 80.]]], np.float32)
        error = np.linalg.norm(cv2.perspectiveTransform(points, H) -
                               cv2.perspectiveTransform(points, actual), axis=2)
        self.assertTrue(result['geometry_accepted'])
        self.assertLess(float(error.max()), 1.5)

    def test_ranking_and_equal_template_rejection(self):
        source = texture()
        result = select_templates([texture(99), source], source, self.cfg)
        self.assertEqual(result['best_template_index'], 1)
        self.assertTrue(result['accepted'])
        ambiguous = select_templates([source, source], source, self.cfg)
        self.assertEqual(ambiguous['status'], 'templates_too_close')
        self.assertFalse(ambiguous['accepted'])

    def test_blank_rejected(self):
        blank = np.zeros((100, 100, 3), np.uint8)
        result = select_templates([blank], blank, self.cfg)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['best_template_index'])

    def test_integer_labels_valid_background_and_direction(self):
        labels = np.zeros((40, 50), np.uint16)
        labels[10:15, 12:18] = 1001
        valid = np.zeros_like(labels)
        valid[5:25, 5:30] = 1
        H = np.array([[1., 0, 7], [0, 1., 3], [0, 0, 1.]])
        mapped, known = warp_labels(labels, H, (45, 60), valid)
        self.assertEqual(mapped[13, 19], 1001)
        self.assertTrue(known[8, 12])
        self.assertEqual(mapped[8, 12], 0)  # Real background remains known.
        self.assertFalse(known[0, 0])
        self.assertEqual(set(np.unique(mapped)), {0, 1001})
        self.assertEqual(mapped.dtype, np.uint16)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_image(self.root / 'template.png', texture())
        write_image(self.root / 'target.png', texture())

    def config(self, **extra):
        spec = {'target': 'target.png', 'templates': [{'image': 'template.png', 'name': '<template>'}],
                'options': {'max_side': 200, 'angles': [0], 'scales': [1], 'refine_candidates': 2}}
        spec.update(extra)
        path = self.root / 'config.json'
        path.write_text(json.dumps(spec), encoding='utf-8')
        return path

    def test_duplicate_content_collapsed_and_conflicts_rejected(self):
        write_image(self.root / 'copy.png', texture())
        specs = [{'image': 'template.png'}, {'image': 'copy.png'}]
        unique, aliases = unique_templates(specs, lambda p: self.root / p)
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(aliases[0]), 2)
        specs[1]['labels'] = 'different.png'
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            unique_templates(specs, lambda p: self.root / p)

    def test_labelme_classes_and_invalid_shapes(self):
        annotation = {'imageHeight': 160, 'imageWidth': 200, 'shapes': [
            {'label': 'wire', 'shape_type': 'polygon', 'points': [[20, 20], [40, 20], [30, 40]]},
            {'label': 'via', 'shape_type': 'circle', 'points': [[70, 60], [76, 60]]},
            {'label': 'wire', 'shape_type': 'rectangle', 'points': [[120, 100], [110, 90]]}]}
        path = self.root / 'annotation.json'
        path.write_text(json.dumps(annotation), encoding='utf-8')
        spec = {'labelme': 'annotation.json', 'class_map': {'wire': 257, 'via': 2}}
        labels = load_labels(spec, lambda p: self.root / p, (160, 200))
        self.assertEqual(labels[25, 30], 257)
        self.assertEqual(labels[60, 70], 2)
        self.assertEqual(labels[95, 115], 257)
        self.assertEqual(labels[0, 0], 0)
        spec['class_map'] = {'wire': 1}
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            load_labels(spec, lambda p: self.root / p, (160, 200))
        annotation['shapes'][0]['shape_type'] = 'line'
        path.write_text(json.dumps(annotation), encoding='utf-8')
        spec['class_map'] = {'wire': 1, 'via': 2}
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            load_labels(spec, lambda p: self.root / p, (160, 200))

    def test_real_transfer_then_demo_cleans_stale_outputs(self):
        labels = np.zeros((160, 200), np.uint16)
        labels[35:65, 40:70] = 321
        write_image(self.root / 'labels.png', labels)
        known = np.zeros((160, 200), np.uint8)
        known[20:80, 20:90] = 255
        write_image(self.root / 'known.png', known)
        out = self.root / 'out'
        config = self.config(templates=[{'image': 'template.png', 'labels': 'labels.png',
                                         'annotation_valid': 'known.png', 'name': '<template>'}])
        result = run_config(config, out)
        self.assertTrue(result['labels_exported'])
        mapped = read_image(out / 'mapped_labels.png', cv2.IMREAD_UNCHANGED)
        valid = read_image(out / 'label_valid.png', cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mapped[50, 55], 321)
        self.assertEqual(valid[0, 0], 0)
        self.assertEqual(valid[25, 25], 255)
        self.assertIn('&lt;template&gt;', (out / 'report.html').read_text(encoding='utf-8'))
        config = self.config(demo_annotations=True)
        result = run_config(config, out)
        self.assertFalse(result['labels_exported'])
        self.assertTrue(result['demo_labels_exported'])
        self.assertFalse((out / 'mapped_labels.png').exists())
        self.assertFalse((out / 'label_valid.png').exists())
        self.assertTrue((out / 'demo_mapped_labels.png').exists())
        # Rejected reruns must not leave even illustrative maps behind.
        write_image(self.root / 'target.png', np.zeros((160, 200, 3), np.uint8))
        result = run_config(config, out)
        self.assertFalse(result['accepted'])
        self.assertFalse((out / 'demo_mapped_labels.png').exists())
        self.assertTrue((out / 'report.html').exists())

    def test_instance_mode_blocks_periodic_label_export(self):
        labels = np.ones((160, 200), np.uint8)
        write_image(self.root / 'labels.png', labels)
        config = self.config(label_mode='instance', templates=[{'image': 'template.png', 'labels': 'labels.png'}])
        # Exercise the IO safety gate with explicit periodic/ambiguous evidence.
        original = select_templates([texture()], texture(), MatcherConfig(max_side=200, angles=(0,), scales=(1,)))
        original.update(accepted=False, status='absolute_cell_unverified', absolute_mapping_accepted=False)
        original['matches'][0].update(periodic_detected=True, absolute_cell_verified=False)
        with patch('periodic_template_matcher.matcher.select_templates', return_value=original):
            result = run_config(config, self.root / 'out')
        self.assertFalse(result['labels_exported'])
        self.assertEqual(result['annotation_status'], 'blocked_by_match')
        self.assertFalse((self.root / 'out' / 'mapped_labels.png').exists())


if __name__ == '__main__':
    unittest.main()
