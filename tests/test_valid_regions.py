"""Valid-height policy must agree across matching, completion, evaluation and IO."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher, run_config
from periodic_template_matcher.utils import read_image, write_image
from tests.helpers import texture


class ValidRegionTests(unittest.TestCase):
    def test_removed_fraction_option_is_rejected_in_flat_and_grouped_options(self):
        for options in ({'bottom_ignore_fraction': .08},
                        {'step1': {'bottom_ignore_fraction': .08}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MatcherConfig.from_options(options)

    def test_default_region_uses_full_image_and_preserves_rotated_mask(self):
        from periodic_template_matcher.utils import ValidRegion
        image = np.zeros((160, 200), np.uint8)
        region = ValidRegion()
        self.assertEqual(region.resolved_height(image), 160)
        self.assertEqual(region.describe(image, region.build_mask(image))['height_source'], 'full_image')
        original = ValidRegion(valid_height=147).build_mask(image)
        for k in (-1, 1):
            rotated_image = np.ascontiguousarray(np.rot90(image, k))
            rotated_mask = np.ascontiguousarray(np.rot90(original, k))
            actual = region.build_mask(rotated_image, rotated_mask)
            np.testing.assert_array_equal(actual, rotated_mask)

    def test_height_intersects_mask_without_mutation(self):
        from periodic_template_matcher.utils import ValidRegion
        image = np.zeros((160, 200, 3), np.uint8)
        mask = np.ones((160, 200), np.uint8)
        mask[:10, :20] = 0
        original = mask.copy()
        region = ValidRegion(valid_height=150)
        actual = region.build_mask(image, mask)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(actual[149, 199])
        self.assertFalse(actual[150:].any())
        self.assertFalse(actual[:10, :20].any())
        np.testing.assert_array_equal(mask, original)
        self.assertEqual(region.crop(image).shape, (150, 200, 3))
        self.assertEqual(ValidRegion(valid_height=144).resolved_height(image), 144)

    def test_invalid_heights_fail_clearly(self):
        from periodic_template_matcher.utils import ValidRegion
        image = np.zeros((160, 200), np.uint8)
        for height in (0, -1, 161, 10.5, True, '150'):
            with self.subTest(height=height), self.assertRaises(ValueError):
                ValidRegion(valid_height=height).build_mask(image)
        with self.assertRaises(ValueError):
            ValidRegion(valid_height=4).build_mask(image)
        for height in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                MatcherConfig(valid_height=height).validate()

    def test_match_resolves_independent_heights_before_registration(self):
        image = texture()
        cfg = MatcherConfig(valid_height=140)
        from periodic_template_matcher.matcher import select_templates as real_select
        with patch('periodic_template_matcher.matcher.select_templates', wraps=real_select) as select:
            result = PeriodicTemplateMatcher(cfg).match(
                [image], [np.ones(image.shape[:2], np.uint16)], image,
                template_valid_heights=[150], target_valid_height=145)
        self.assertTrue(result.accepted)
        self.assertTrue(result.label_valid[140:145, 20:180].all())
        self.assertFalse(result.label_valid[145:].any())
        self.assertIsNone(select.call_args.args[2].valid_height)
        self.assertTrue(select.call_args.args[3][0][149].all())
        self.assertFalse(select.call_args.args[3][0][150:].any())
        self.assertEqual(result.registration['valid_regions']['target']['valid_height'], 145)
        self.assertEqual(result.registration['config']['valid_height'], 140)

    def test_legacy_height_preserves_support_for_thin_valid_bands(self):
        image = np.random.default_rng(41).integers(0, 256, (128, 128), dtype=np.uint8)
        matcher = PeriodicTemplateMatcher(MatcherConfig(valid_height=24, max_side=128,
            angles=(0.,), scales=(1.,), refine_candidates=1, ecc_iterations=10))
        result = matcher.match([image], [None], image)
        self.assertEqual(result.geometric_valid.shape, image.shape)
        self.assertFalse(result.geometric_valid[24:].any())

    def test_rectangle_crops_the_pipeline_and_restores_canvas_and_transform(self):
        from tests.test_matcher_api import selection
        cfg = MatcherConfig.from_options(dict(step1=dict(
            valid_x_start=20, valid_x_end=180, valid_y_start=10, valid_y_end=130)))
        image = texture()
        matcher = PeriodicTemplateMatcher(cfg)
        np.testing.assert_array_equal(matcher.get_valid(image), image[10:130, 20:180])
        d = selection()
        d['matches'][0]['matrix_template_to_target'] = [[1.02, 0., 0.], [0., 1.04, 0.], [0., 0., 1.]]
        local = d['matches'][0]['matrix_template_to_target']
        d['matches'][0]['candidates'] = [dict(matrix_template_to_target=local)]
        d['ordered_objects'] = dict(objects=[
            dict(identity='A', template_index=0, center_x=35., matrix_template_to_target=local,
                 supporting_references=[dict(template_index=1, matrix_template_to_target=local)],
                 supporting_transforms=[local])],
            support_completion=dict(copies=[dict(identity='A', regions=[dict(corrected_matrix=local)])]))
        d['periodic_completion'] = dict(supplemental_templates=[
            dict(template_index=1, periodic_completion=dict(copies=[dict(matrix_template_to_target=local)]))])
        with patch('periodic_template_matcher.matcher.select_templates', return_value=d) as select:
            labels = np.ones(image.shape[:2], np.uint16)
            result = matcher.match([image, image], [labels, labels], image)
        self.assertEqual(select.call_args.args[1].shape[:2], (120, 160))
        np.testing.assert_array_equal(result.input_image, image)
        self.assertEqual(result.label_valid.shape, image.shape[:2])
        self.assertFalse(result.label_valid[:10].any())
        self.assertFalse(result.label_valid[:, :20].any())
        self.assertFalse(result.label_valid[130:].any())
        self.assertFalse(result.label_valid[:, 180:].any())
        self.assertTrue(result.label_valid[20:120, 30:170].all())
        expected = [[1.02, 0., -.4], [0., 1.04, -.4], [0., 0., 1.]]
        np.testing.assert_allclose(result.matrix_template_to_target, expected)
        np.testing.assert_allclose(result.registration['matches'][0]['candidates'][0]
                                   ['matrix_template_to_target'], expected)
        ordered = result.registration['ordered_objects']
        self.assertEqual(ordered['objects'][0]['center_x'], 55.)
        np.testing.assert_allclose(ordered['objects'][0]['supporting_transforms'][0], expected)
        np.testing.assert_allclose(ordered['support_completion']['copies'][0]['regions'][0]['corrected_matrix'],
                                   expected)
        supplemental = result.registration['periodic_completion']['supplemental_templates'][0]
        np.testing.assert_allclose(supplemental['periodic_completion']['copies'][0]['matrix_template_to_target'],
                                   expected)

    def test_footer_does_not_change_matching_for_either_backend(self):
        image = texture()
        changed = image.copy()
        changed[130:] = np.random.default_rng(5).integers(0, 256, changed[130:].shape, dtype=np.uint8)
        for backend in ('fft', 'opencv'):
            with self.subTest(backend=backend):
                matcher = PeriodicTemplateMatcher(MatcherConfig(backend=backend, valid_height=130,
                    max_side=200, angles=(0.,), scales=(1.,), refine_candidates=2, ecc_iterations=20))
                first = matcher.match([image], [None], image)
                second = matcher.match([changed], [None], image)
                self.assertTrue(second.accepted)
                self.assertAlmostEqual(first.match_score, second.match_score, places=5)
                np.testing.assert_allclose(first.matrix_template_to_target, second.matrix_template_to_target, atol=1e-5)

    def test_excluded_footer_cannot_leak_through_downsampling(self):
        rng = np.random.default_rng(2)
        image = cv2.GaussianBlur(rng.integers(20, 230, (500, 600, 3), dtype=np.uint8), (0, 0), 2)
        changed = image.copy()
        changed[391:] = 255
        for backend in ('fft', 'opencv'):
            matcher = PeriodicTemplateMatcher(MatcherConfig(valid_height=391, max_side=200,
                backend=backend, angles=(0.,), scales=(1.,), refine_candidates=1, ecc_iterations=10))
            first = matcher.match([image], [None], image)
            second = matcher.match([changed], [None], image)
            with self.subTest(backend=backend):
                self.assertAlmostEqual(first.match_score, second.match_score, places=5)
                np.testing.assert_allclose(first.matrix_template_to_target, second.matrix_template_to_target, atol=1e-5)

    def test_evaluation_respects_height_and_target_mask(self):
        image = texture()
        labels = np.ones(image.shape[:2], np.uint16)
        mask = np.ones(image.shape[:2], np.uint8)
        mask[:10] = 0
        result = PeriodicTemplateMatcher(MatcherConfig(max_side=200, angles=(0.,), scales=(1.,))).evaluate(
            [image], [labels], image, labels, template_valid_heights=[150],
            target_valid_height=130, target_valid=mask)
        self.assertEqual(result.metrics['evaluated_pixels'], 120*200)
        self.assertFalse(result.ground_truth_valid[130:].any())
        self.assertFalse(result.match.label_valid[:10].any())

    def test_configured_matcher_preloads_assets_and_keeps_results_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = texture()
            write_image(root/'template.png', image)
            write_image(root/'target.png', image)
            write_image(root/'labels.png', np.ones(image.shape[:2], np.uint16))
            path = root/'config.json'
            path.write_text(json.dumps(dict(
                target='target.png', target_valid_height=130,
                templates=[dict(image='template.png', labels='labels.png', valid_height=150)],
                options=dict(max_side=200, angles=[0], scales=[1], template_workers=1,
                             refine_candidates=2, ecc_iterations=20))),
                encoding='utf-8')
            matcher = PeriodicTemplateMatcher(path)
            for name in ('config.json', 'template.png', 'target.png', 'labels.png'):
                (root/name).unlink()
            first = matcher.infer()
            self.assertTrue(first.accepted)
            self.assertTrue((first.mapped_labels[5:120, 20:180] == 1).all())
            first.template_image[:] = 0
            first.input_image[:] = 0
            first.registration['valid_regions']['target']['valid_height'] = 0
            second = matcher.infer()
            self.assertTrue(second.template_image.any())
            self.assertTrue(second.input_image.any())
            self.assertEqual(second.registration['valid_regions']['target']['valid_height'], 130)
            self.assertEqual(matcher.infer_to(root/'out')['prediction_status'], 'available')

    def test_configured_matcher_recomputes_scale_for_a_larger_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = texture()
            target = np.pad(source, ((0, 0), (0, 100), (0, 0)), mode='edge')
            write_image(root/'source.png', source)
            write_image(root/'target.png', target)
            path = root/'config.json'
            options = dict(max_side=200, angles=[0], scales=[1], template_workers=1,
                           refine_candidates=2, ecc_iterations=20)
            path.write_text(json.dumps(dict(templates=[dict(image='source.png')], options=options)),
                            encoding='utf-8')
            matcher = PeriodicTemplateMatcher(path)
            result = matcher.infer('target.png')
            with patch('periodic_template_matcher.matcher.prepare_registration_sources',
                       side_effect=AssertionError('Repeated targets at this scale should reuse source features')):
                repeated = matcher.infer('target.png')
            expected = PeriodicTemplateMatcher(MatcherConfig.from_options(options)).match(
                [source], [None], target)
            self.assertEqual(result.registration['matches'][0]['working_scale'], 200 / target.shape[1])
            self.assertEqual(result.registration['matches'][0]['candidates'],
                             expected.registration['matches'][0]['candidates'])
            self.assertEqual(result.registration, repeated.registration)
            np.testing.assert_array_equal(result.geometric_valid, expected.geometric_valid)

    def test_json_height_exports_and_duplicate_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = texture()
            write_image(root/'image.png', image)
            write_image(root/'labels.png', np.ones(image.shape[:2], np.uint16))
            spec = dict(target='image.png', target_valid_height=130,
                        templates=[dict(image='image.png', labels='labels.png', valid_height=150)],
                        options=dict(valid_x_start=15, valid_x_end=185, valid_y_start=10, valid_y_end=140,
                                     max_side=200,
                                     angles=[0], scales=[1], refine_candidates=2, ecc_iterations=20))
            path = root/'config.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            result = run_config(path, root/'out')
            mask = read_image(root/'out/label_valid.png', 0)
            self.assertEqual(mask.shape, image.shape[:2])
            self.assertTrue(mask[125:130, 20:180].all())
            self.assertFalse(mask[130:].any())
            self.assertFalse(mask[:10].any())
            self.assertFalse(mask[:, :15].any())
            self.assertFalse(mask[:, 185:].any())
            np.testing.assert_array_equal(read_image(root/'out/input.png'), image)
            self.assertEqual(result['valid_regions']['templates'][0]['valid_height'], 150)
            spec['templates'].append(dict(spec['templates'][0], valid_height=149))
            path.write_text(json.dumps(spec), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                run_config(path, root/'out')


if __name__ == '__main__':
    unittest.main()
