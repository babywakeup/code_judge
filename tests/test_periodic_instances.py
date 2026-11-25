"""Independent periodic contours may fill known background, never old foreground."""

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.models import MatchResult
from periodic_template_matcher._internal.objects.axes import transpose_config, transpose_diagnostics
from periodic_template_matcher._internal.segmentation.crop import rebase_diagnostics
from periodic_template_matcher._internal.segmentation.instances import _complete_observed_instances as complete_periodic_instances


class PeriodicInstanceTests(unittest.TestCase):
    def fixture(self):
        tile = np.random.default_rng(128).integers(20, 235, (80, 80), dtype=np.uint8)
        tile = cv2.GaussianBlur(tile, (3, 3), .6)
        image = np.tile(tile, (3, 3))
        labels = np.zeros(image.shape, np.uint16)
        labels[20:44, 20:44] = 1
        known = np.ones(image.shape, bool)
        result = MatchResult(dict(periodic_completion=dict(lattice=[
            dict(source_vector=[80., 0.], target_vector=[80., 0.], verified=True),
            dict(source_vector=[0., 80.], target_vector=[0., 80.], verified=True)])),
            image, image.copy(), image.copy(), known.copy(), labels, known.copy(),
            label_periodic_valid=np.zeros(image.shape, bool),
            object_identities=np.zeros(image.shape, np.uint16), object_valid=known.copy(),
            base_labels=labels.copy(), base_valid=known.copy())
        cfg = MatcherConfig(max_side=240, periodic_completion=True, shared_periods=True,
                            periodic_max_steps=1, ordered_object_class=2,
                            class_periodicity={'1': ['x', 'y'], '2': ['y']})
        return result, known, cfg

    def run_fill(self, data):
        complete_periodic_instances(*data)

    def test_complete_contours_fill_known_background_with_own_evidence(self):
        data = self.fixture()
        before = deepcopy(data[0])
        with patch.object(cv2, 'findTransformECC', side_effect=AssertionError('No new ECC')):
            self.run_fill(data)
        result = data[0]
        for y, x in ((20, 100), (100, 20), (100, 100)):
            np.testing.assert_array_equal(result.mapped_labels[y:y+24, x:x+24], 1)
            self.assertTrue(result.label_periodic_valid[y:y+24, x:x+24].all())
        np.testing.assert_array_equal(result.label_valid, before.label_valid)
        for field in ('base_labels', 'base_valid', 'object_identities', 'object_valid',
                      'geometric_valid', 'warped_template'):
            np.testing.assert_array_equal(getattr(result, field), getattr(before, field))
        records = result.registration['periodic_completion']['interior_instances']
        self.assertEqual(len(records['copies']), 3)
        self.assertEqual(records['added_label_pixels'], 3*24*24)
        self.assertEqual(result.registration['periodic_completion']['added_label_pixels'],
                 int(result.label_periodic_valid.sum()))
        self.assertTrue(all(c['raw_pixel_ncc'] > .99 for c in records['copies']))

    def test_new_copies_cannot_bootstrap_beyond_period_budget(self):
        data = self.fixture()
        self.run_fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())
        self.assertFalse(data[0].mapped_labels[180:204, 20:44].any())

    def test_unresolved_objects_do_not_disable_independent_semantic_periods(self):
        data = self.fixture()
        result = data[0]
        result.object_valid[:] = False
        result.label_valid[:] = result.mapped_labels > 0
        self.run_fill(data)
        np.testing.assert_array_equal(result.mapped_labels[100:124, 100:124], 1)
        self.assertTrue(result.label_valid[100:124, 100:124].all())
        self.assertTrue(result.label_periodic_valid[100:124, 100:124].all())
        self.assertFalse(result.label_valid[result.mapped_labels == 0].any())
        self.assertFalse(result.object_valid.any())
        self.assertFalse(result.object_identities.any())

    def test_verified_target_period_can_finish_a_contour_cut_by_direct_footprint(self):
        data = self.fixture()
        result = data[0]
        result.mapped_labels[100:120, 20:44] = 1
        result.geometric_valid[120:] = False
        result.label_valid[120:] = False
        result.direct_labels = result.mapped_labels.copy()
        result.direct_valid = result.geometric_valid.copy()
        result.registration['periodic_completion']['lattice'][1]['target_verification'] = {'verified': True}
        with_annotation_hole = deepcopy(result)
        with_annotation_hole.geometric_valid[120, 25] = True
        self.run_fill((with_annotation_hole, *data[1:]))
        self.assertFalse(with_annotation_hole.label_valid[120, 25])
        self.run_fill(data)
        np.testing.assert_array_equal(result.mapped_labels[100:124, 20:44], 1)
        self.assertTrue(result.label_periodic_valid[120:124, 20:44].all())
        self.assertFalse(result.label_periodic_valid[100:120, 20:44].any())
        self.assertFalse(result.label_valid[124:].any())
        self.assertFalse(result.label_valid[120:124, :20].any())

    def test_tnxxx0_77_recovers_periodic_foreground_below_the_direct_footprint(self):
        root = Path(__file__).resolve().parents[1]
        image = root/'source/SD-1/imgs/TNXXX0/18851_AFE55CN029A_TDAOI810_77_225.545_79.559_O_M_20260119_085609.jpg'
        if not image.is_file():
            self.skipTest('Local SD-1 photograph is not installed')
        result = PeriodicTemplateMatcher(root/'periodic_template_matcher/config.sd1.labelme.opencv.json').infer(image)
        for x, y in ((381, 1804), (1164, 1801), (2041, 1801)):
            with self.subTest(x=x, y=y):
                self.assertFalse(result.geometric_valid[y, x])
                self.assertEqual(result.direct_labels[y, x], 0)
                self.assertEqual(result.completion_labels[y, x], 1)
                self.assertEqual(result.mapped_labels[y, x], 1)
        self.assertEqual(result.registration['ordered_objects']['visible_order'], 'ABC')
        self.assertFalse(result.completion_labels[1891:].any())

    def test_nonshared_disabled_and_nonsemantic_modes_keep_old_policy(self):
        for options in (dict(shared_periods=False), dict(periodic_completion=False),
                        dict(label_mode='instance')):
            result, valid, cfg = self.fixture()
            before = result.mapped_labels.copy()
            complete_periodic_instances(result, valid, replace(cfg, **options))
            np.testing.assert_array_equal(result.mapped_labels, before)

    def test_unverified_axis_cannot_propose_cardinal_or_diagonal_fill(self):
        data = self.fixture()
        data[0].registration['periodic_completion']['lattice'][0]['verified'] = False
        self.run_fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 100:124].any())
        self.assertFalse(data[0].mapped_labels[100:124, 100:124].any())
        self.assertTrue(data[0].mapped_labels[100:124, 20:44].all())

    def test_forbidden_source_axis_cannot_be_authorized_by_target_direction(self):
        result, valid, cfg = self.fixture()
        complete_periodic_instances(result, valid, replace(cfg, class_periodicity={'1': ['y']}))
        self.assertFalse(result.mapped_labels[20:44, 100:124].any())
        self.assertTrue(result.mapped_labels[100:124, 20:44].all())

    def test_ordered_identity_contours_are_not_generic_periodic_seeds(self):
        data = self.fixture()
        data[0].mapped_labels[data[0].mapped_labels > 0] = 2
        before = data[0].mapped_labels.copy()
        self.run_fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_other_foreground_blocks_whole_destination_not_just_overlap(self):
        for class_id in (1, 2, 1001):
            data = self.fixture()
            data[0].mapped_labels[25, 105] = class_id
            data[0].object_identities[25, 105] = 4
            before = data[0].mapped_labels[20:44, 100:124].copy()
            self.run_fill(data)
            np.testing.assert_array_equal(data[0].mapped_labels[20:44, 100:124], before)
            self.assertEqual(data[0].object_identities[25, 105], 4)

    def test_unknown_seed_halo_does_not_establish_complete_contour(self):
        data = self.fixture()
        data[0].label_valid[19, 25] = False
        before = data[0].mapped_labels.copy()
        self.run_fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_foreign_class_occlusion_does_not_establish_complete_seed(self):
        data = self.fixture()
        data[0].mapped_labels[19, 25] = 2
        before = data[0].mapped_labels.copy()
        self.run_fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_unknown_destination_and_roi_hole_are_not_background(self):
        for source in ('annotation', 'roi'):
            data = self.fixture()
            (data[0].label_valid if source == 'annotation' else data[1])[25, 105] = False
            self.run_fill(data)
            self.assertFalse(data[0].mapped_labels[20:44, 100:124].any())

    def test_matching_context_cannot_hide_mismatching_own_pixels(self):
        data = self.fixture()
        data[0].input_image[20:44, 100:124] = np.random.default_rng(39).integers(
            20, 235, (24, 24), dtype=np.uint8)
        self.run_fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 100:124].any())

    def test_own_pixel_match_does_not_bypass_edge_gate(self):
        for metrics in (dict(structure_ncc=1., edge_alignment=0.),
                        dict(structure_ncc=float('nan'), edge_alignment=1.),
                        dict(structure_ncc=1., edge_alignment=float('nan'))):
            data = self.fixture()
            before = data[0].mapped_labels.copy()
            with patch('periodic_template_matcher._internal.segmentation.instances._score',
                       return_value=metrics):
                self.run_fill(data)
            np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_subpixel_geometry_does_not_smooth_raw_pixel_evidence(self):
        data = self.fixture()
        data[0].registration['periodic_completion']['lattice'] = [dict(
            source_vector=[80., 0.], target_vector=[80.35, .2], verified=True)]
        self.run_fill(data)
        copies = data[0].registration['periodic_completion']['interior_instances']['copies']
        self.assertEqual(len(copies), 1)
        self.assertAlmostEqual(copies[0]['raw_pixel_ncc'], 1., places=6)
        self.assertEqual(copies[0]['target_bbox'], [100, 20, 124, 44])

    def test_image_clipped_source_component_is_not_a_complete_seed(self):
        data = self.fixture()
        data[0].mapped_labels[:] = 0
        data[0].mapped_labels[20:44, :24] = 1
        before = data[0].mapped_labels.copy()
        self.run_fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_flat_appearance_is_not_positive_evidence(self):
        data = self.fixture()
        data[0].input_image[:] = 100
        before = data[0].mapped_labels.copy()
        self.run_fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_transposed_target_and_source_axis_policy_are_equivariant(self):
        first, valid, cfg = self.fixture()
        second = deepcopy(first)
        for field in fields(MatchResult):
            value = getattr(second, field.name)
            if isinstance(value, np.ndarray):
                setattr(second, field.name, np.ascontiguousarray(value.T))
        second.registration = transpose_diagnostics(second.registration)
        complete_periodic_instances(first, valid, cfg)
        complete_periodic_instances(second, valid.T.copy(), transpose_config(cfg))
        np.testing.assert_array_equal(second.mapped_labels.T, first.mapped_labels)

    def test_wrong_short_period_cannot_borrow_valid_two_step_harmonic(self):
        data = self.fixture()
        data[0].registration['periodic_completion']['lattice'] = [dict(
            source_vector=[40., 0.], target_vector=[40., 0.], verified=True)]
        before = data[0].mapped_labels.copy()
        self.run_fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_competing_aliases_do_not_union_or_repeat_acceptance(self):
        data = self.fixture()
        data[0].mapped_labels[100:124, 20:44] = 1
        self.run_fill(data)
        copies = data[0].registration['periodic_completion']['interior_instances']['copies']
        self.assertEqual(sum(c['target_bbox'] == [100, 100, 124, 124] for c in copies), 1)

    def test_source_and_destination_target_boxes_rebase_and_transpose(self):
        value = dict(copies=[dict(source_target_bbox=[20, 30, 44, 54],
                                 target_bbox=[100, 110, 124, 134], target_vector=[80., 80.])])
        swapped = transpose_diagnostics(value)
        self.assertEqual(swapped['copies'][0]['source_target_bbox'], [30, 20, 54, 44])
        self.assertEqual(transpose_diagnostics(swapped), value)
        restored = rebase_diagnostics(value, (7, 11), [(400, 500)], 0)['copies'][0]
        self.assertEqual(restored['source_target_bbox'], [27, 41, 51, 65])
        self.assertEqual(restored['target_bbox'], [107, 121, 131, 145])
        self.assertEqual(restored['target_vector'], [80., 80.])

    def test_actual_tnxdp0_640_recovers_four_independently_verified_instances(self):
        root = Path(__file__).resolve().parents[1]
        target = root/'source/SD-1/imgs/TNXDP0/18851_AFE55CN01KA_TDAOI810_59_-446.009_132.990_O_M_20260119_123334.jpg'
        if not target.is_file():
            self.skipTest('Local SD-1 photographs are not installed')
        config = root/'periodic_template_matcher/config.sd1.labelme.opencv.json'
        spec = json.loads(config.read_text(encoding='utf-8'))
        for template in spec['templates']:
            for key in ('image', 'labelme'):
                template[key] = str((config.parent/template[key]).resolve())
        spec['target'] = str(target)
        spec['options']['step1'].update(max_side=640, scale=None)
        spec['options']['step2']['template_workers'] = 1
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)/'config.json'
            snapshot.write_text(json.dumps(spec), encoding='utf-8')
            matcher = PeriodicTemplateMatcher(snapshot)
            # Isolate the target-to-target fallback. Directly supported
            # contours take precedence when the normal pipeline can find them.
            with patch('periodic_template_matcher._internal.pipeline.complete_direct_references'):
                with patch('periodic_template_matcher._internal.pipeline.complete_periodic_instances'):
                    before = matcher.infer()
                after = matcher.infer()
        self.assertTrue(after.accepted)
        self.assertEqual(after.registration['ordered_objects']['visible_order'], 'CDA')
        for field in ('matrix_template_to_target', 'object_identities', 'object_valid',
                      'base_labels', 'base_valid', 'label_valid', 'geometric_valid'):
            np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
        foreground = before.mapped_labels > 0
        np.testing.assert_array_equal(after.mapped_labels[foreground], before.mapped_labels[foreground])
        copies = after.registration['periodic_completion']['interior_instances']['copies']
        self.assertEqual(len(copies), 4)
        for x, y in ((2070, 470), (1530, 590), (2070, 1350), (1530, 1470)):
            self.assertTrue(any(x0 <= x < x1 and y0 <= y < y1
                                for x0, y0, x1, y1 in (c['target_bbox'] for c in copies)))
        self.assertTrue(all(c['raw_pixel_ncc'] > .95 for c in copies))
        self.assertGreater(int(((after.mapped_labels == 1) & (before.mapped_labels == 0)).sum()), 26000)


if __name__ == '__main__':
    unittest.main()
