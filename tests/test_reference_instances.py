"""Reference-backed periodic holes without a usable neighbouring target seed."""

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import PeriodicTemplateMatcher
from periodic_template_matcher._internal.models import MatchResult
from periodic_template_matcher._internal.objects.axes import transpose_config, transpose_diagnostics
from periodic_template_matcher._internal.segmentation.crop import rebase_diagnostics
from periodic_template_matcher._internal.segmentation.instances import complete_periodic_instances
from periodic_template_matcher._internal.segmentation.reference_instances import complete_reference_instances
from tests import test_periodic_instances as observed_tests


class ReferenceInstanceTests(unittest.TestCase):
    def fixture(self):
        result, valid, cfg = observed_tests.PeriodicInstanceTests().fixture()
        result.registration.update(ranking=[0], best_template_index=0,
            matches=[dict(matrix_template_to_target=np.eye(3).tolist())])
        # The nearest observed contour is occluded by an identity mask. The
        # remaining same-class target seed is TWO periods away from the hole.
        result.mapped_labels[20:44, 100:124] = 2
        result.object_identities[20:44, 100:124] = 4
        source_labels = np.zeros(valid.shape, np.uint16)
        source_labels[20:44, 100:124] = 1
        reference = dict(labels=source_labels, annotation_valid=valid.copy(),
                         source_valid=valid.copy(), template_index=0)
        return result, valid, replace(cfg, class_periodicity={'1': ['x'], '2': ['y']}), reference

    def fill(self, data):
        result, valid, cfg, reference = data
        complete_reference_instances(result, valid, cfg, **reference)

    def test_reference_restores_hole_beyond_observed_seed_without_extending_budget(self):
        result, valid, cfg, reference = self.fixture()
        before = deepcopy(result)
        with patch.object(cv2, 'findTransformECC', side_effect=AssertionError('No new ECC')):
            complete_periodic_instances(result, valid, cfg, reference=reference)
        np.testing.assert_array_equal(result.mapped_labels[20:44, 180:204], 1)
        self.assertEqual(cfg.periodic_max_steps, 1)
        for field in ('object_identities', 'object_valid', 'base_labels', 'base_valid',
                      'label_valid', 'warped_template', 'geometric_valid'):
            np.testing.assert_array_equal(getattr(result, field), getattr(before, field))
        np.testing.assert_array_equal(result.mapped_labels[before.mapped_labels > 0],
                                      before.mapped_labels[before.mapped_labels > 0])
        copies = result.registration['periodic_completion']['reference_instances']['copies']
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0]['source_bbox'], [100, 20, 124, 44])
        self.assertEqual(copies[0]['target_bbox'], [180, 20, 204, 44])
        self.assertEqual(copies[0]['offset'], [1, 0])

    def test_partial_source_annotation_or_source_roi_is_not_a_complete_contour(self):
        for key in ('annotation_valid', 'source_valid'):
            data = self.fixture()
            data[3][key][19, 105] = False
            self.fill(data)
            self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_foreign_source_class_blocks_component_at_its_own_boundary(self):
        data = self.fixture()
        data[3]['labels'][19, 105] = 2
        self.fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_missing_source_class_is_not_invented_from_target_texture(self):
        data = self.fixture()
        data[3]['labels'][:] = 0
        self.fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_reference_from_another_template_or_shape_is_rejected(self):
        for kind in ('template', 'shape'):
            data = self.fixture()
            if kind == 'template':
                data[3]['template_index'] = 1
            else:
                data[3]['annotation_valid'] = data[3]['annotation_valid'][:-1]
            before = data[0].mapped_labels.copy()
            with self.assertRaises(ValueError):
                self.fill(data)
            np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_ordered_identity_annotations_cannot_be_generic_seeds(self):
        data = self.fixture()
        data[3]['labels'][data[3]['labels'] > 0] = 2
        self.fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_known_other_foreground_and_its_identity_are_never_cut_or_overwritten(self):
        data = self.fixture()
        data[0].mapped_labels[25, 185] = 2
        data[0].object_identities[25, 185] = 3
        before = data[0].mapped_labels.copy()
        self.fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)
        self.assertEqual(data[0].object_identities[25, 185], 3)

    def test_destination_unknown_and_roi_holes_cannot_be_certified(self):
        for kind in ('annotation', 'roi'):
            data = self.fixture()
            (data[0].label_valid if kind == 'annotation' else data[1])[25, 185] = False
            self.fill(data)
            self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_mismatching_own_pixels_are_not_hidden_by_matching_context(self):
        data = self.fixture()
        data[0].input_image[20:44, 180:204] = np.random.default_rng(216).integers(
            20, 235, (24, 24), dtype=np.uint8)
        self.fill(data)
        self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_own_pixels_do_not_bypass_edge_or_finite_structure_gate(self):
        for metrics in (dict(structure_ncc=1., edge_alignment=0.),
                        dict(structure_ncc=float('nan'), edge_alignment=1.)):
            data = self.fixture()
            with patch('periodic_template_matcher._internal.segmentation.reference_instances._score',
                       return_value=metrics):
                self.fill(data)
            self.assertFalse(data[0].mapped_labels[20:44, 180:204].any())

    def test_unverified_or_forbidden_axis_is_not_authorized_by_image_similarity(self):
        for kind in ('unverified', 'forbidden'):
            result, valid, cfg, reference = self.fixture()
            if kind == 'unverified':
                result.registration['periodic_completion']['lattice'][0]['verified'] = False
            else:
                cfg = replace(cfg, class_periodicity={'1': ['y']})
            complete_reference_instances(result, valid, cfg, **reference)
            self.assertFalse(result.mapped_labels[20:44, 180:204].any())

    def test_nonshared_and_nonsemantic_modes_keep_original_behavior(self):
        for options in (dict(shared_periods=False), dict(periodic_completion=False),
                        dict(label_mode='instance')):
            result, valid, cfg, reference = self.fixture()
            before = result.mapped_labels.copy()
            complete_reference_instances(result, valid, replace(cfg, **options), **reference)
            np.testing.assert_array_equal(result.mapped_labels, before)

    def test_wrong_short_period_cannot_use_an_out_of_budget_harmonic(self):
        data = self.fixture()
        data[0].registration['periodic_completion']['lattice'] = [dict(
            source_vector=[40., 0.], target_vector=[40., 0.], verified=True)]
        before = data[0].mapped_labels.copy()
        self.fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_source_clipped_contour_cannot_be_projected_as_complete(self):
        data = self.fixture()
        data[3]['labels'][:] = 0
        data[3]['labels'][20:44, :24] = 1
        before = data[0].mapped_labels.copy()
        self.fill(data)
        np.testing.assert_array_equal(data[0].mapped_labels, before)

    def test_source_target_frames_and_template_provenance_rebase_separately(self):
        record = dict(template_index=1, source_bbox=[100, 20, 124, 44],
                      target_bbox=[180, 20, 204, 44],
                      matrix_template_to_target=[[1., 0., 80.], [0., 1., 0.], [0., 0., 1.]])
        swapped = transpose_diagnostics(record)
        self.assertEqual(swapped['source_bbox'], [20, 100, 44, 124])
        self.assertEqual(transpose_diagnostics(swapped), record)
        restored = rebase_diagnostics(record, (7, 11), [(900, 800), (200, 300)], 0)
        self.assertEqual(restored['source_bbox'], [300, 320, 324, 344])
        self.assertEqual(restored['target_bbox'], [187, 31, 211, 55])
        np.testing.assert_array_equal(restored['matrix_template_to_target'],
                                      [[1., 0., -113.], [0., 1., -289.], [0., 0., 1.]])

    def test_transposition_preserves_source_axis_policy_and_labels(self):
        first, valid, cfg, reference = self.fixture()
        second, other_reference = deepcopy(first), deepcopy(reference)
        for field in fields(MatchResult):
            value = getattr(second, field.name)
            if isinstance(value, np.ndarray):
                setattr(second, field.name, np.ascontiguousarray(value.T))
        for key in ('labels', 'source_valid', 'annotation_valid'):
            other_reference[key] = other_reference[key].T.copy()
        second.registration = transpose_diagnostics(second.registration)
        complete_reference_instances(first, valid, cfg, **reference)
        complete_reference_instances(second, valid.T.copy(), transpose_config(cfg), **other_reference)
        np.testing.assert_array_equal(second.mapped_labels.T, first.mapped_labels)

    def test_actual_tnxss2_640_restores_both_user_circled_instances(self):
        root = Path(__file__).resolve().parents[1]
        target = root/'source/SD-1/imgs/TNXSS2/18851_AFE55CS03AA_TDAOI810_50_449.044_-50.227_O_L_20260123_153618.jpg'
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
            with patch('periodic_template_matcher._internal.segmentation.instances.complete_reference_instances'):
                before = matcher.infer()
            after = matcher.infer()
        self.assertTrue(after.accepted)
        self.assertEqual(after.registration['ordered_objects']['visible_order'], 'ABC')
        for field in ('matrix_template_to_target', 'object_identities', 'object_valid',
                      'base_labels', 'base_valid', 'label_valid', 'geometric_valid', 'completed_template'):
            np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
        old = before.mapped_labels > 0
        np.testing.assert_array_equal(after.mapped_labels[old], before.mapped_labels[old])
        copies = after.registration['periodic_completion']['reference_instances']['copies']
        self.assertEqual(len(copies), 2)
        for x, y in ((2255, 490), (2255, 1370)):
            self.assertEqual(before.mapped_labels[y, x], 0)
            self.assertEqual(after.mapped_labels[y, x], 1)
            self.assertTrue(any(x0 <= x < x1 and y0 <= y < y1
                for x0, y0, x1, y1 in (copy['target_bbox'] for copy in copies)))
        self.assertEqual(sum(copy['added_label_pixels'] for copy in copies), 16176)
        self.assertTrue(all(copy['raw_pixel_ncc'] >= matcher.config.periodic_min_ncc for copy in copies))
        self.assertTrue(all(max(abs(n) for n in copy['offset']) <= 1 for copy in copies))


if __name__ == '__main__':
    unittest.main()