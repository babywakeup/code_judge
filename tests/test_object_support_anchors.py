"""Canonical periodic anchors and two-sided support through interior defects."""
import unittest
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import cv2

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.objects.completion import complete_object_support
from periodic_template_matcher._internal.objects.axes import transpose_diagnostics
from periodic_template_matcher._internal.objects.matching import ObjectPrediction
from periodic_template_matcher._internal.objects.support_anchors import canonical_support_objects, interior_bridges, periodic_affine
from periodic_template_matcher._internal.registration.scoring import _ecc_translation
from periodic_template_matcher._internal.segmentation.crop import rebase_diagnostics
from tests import test_object_completion as object_tests
from tests.test_ordered_objects import scene


class ObjectSupportAnchorTests(unittest.TestCase):
    def fixture(self):
        source, target, refs, truth, prediction = object_tests.ObjectCompletionTests().fixture()
        prediction.labels[:] = truth
        prediction.identities[:] = np.tile(scene()[3], (3, 1))
        prediction.valid[:] = True
        prediction.diagnostics['periodic_lattice'] = [dict(
            source_vector=[0., 100.], target_vector=[0., 100.], verified=True)]
        cfg = MatcherConfig(max_side=640, ordered_object_class=2, shared_periods=True,
                            periodic_max_steps=1)
        return source, target, refs, truth, prediction, cfg

    def run_support(self, data, vectors=((0., 100.),)):
        source, target, refs, _, prediction, cfg = data
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), vectors, cfg)

    def erase(self, prediction, start, end):
        prediction.labels[start:end, 26:54] = 0
        prediction.identities[start:end, 26:54] = 0
        prediction.valid[start:end, 26:54] = False

    def test_observed_supporting_occurrence_is_used_instead_of_off_canvas_primary(self):
        data = self.fixture()
        self.erase(data[4], 200, 300)
        obj = data[4].diagnostics['objects'][0]
        obj['supporting_references'] = [dict(template_index=0, reference_index=0,
            matrix_template_to_target=np.eye(3).tolist())]
        obj['matrix_template_to_target'][1][2] = 1e8
        self.run_support(data)
        np.testing.assert_array_equal(data[4].labels, data[3])
        self.assertEqual(data[5].periodic_max_steps, 1)

    def test_interior_defect_has_two_verified_same_identity_anchors(self):
        data = self.fixture()
        self.erase(data[4], 100, 200)
        data[1][100:200, 26:54] = np.random.default_rng(26).integers(
            20, 200, (100, 28), dtype=np.uint8)
        self.run_support(data)
        np.testing.assert_array_equal(data[4].labels, data[3])
        self.assertTrue((data[4].identities[100:200, 26:54] == 1).all())
        self.assertGreater(data[4].diagnostics['support_completion']['bridged_foreground_pixels'], 0)

    def test_one_sided_damage_is_not_extrapolated(self):
        data = self.fixture()
        self.erase(data[4], 200, 300)
        data[1][200:, :80] = np.random.default_rng(77).integers(0, 255, (100, 80), dtype=np.uint8)
        before = data[4].valid.copy()
        self.run_support(data)
        np.testing.assert_array_equal(data[4].valid, before)

    def test_other_identity_cannot_supply_a_bridge_anchor(self):
        data = self.fixture()
        self.erase(data[4], 100, 200)
        data[1][100:200, 26:54] = 100
        data[4].identities[200:, 26:54] = 2
        self.run_support(data)
        self.assertFalse(data[4].valid[100:200, 26:54].any())
        self.assertTrue((data[4].identities[200:, 26:54] == 2).all())

    def test_unverified_period_and_partial_reference_never_bridge(self):
        for reason in ('unverified', 'partial'):
            data = self.fixture()
            self.erase(data[4], 100, 200)
            data[1][100:200, 26:54] = 100
            if reason == 'unverified':
                data[4].diagnostics['periodic_lattice'][0]['verified'] = False
            else:
                data[4].diagnostics['objects'][0]['reference_partial'] = True
            with self.subTest(reason=reason):
                self.run_support(data)
                self.assertFalse(data[4].valid[100:200, 26:54].any())

    def test_bridge_does_not_certify_surrounding_unknown_background(self):
        data = self.fixture()
        self.erase(data[4], 100, 200)
        data[4].valid[100:200, 20:60] = False
        data[1][100:200, 20:60] = 100
        self.run_support(data)
        self.assertTrue(data[4].valid[100:200, 26:54].all())
        self.assertFalse(data[4].valid[100:200, 20:26].any())

    def test_partial_primary_does_not_hide_complete_supporting_reference(self):
        data = self.fixture()
        self.erase(data[4], 100, 200)
        data[1][100:200, 26:54] = 100
        obj = data[4].diagnostics['objects'][0]
        obj['reference_partial'] = True
        obj['supporting_references'] = [dict(template_index=0, reference_index=0,
            matrix_template_to_target=np.eye(3).tolist())]
        self.run_support(data)
        self.assertTrue(data[4].valid[100:200, 26:54].all())

    def test_partially_missing_rows_can_use_their_own_two_ends(self):
        data = self.fixture()
        p = data[4]
        p.labels[100:200, 26:40] = 0
        p.identities[100:200, 26:40] = 0
        p.valid[100:200, 26:40] = False
        data[1][100:200, 26:40] = 100
        self.run_support(data)
        self.assertTrue(p.valid[100:200, 26:40].all())
        np.testing.assert_array_equal(p.labels, data[3])

    def bridge_fixture(self):
        mapped = np.zeros((240, 160), np.uint16)
        mapped[:, 20:80] = 2
        mapped[:, 100:104] = 2
        observed = np.random.default_rng(42).integers(20, 200, mapped.shape, dtype=np.uint8)
        target = observed.copy()
        known = np.ones(mapped.shape, bool)
        known[80:160] &= mapped[80:160] == 0
        ids = ((mapped > 0) & known).astype(np.uint16)
        p = ObjectPrediction(np.where(known, mapped, 0), ids, known, {})
        return mapped, p, observed, target

    def bridge(self, data):
        mapped, p, observed, target = data
        def verify(region, ncc):
            return dict(region_ncc=ncc) if ncc >= .72 else None
        return interior_bridges(mapped, p, 1, observed, target, np.zeros(mapped.shape, bool),
                                100, 40, verify)

    def test_small_bad_component_cannot_borrow_large_good_components_ends(self):
        data = self.bridge_fixture()
        data[3][:, 100:104] = 220-data[2][:, 100:104]
        bridge, records = self.bridge(data)
        self.assertTrue(bridge[80:160, 20:80].all())
        self.assertFalse(bridge[:, 100:104].any())
        self.assertEqual(len(records), 1)

    def test_one_correct_pixel_among_other_identity_is_not_an_anchor(self):
        data = self.bridge_fixture()
        p = data[1]
        p.identities[:80, 100:104] = 2
        p.identities[160:, 100:104] = 2
        p.identities[79, 100] = p.identities[160, 100] = 1
        bridge, _ = self.bridge(data)
        self.assertFalse(bridge[:, 100:104].any())

    def test_defect_crossing_nearest_anchor_uses_further_healthy_strip_within_budget(self):
        data = self.bridge_fixture()
        data[3][160:200, 20:80] = 100
        mapped, p, observed, target = data
        def verify(region, ncc):
            return dict(region_ncc=ncc) if ncc >= .72 else None
        bridge, records = interior_bridges(mapped, p, 1, observed, target,
            np.zeros(mapped.shape, bool), 160, 40, verify)
        self.assertTrue(bridge[80:160, 20:80].all())
        self.assertEqual(records[0]['anchors'][1]['target_bbox'], [20, 200, 80, 240])
        # The SAME evidence is not admissible with a shorter extrapolation budget.
        bridge, _ = interior_bridges(mapped, p, 1, observed, target,
            np.zeros(mapped.shape, bool), 100, 40, verify)
        self.assertFalse(bridge[80:160, 20:80].any())

    def test_outward_anchor_search_cannot_jump_over_another_identity(self):
        data = self.bridge_fixture()
        data[1].identities[160:200, 20:80] = 2
        mapped, p, observed, target = data
        bridge, _ = interior_bridges(mapped, p, 1, observed, target,
            np.zeros(mapped.shape, bool), 160, 40, lambda region, ncc: dict(region_ncc=ncc))
        self.assertFalse(bridge[80:160, 20:80].any())

    def test_disconnected_islands_do_not_produce_spurious_bridge_records(self):
        data = self.bridge_fixture()
        data[0][70:80] = 0
        data[0][160:170] = 0
        bridge, records = self.bridge(data)
        self.assertFalse(bridge.any())
        self.assertEqual(records, [])

    def test_duplicate_canonical_pose_reuses_failed_direct_fit(self):
        data = self.fixture()
        self.erase(data[4], 200, 300)
        data[1][200:, :80] = np.random.default_rng(77).integers(0, 255, (100, 80), dtype=np.uint8)
        with patch('periodic_template_matcher._internal.objects.completion._ecc_translation',
                   wraps=_ecc_translation) as ecc:
            self.run_support(data)
        self.assertEqual(ecc.call_count, 1)
        copies = data[4].diagnostics['support_completion']['copies']
        self.assertTrue(any(c.get('canonical_anchor') and not c['regions'] for c in copies))

    def test_bad_raw_evidence_skips_context_score_but_still_attempts_bounded_rescue(self):
        source, target, refs, _, p = object_tests.ObjectCompletionTests().fixture(damage=True)
        before = p.valid.copy()
        with patch('periodic_template_matcher._internal.objects.completion._score',
                   side_effect=AssertionError('Bad raw evidence cannot pass context scoring')), \
             patch('periodic_template_matcher._internal.objects.completion._ecc_translation',
                   side_effect=cv2.error('Failed solve')) as ecc:
            complete_object_support(p, [source], [refs], [np.ones(source.shape, np.uint8)],
                target, np.ones(target.shape, np.uint8), [[0., 100.]],
                MatcherConfig(max_side=300, ordered_object_class=2, shared_periods=True))
        self.assertEqual(ecc.call_count, 1)
        np.testing.assert_array_equal(p.valid, before)

    def test_off_canvas_unobserved_transform_cannot_seed_a_canonical_pose(self):
        data = self.fixture()
        self.erase(data[4], 100, 200)
        data[4].diagnostics['objects'][0]['matrix_template_to_target'][1][2] = 1e8
        self.run_support(data)
        self.assertEqual(data[4].diagnostics['support_completion']['canonical_anchors'], [])

    def test_existing_copy_that_completes_support_skips_canonical_ranking(self):
        data = self.fixture()
        self.erase(data[4], 200, 300)
        with patch('periodic_template_matcher._internal.objects.completion.canonical_support_objects',
                   side_effect=AssertionError('Complete support needs no additional ranking')):
            self.run_support(data)
        np.testing.assert_array_equal(data[4].labels, data[3])

    def test_cross_template_anchor_keeps_its_own_source_and_lists(self):
        source, target, refs, _, p, cfg = self.fixture()
        original = deepcopy(p.diagnostics['objects'][0])
        obj = p.diagnostics['objects'][0]
        obj['reference_partial'] = True
        obj['supporting_references'] = [dict(template_index=1, reference_index=0,
            matrix_template_to_target=np.eye(3).tolist())]
        anchors = canonical_support_objects(p, [source, source], [refs, refs],
            [np.ones(source.shape, np.uint8)]*2, [np.array([0., 100.])], target,
            target.shape[::-1], cfg, target_valid=np.ones(target.shape, np.uint8))
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]['template_index'], 1)
        self.assertFalse(anchors[0]['reference_partial'])
        self.assertNotIn('supporting_transforms', anchors[0])
        self.assertEqual(obj['supporting_transforms'], original['supporting_transforms'])

    def test_spatial_diagnostics_follow_target_crop_and_transpose(self):
        info = dict(template_index=1, matrix_template_to_target=np.eye(3).tolist(),
            interior_bridges=[dict(target_bbox=[10, 20, 30, 40],
                anchors=[dict(target_bbox=[10, 5, 30, 20])])])
        swapped = transpose_diagnostics(info)
        self.assertEqual(swapped['interior_bridges'][0]['target_bbox'], [20, 10, 40, 30])
        self.assertEqual(transpose_diagnostics(swapped), info)
        restored = rebase_diagnostics(info, (50, 200), [(0, 0), (10, 20)], 0)
        self.assertEqual(restored['interior_bridges'][0]['target_bbox'], [60, 220, 80, 240])
        self.assertEqual(restored['matrix_template_to_target'][0][2], 40.)
        self.assertEqual(restored['matrix_template_to_target'][1][2], 180.)

    def test_periodic_affine_maps_exact_lattice_and_preserves_observed_pivot(self):
        H = np.array([[1., .01, 4.], [0., 1.002, 20.], [0., 0., 1.]])
        pivot = np.array([40., 100.])
        corners = np.array([[26., 0.], [54., 0.], [26., 300.], [54., 300.]])
        fitted = periodic_affine(H, [0., 100.], [0., 100.], pivot, corners, np.eye(3))
        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(fitted[:2, :2] @ [0., 100.], [0., 100.])
        np.testing.assert_allclose(fitted @ [*pivot, 1.], H @ [*pivot, 1.])
        self.assertIsNone(periodic_affine(H, [0., 100.], [30., 100.], pivot, corners, np.eye(3)))
        self.assertIsNone(periodic_affine(np.eye(3), [0., 100.], [100., 0.],
            np.array([0., 0.]), np.array([[0., 0.], [1., 1.]]), np.eye(3)))

    def test_a_lattice_from_another_source_cannot_constrain_an_alias(self):
        source, target, refs, _, p, cfg = self.fixture()
        obj = p.diagnostics['objects'][0]
        obj['reference_partial'] = True
        obj['supporting_references'] = [dict(template_index=1, reference_index=0,
            matrix_template_to_target=np.eye(3).tolist())]
        p.diagnostics['periodic_lattice_template_index'] = 0
        with patch('periodic_template_matcher._internal.objects.support_anchors.periodic_affine',
                   side_effect=AssertionError('The source lattice belongs to template zero')):
            anchors = canonical_support_objects(p, [source, source], [refs, refs],
                [np.ones(source.shape, np.uint8)]*2, [np.array([0., 100.])], target,
                target.shape[::-1], cfg, target_valid=np.ones(target.shape, np.uint8))
        self.assertEqual(len(anchors), 1)
        self.assertFalse(anchors[0]['support_anchor']['lattice_constrained'])

    def test_downsampled_ranking_ignores_pixels_outside_roi(self):
        source, target, refs, _, p, cfg = self.fixture()
        up = lambda a: np.repeat(np.repeat(a, 5, axis=0), 5, axis=1)
        source, target = up(source), up(target)
        refs = [dict(r, mask=up(r['mask'])) for r in refs]
        p = ObjectPrediction(up(p.labels), up(p.identities), up(p.valid), deepcopy(p.diagnostics))
        p.diagnostics['objects'][0]['matrix_template_to_target'][1][2] *= 5
        p.diagnostics['periodic_lattice'][0]['target_vector'] = [0., 500.]
        roi = np.zeros(target.shape, np.uint8)
        roi[:306] = 1
        def anchors():
            return canonical_support_objects(p, [source], [refs], [np.ones(source.shape, np.uint8)],
                [np.array([0., 500.])], target, (240, 300), cfg, target_valid=roi)
        before = anchors()
        self.assertTrue(before)
        target[306:310] = 255-target[306:310]
        self.assertEqual(anchors(), before)

    def test_actual_tn8pr2_640_restores_three_continuous_identity_contours(self):
        root = Path(__file__).resolve().parents[1]
        target = root/'source/SD-1/imgs/TN8PR2/18851_AFE564H02RA_TDAOI510_0_450.392_506.722_O_O_20260514_165028.jpg'
        if not target.is_file():
            self.skipTest('Local SD-1 photographs are not installed')
        config = root/'periodic_template_matcher/config.sd1.labelme.opencv.json'
        spec = json.loads(config.read_text(encoding='utf-8'))
        spec['target'] = str(target)
        for template in spec['templates']:
            for key in ('image', 'labelme'):
                template[key] = str((config.parent/template[key]).resolve())
        spec['options']['step1'].update(max_side=640, scale=None)
        spec['options']['step2']['template_workers'] = 1
        spec['options']['periodic']['shared_periods'] = True
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)/'config.json'
            snapshot.write_text(json.dumps(spec), encoding='utf-8')
            result = PeriodicTemplateMatcher(snapshot).infer()
        self.assertTrue(result.accepted)
        objects = result.registration['ordered_objects']
        self.assertEqual(objects['visible_order'], 'DAB')
        support = objects['support_completion']
        self.assertEqual({c['identity'] for c in support['copies'] if c.get('interior_bridges')}, {'D', 'A'})
        # These are source-contour/identity continuity contracts, not GT IoU.
        for identity in (1, 2, 4):
            mask = (result.object_identities == identity) & (result.mapped_labels == 2)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            with self.subTest(identity=identity):
                self.assertEqual(count, 2)
                self.assertLessEqual(stats[1, cv2.CC_STAT_TOP], 2)
                self.assertGreaterEqual(stats[1, cv2.CC_STAT_HEIGHT], 1888)


if __name__ == '__main__':
    unittest.main()