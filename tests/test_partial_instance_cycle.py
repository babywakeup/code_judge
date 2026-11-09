"""Partial cyclic windows use complementary templates and retain clipped identities."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.io.annotations import load_objects
from periodic_template_matcher._internal.objects.matching import match_ordered_objects


def scene(order, offset=0):
    rng = np.random.default_rng(723)
    patterns = {name: cv2.GaussianBlur(rng.integers(20, 250, (100, 28), dtype=np.uint8), (3, 3), .5)
                for name in 'ABCD'}
    image = np.full((100, 240), 80, np.uint8)
    ids = np.zeros(image.shape, np.uint16)
    refs = []
    for i, name in enumerate(order):
        x = 26 + 80*i + offset
        lo, hi = max(0, x), min(240, x+28)
        mask = np.zeros(image.shape, np.uint8)
        if hi > lo:
            image[:, lo:hi] = patterns[name][:, lo-x:hi-x]
            ids[:, lo:hi] = 'ABCD'.index(name)+1
            mask[:, lo:hi] = 1
        refs.append(dict(identity=name, mask=mask))
    return image, refs, ids


class PartialCycleTests(unittest.TestCase):
    def test_vertical_repeats_do_not_exhaust_candidates_before_clipped_edge(self):
        from periodic_template_matcher._internal.objects.appearance import _correlation_peaks
        rng = np.random.default_rng(915)
        patch_image = rng.uniform(.1, .9, (80, 30)).astype(np.float32)
        target = np.full((260, 200), .5, np.float32)
        for x in (40, 120):
            for y in (0, 90, 180):
                target[y:y+80, x:x+30] = patch_image
        target[90:170, :6] = patch_image[:, -6:] + rng.normal(0, .06, (80, 6))
        cfg = replace(self.cfg(), object_candidates=4, object_min_visible=.1)
        peaks = _correlation_peaks(patch_image, np.ones_like(patch_image), target,
                                   np.ones_like(target), cfg)
        self.assertTrue(any(x < 0 and 75 < y < 105 for _, x, y in peaks), peaks)

    def cfg(self, **options):
        return MatcherConfig(ordered_object_class=2, object_cycle=list('ABCD'),
                             object_visible_count=3, max_side=240, angles=(0.,), scales=(1.,),
                             object_min_score=.65, object_min_gap=.03, **options)

    def test_visible_count_schema_and_validation(self):
        cfg = MatcherConfig.from_options({'classes': {'2': {'repeat_axes': ['y'],
            'instance_matching': {'cycle': list('ABCD'), 'visible_count': 3}}}})
        self.assertEqual(cfg.object_visible_count, 3)
        for count in (0, -1, 5, True, 2.5, '3'):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'visible_count'):
                replace(cfg, object_visible_count=count).validate()

    def test_all_four_windows_and_both_clipped_boundaries(self):
        # Neither source alone contains all four identities.
        a, ar, _ = scene('ABC'); b, br, _ = scene('DAB')
        for order in ('ABC', 'BCD', 'CDA', 'DAB'):
            for offset in (-43, 0, 41):
                target, _, truth = scene(order, offset)
                with self.subTest(order=order, offset=offset):
                    result = match_ordered_objects([a, b], [ar, br], [np.ones(a.shape, np.uint8)]*2,
                        target, np.ones(target.shape, np.uint8), [np.eye(3)]*2, self.cfg())
                    self.assertEqual(result.diagnostics['status'], 'accepted')
                    self.assertEqual(result.diagnostics['visible_order'], order)
                    self.assertEqual(len(result.diagnostics['objects']), 3)
                    np.testing.assert_array_equal(result.identities, truth)
                    if order in ('BCD', 'CDA'):
                        self.assertEqual({o['template_index'] for o in result.diagnostics['objects']}, {0, 1})

    def test_reference_without_global_geometry_still_contributes_local_evidence(self):
        a, ar, _ = scene('ABC'); b, br, _ = scene('DAB')
        target, _, truth = scene('CDA', -43)
        result = match_ordered_objects([a, b], [ar, br], [np.ones(a.shape, np.uint8)]*2,
            target, np.ones(target.shape, np.uint8), [np.eye(3), None], self.cfg())
        self.assertEqual(result.diagnostics['visible_order'], 'CDA')
        np.testing.assert_array_equal(result.identities, truth)

    def test_noise_is_not_completed_by_cycle_order(self):
        source, refs, _ = scene('ABC')
        target = np.random.default_rng(5).integers(0, 255, source.shape, dtype=np.uint8)
        result = match_ordered_objects([source], [refs], [np.ones(source.shape, np.uint8)],
            target, np.ones(target.shape, np.uint8), [None], self.cfg())
        self.assertNotEqual(result.diagnostics['status'], 'accepted')
        self.assertFalse(result.identities.any())

    def test_identical_appearances_do_not_force_a_phase(self):
        a, ar, _ = scene('ABC'); b, br, _ = scene('DAB')
        for image in (a, b):
            for x in (106, 186):
                image[:, x:x+28] = a[:, 26:54]
        b[:, 26:54] = a[:, 26:54]
        result = match_ordered_objects([a, b], [ar, br], [np.ones(a.shape, np.uint8)]*2,
            a, np.ones(a.shape, np.uint8), [np.eye(3)]*2, self.cfg())
        self.assertEqual(result.diagnostics['status'], 'ambiguous_identity')
        self.assertFalse(result.identities.any())

    def test_array_api_rejects_noncontiguous_and_duplicate_reference_identities(self):
        source, refs, _ = scene('ABC')
        matcher = PeriodicTemplateMatcher(self.cfg())
        for order in ('ACD', 'AAB', 'ABX'):
            invalid = [dict(ref, identity=name) for ref, name in zip(refs, order)]
            with self.subTest(order=order), self.assertRaises(ValueError):
                matcher.match([source], [None], source, template_objects=[invalid])

    def test_partial_cycle_transposes_without_relabeling(self):
        a, ar, _ = scene('ABC'); b, br, _ = scene('DAB')
        target, _, truth = scene('CDA', 41)
        result = match_ordered_objects([a.T.copy(), b.T.copy()],
            [[dict(r, mask=r['mask'].T.copy()) for r in refs] for refs in (ar, br)],
            [np.ones(a.T.shape, np.uint8)]*2, target.T.copy(), np.ones(target.T.shape, np.uint8),
            [None]*2, self.cfg(object_order_axis='y'))
        self.assertEqual(result.diagnostics['visible_order'], 'CDA')
        np.testing.assert_array_equal(result.identities, truth.T)

    def test_partial_labelme_order_and_noncontiguous_rejection(self):
        image, _, _ = scene('DAB')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shapes = [dict(label='2', shape_type='rectangle', points=[[26+80*i, 0], [53+80*i, 99]]) for i in range(3)]
            (root/'labels.json').write_text(json.dumps(dict(imageHeight=100, imageWidth=240, shapes=shapes)), encoding='utf-8')
            spec = dict(labelme='labels.json', class_map={'2': 2}, instance_layouts={'2': dict(order_axis='x', order=list('DAB'))})
            refs = load_objects(spec, lambda p: root/p, image.shape, 2, list('ABCD'))
            self.assertEqual([r['identity'] for r in refs], list('DAB'))
            spec['instance_layouts']['2']['order'] = list('ACD')
            with self.assertRaisesRegex(ValueError, 'cyclic'):
                load_objects(spec, lambda p: root/p, image.shape, 2, list('ABCD'))

    def test_local_only_prediction_survives_base_rejection_and_export(self):
        a, ar, _ = scene('ABC'); b, br, _ = scene('DAB')
        target, _, truth = scene('BCD')
        rejected = dict(best_template_index=None, accepted=False, status='low_match_score', ranking=[0, 1],
                        matches=[dict(template_index=i, geometry_accepted=False, match_score=0.,
                                      matrix_template_to_target=None) for i in range(2)])
        matcher = PeriodicTemplateMatcher(self.cfg())
        with patch('periodic_template_matcher.matcher.select_templates', return_value=rejected):
            result = matcher.match([a, b], [None, None], target, template_objects=[ar, br])
        self.assertTrue(result.accepted)
        self.assertEqual(result.registration['status'], 'accepted_objects_only')
        self.assertFalse(result.registration['base_registration_accepted'])
        np.testing.assert_array_equal(result.object_identities, truth)
        np.testing.assert_array_equal(result.label_valid, truth > 0)
        self.assertFalse((result.mapped_labels == 1).any())
        with tempfile.TemporaryDirectory() as temp:
            manifest = matcher.save_result(result, temp)
            self.assertTrue(json.loads(manifest.read_text())['labels_exported'])


if __name__ == '__main__':
    unittest.main()
