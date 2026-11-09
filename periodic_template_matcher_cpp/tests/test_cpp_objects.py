"""Differential checks for the standalone native object solver."""
from dataclasses import asdict
import unittest
import numpy as np
from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.objects.matching import match_ordered_objects
from periodic_template_matcher._internal.objects.completion import complete_object_support
from tests.test_ordered_objects import scene
from tests import test_object_completion as completion_fixtures
from periodic_template_matcher_cpp import _native as cpp


class NativeObjectsTests(unittest.TestCase):
    def test_partial_cycle_nominal_pose_multiple_references(self):
        from tests.test_partial_instance_cycle import scene as partial_scene
        a, ar, _ = partial_scene('ABC')
        b, br, _ = partial_scene('DAB')
        for order, offset in [('BCD', 0), ('CDA', -43), ('DAB', 41)]:
            target = partial_scene(order, offset)[0]
            cfg = MatcherConfig(ordered_object_class=2, object_cycle=list('ABCD'),
                                object_visible_count=3, object_min_score=.65,
                                max_side=240, angles=(0.,), scales=(1.,))
            args = ([a, b], [ar, br], [np.ones(a.shape, np.uint8)]*2,
                    target, np.ones(target.shape, np.uint8), [np.eye(3), None])
            expected = match_ordered_objects(*args, cfg)
            actual = cpp.match_ordered_objects(*args, asdict(cfg))
            self.assertEqual(actual['diagnostics']['status'], expected.diagnostics['status'])
            self.assertEqual(actual['diagnostics']['visible_order'], expected.diagnostics['visible_order'])
            for key in ('labels', 'identities', 'valid'):
                np.testing.assert_array_equal(actual[key], getattr(expected, key))

    def test_partial_reference_background_remains_unknown(self):
        source, _, refs, _ = scene('ABC', -43)
        target = scene('ABC')[0]
        valid = np.ones(source.shape, np.uint8)
        cfg = MatcherConfig(ordered_object_class=2, object_min_score=.65,
                            max_side=240, angles=(0.,), scales=(1.,))
        args = ([source], [refs], [valid], target, valid, [np.eye(3)])
        expected = match_ordered_objects(*args, cfg)
        actual = cpp.match_ordered_objects(*args, asdict(cfg))
        self.assertEqual(actual['diagnostics']['status'], expected.diagnostics['status'])
        np.testing.assert_array_equal(actual['valid'], expected.valid)

    def test_cyclic_clipped_and_transposed(self):
        for axis, order, offset in [('x', 'ABC', 0), ('x', 'BCA', -43), ('y', 'CAB', 41)]:
            source, _, refs, _ = scene()
            target = scene(order, offset)[0]
            if axis == 'y':
                source, target = source.T.copy(), target.T.copy()
                refs = [dict(r, mask=r['mask'].T.copy()) for r in refs]
            valid = np.ones(source.shape, np.uint8)
            cfg = MatcherConfig(ordered_object_class=2, object_order_axis=axis,
                                object_min_score=.65, max_side=240, angles=(0.,), scales=(1.,))
            args = ([source], [refs], [valid], target, valid, [np.eye(3)])
            expected = match_ordered_objects(*args, cfg)
            actual = cpp.match_ordered_objects(*args, asdict(cfg))
            with self.subTest(axis=axis, order=order):
                for key in ('labels', 'identities', 'valid'):
                    np.testing.assert_array_equal(actual[key], getattr(expected, key))
                self.assertEqual(actual['diagnostics']['status'], expected.diagnostics['status'])
                self.assertEqual(actual['diagnostics']['visible_order'], expected.diagnostics['visible_order'])

    def test_completion_clean_and_damaged(self):
        for mode in ('clean', 'damaged', 'white', 'shifted'):
            source, target, refs, _, prediction = completion_fixtures.ObjectCompletionTests().fixture(mode == 'damaged')
            if mode in ('white', 'shifted'):
                prediction.labels[:, 26:54] = 0
                prediction.identities[:, 26:54] = 0
                prediction.valid[:, 26:54] = False
            if mode == 'white':
                target[100:200, 26:54] = 255
            elif mode == 'shifted':
                target = np.roll(target, 3, axis=1)
            cfg = MatcherConfig(max_side=300, ordered_object_class=2)
            before = {k: getattr(prediction, k).copy() for k in ('labels', 'identities', 'valid')}
            import copy
            before['diagnostics'] = copy.deepcopy(prediction.diagnostics)
            args = ([source], [refs], [np.ones(source.shape, np.uint8)], target,
                    np.ones(target.shape, np.uint8), [[0., 100.]])
            complete_object_support(prediction, *args, cfg)
            actual = cpp.complete_object_support(before, *args, asdict(cfg))
            for key in ('labels', 'identities', 'valid'):
                np.testing.assert_array_equal(actual[key], getattr(prediction, key))


