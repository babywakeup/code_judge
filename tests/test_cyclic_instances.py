"""Instance cycles preserve base segmentation and use directed layout evidence."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.objects.matching import ObjectPrediction
from tests.test_periodic_completion import fixture, selection


class CyclicInstanceTests(unittest.TestCase):
    def test_five_names_config(self):
        MatcherConfig(ordered_object_class=2, object_cycle=list('abcde')).validate()
        for cycle in ([], ['a', 'a'], ['a', 2], 'abcde'):
            with self.subTest(cycle=cycle), self.assertRaises(ValueError):
                MatcherConfig(ordered_object_class=2, object_cycle=cycle).validate()

    def test_template_order_rejects_noncyclic_permutation(self):
        from periodic_template_matcher._internal.io.annotations import load_objects
        with self.assertRaisesRegex(ValueError, 'cyclic rotation'):
            load_objects(dict(object_order=list('acbde')), lambda p: p, (20, 20), 2, list('abcde'))

    def test_dp_matches_exhaustive_layout_optimum(self):
        from itertools import product
        from periodic_template_matcher._internal.objects.matching import solve_layout, _quality
        rng = np.random.default_rng(31)
        cycle = list('abcd')
        pools = {}
        spacing = {(cycle[i], cycle[(i+1) % 4]): 25.+5*i for i in range(4)}
        for name in cycle:
            pool = []
            for x in rng.choice(np.arange(5, 110, 10), 4, replace=False):
                mask = np.zeros((4, 120), bool)
                mask[1:3, x:x+2] = True
                pool.append(dict(identity=name, center_x=float(x), score=float(rng.uniform(.7, 1)),
                    vertical_fraction=1., horizontal_fraction=1., reference_partial=False, _mask=mask))
            pools[name] = dict(distinct=pool)
        groups = solve_layout(pools, cycle, spacing, .3)
        for phase in range(4):
            order = cycle[phase:]+cycle[:phase]
            scores = []
            for path in product(*(pools[n]['distinct'] for n in order)):
                if any(a['center_x'] >= b['center_x'] or (a['_mask'] & b['_mask']).any()
                       for a, b in zip(path, path[1:])):
                    continue
                cost = sum(.3*min(abs((b['center_x']-a['center_x'])/spacing[a['identity'], b['identity']]-1), 1)/3
                           for a, b in zip(path, path[1:]))
                scores.append(sum(_quality(c) for c in path)/4-cost)
            actual = [s for s, o, _ in groups if o == order]
            np.testing.assert_allclose(actual, [max(scores)] if scores else [])

    def test_single_identity_duplicate_locations_remain_ambiguous(self):
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        from tests.test_ordered_objects import scene
        source, _, refs, _ = scene()
        source = source[:, :80]
        refs = [dict(identity='only', mask=refs[0]['mask'][:, :80])]
        target = np.tile(source, (1, 2))
        cfg = MatcherConfig(ordered_object_class=2, object_cycle=['only'], max_side=160)
        result = match_ordered_objects([source], [refs], [np.ones(source.shape, np.uint8)],
            target, np.ones(target.shape, np.uint8), [np.eye(3)], cfg)
        self.assertEqual(result.diagnostics['status'], 'ambiguous_identity')
        self.assertFalse(result.identities.any())

    def test_spacing_disambiguates_appearance_and_scales_with_n(self):
        from periodic_template_matcher._internal.objects.matching import solve_layout
        cycle = list('abcde')
        gaps = [40., 60., 90., 120., 150.]
        spacing = {(cycle[i], cycle[(i+1) % 5]): gaps[i] for i in range(5)}
        order = list('cdeab')
        xs = [5.]
        for a, b in zip(order, order[1:]):
            xs.append(xs[-1]+spacing[a, b])
        pools = {}
        for name in cycle:
            candidates = []
            for x in xs:
                mask = np.zeros((4, 450), bool)
                mask[1:3, int(x):int(x)+2] = True
                candidates.append(dict(identity=name, center_x=x, score=.9, vertical_fraction=1.,
                                       horizontal_fraction=1., reference_partial=False, _mask=mask))
            pools[name] = dict(distinct=candidates)
        groups = solve_layout(pools, cycle, spacing, .3)
        self.assertEqual(groups[0][1], order)
        self.assertGreater(groups[0][0]-groups[1][0], .02)
        self.assertEqual(len(groups[0][2]), 5)

    def test_base_periodic_foreground_survives_all_object_outcomes(self):
        source, labels, target, _, H = fixture()
        labels[labels != 1] = 0
        cfg = MatcherConfig(max_side=240, periodic_completion=True,
                            period_vectors=((80., 0.), (0., 80.)), periodic_max_steps=1)
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)):
            baseline = PeriodicTemplateMatcher(cfg).match([source], [labels], target)
        self.assertGreater(baseline.label_periodic_valid.sum(), 0)
        refs = [dict(identity=n, mask=np.ones(source.shape, np.uint8)) for n in 'ABC']
        for status in ('accepted', 'ambiguous_identity', 'partial_reference'):
            object_labels = np.zeros_like(labels)
            object_labels[20:30, 20:30] = 2
            # Deliberately conflicting reference support must not erase class 1.
            object_labels[baseline.mapped_labels == 1] = 2
            object_ids = (object_labels > 0).astype(np.uint16)
            prediction = ObjectPrediction(object_labels, object_ids, np.zeros(labels.shape, bool), dict(status=status))
            with self.subTest(status=status), patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)), \
                    patch('periodic_template_matcher.matcher.match_ordered_objects', return_value=prediction):
                result = PeriodicTemplateMatcher(replace(cfg, ordered_object_class=2)).match(
                    [source], [labels], target, template_objects=[refs])
            np.testing.assert_array_equal(result.base_labels, baseline.mapped_labels)
            np.testing.assert_array_equal(result.base_valid, baseline.label_valid)
            np.testing.assert_array_equal(result.mapped_labels == 1, baseline.mapped_labels == 1)
            self.assertTrue(result.label_valid[result.mapped_labels == 1].all())
            self.assertFalse(result.object_identities[result.mapped_labels == 1].any())
            np.testing.assert_array_equal(result.label_periodic_valid & (result.mapped_labels == 1),
                                          baseline.label_periodic_valid & (baseline.mapped_labels == 1))

    def test_all_five_rotations_with_edge_fragments(self):
        import cv2
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        cycle = list('abcde')
        rng = np.random.default_rng(128)
        textures = {n: cv2.GaussianBlur(rng.integers(20, 250, (100, 28), dtype=np.uint8), (3, 3), .5) for n in cycle}
        def build(order, offset=0):
            image = np.zeros((100, 400), np.uint8)+80
            ids = np.zeros(image.shape, np.uint16)
            refs = []
            for i, n in enumerate(order):
                x = 26+80*i+offset
                lo, hi = max(0, x), min(400, x+28)
                mask = np.zeros(image.shape, np.uint8)
                if hi > lo:
                    image[:, lo:hi] = textures[n][:, lo-x:hi-x]
                    ids[:, lo:hi] = cycle.index(n)+1
                    mask[:, lo:hi] = 1
                refs.append(dict(identity=n, mask=mask))
            return image, refs, ids
        source, refs, _ = build(cycle)
        valid = np.ones(source.shape, np.uint8)
        cfg = MatcherConfig(ordered_object_class=2, object_cycle=cycle, max_side=400, object_min_score=.65)
        for phase in range(5):
            order = cycle[phase:]+cycle[:phase]
            for offset in (-43, 0, 41):
                target, _, truth = build(order, offset)
                with self.subTest(order=order, offset=offset):
                    result = match_ordered_objects([source], [refs], [valid], target, valid, [np.eye(3)], cfg)
                    self.assertEqual(result.diagnostics['status'], 'accepted')
                    np.testing.assert_array_equal(result.identities, truth)

    def test_five_labelme_instances_through_workflow_and_viewer(self):
        import cv2
        import json
        import tempfile
        from pathlib import Path
        from periodic_template_matcher import run_config
        from periodic_template_matcher.app.data import load_result, render_view
        from periodic_template_matcher.utils import write_image
        rng = np.random.default_rng(111)
        source = np.tile(rng.integers(40, 150, (100, 80), dtype=np.uint8), (1, 5))
        ids = np.zeros(source.shape, np.uint16)
        shapes = []
        for i in range(5):
            x = 26+80*i
            source[:, x:x+28] = rng.integers(10, 245, (100, 28), dtype=np.uint8)
            ids[:, x:x+28] = i+1
            shapes.extend([dict(label='2', shape_type='rectangle', points=[[x, 0], [x+27, 99]]),
                           dict(label='1', shape_type='rectangle', points=[[80*i+5, 20], [80*i+15, 40]])])
        source = cv2.GaussianBlur(source, (3, 3), .5)
        target = np.roll(source, -80, axis=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_image(root/'source.png', source)
            write_image(root/'target.png', target)
            (root/'labels.json').write_text(json.dumps(dict(imageHeight=100, imageWidth=400, shapes=shapes[::-1])), encoding='utf-8')
            spec = dict(target='target.png', templates=[dict(image='source.png', labelme='labels.json',
                        class_map={'1': 1, '2': 2}, object_order=list('abcde'))],
                        options=dict(objects=dict(ordered_object_class=2, object_cycle=list('abcde')),
                                     step1=dict(max_side=400)))
            (root/'config.json').write_text(json.dumps(spec), encoding='utf-8')
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, np.eye(3))):
                result = run_config(root/'config.json', root/'out')
            self.assertEqual(result['ordered_objects']['visible_identities'], list('bcdea'))
            loaded = load_result(root/'out')
            np.testing.assert_array_equal(loaded.object_identities, np.roll(ids, -80, axis=1))
            self.assertEqual(loaded.metadata['ordered_objects']['identity_names']['5'], 'e')
            self.assertIn(1, loaded.class_ids)
            self.assertIn(2, loaded.class_ids)
            rendered = render_view(loaded, (400, 100), 1., (0., 0.), source='objects', object_identity=5, show_unknown=False)
            self.assertTrue(np.any(rendered[loaded.object_identities == 5] != loaded.image[loaded.object_identities == 5]))


if __name__ == '__main__':
    unittest.main()
