"""Different objects share class 2 but must retain their individual appearances."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher, run_config
from periodic_template_matcher.utils import read_image, write_image


def scene(order='ABC', offset=0):
    rng = np.random.default_rng(22)
    background = cv2.GaussianBlur(rng.integers(40, 170, (100, 80), dtype=np.uint8), (3, 3), .6)
    patterns = {name: cv2.GaussianBlur(rng.integers(30, 245, (100, 28), dtype=np.uint8),
                                      (3, 3), .5) for name in 'ABC'}
    image = np.tile(background, (1, 3))
    labels = np.zeros(image.shape, np.uint16)
    objects = []
    identity = np.zeros(image.shape, np.uint16)
    for i, name in enumerate(order):
        x = 26+i*80+offset
        lo, hi = max(0, x), min(240, x+28)
        mask = np.zeros(image.shape, np.uint8)
        if hi > lo:
            image[:, lo:hi] = patterns[name][:, lo-x:hi-x]
            labels[:, lo:hi] = 2
            mask[:, lo:hi] = 1
            identity[:, lo:hi] = 'ABC'.index(name)+1
        objects.append(dict(identity=name, mask=mask))
    return image, labels, objects, identity


class OrderedObjectTests(unittest.TestCase):
    def cfg(self, **kwargs):
        options = dict(ordered_object_class=2, object_min_score=.65, object_min_gap=.03,
                       max_side=240, angles=(0.,), scales=(1.,), refine_candidates=2,
                       ecc_iterations=15, object_min_visible=.15)
        options.update(kwargs)
        return MatcherConfig(**options)

    def detect(self, order, offset=0, cfg=None):
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        source, _, refs, _ = scene()
        target, truth, _, ids = scene(order, offset)
        valid = np.ones(source.shape, np.uint8)
        result = match_ordered_objects([source], [refs], [valid], target, valid,
                                       [np.eye(3)], cfg or self.cfg())
        return result, truth, ids

    def test_distinguishes_all_cyclic_orders(self):
        for order in ('ABC', 'BCA', 'CAB'):
            with self.subTest(order=order):
                result, truth, ids = self.detect(order)
                self.assertEqual(result.diagnostics['status'], 'accepted')
                self.assertEqual(result.diagnostics['visible_order'], order)
                np.testing.assert_array_equal(result.identities, ids)
                np.testing.assert_array_equal(result.labels, truth)

    def test_partial_identity_at_either_boundary(self):
        for order in ('ABC', 'BCA', 'CAB'):
            for offset in (-43, 41):
                with self.subTest(order=order, offset=offset):
                    result, truth, ids = self.detect(order, offset)
                    self.assertEqual(result.diagnostics['status'], 'accepted')
                    np.testing.assert_array_equal(result.labels, truth)
                    np.testing.assert_array_equal(result.identities, ids)
                    self.assertTrue(any(o['partial'] for o in result.diagnostics['objects']))

    def test_insufficient_visible_evidence_is_unknown(self):
        result, _, _ = self.detect('ABC', -52)
        self.assertNotEqual(result.diagnostics['status'], 'accepted')
        self.assertFalse(result.valid.all())

    def test_noise_cannot_be_filled_by_order_alone(self):
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        source, _, refs, _ = scene()
        target = np.random.default_rng(17).integers(0, 255, source.shape, dtype=np.uint8)
        valid = np.ones(source.shape, np.uint8)
        result = match_ordered_objects([source], [refs], [valid], target, valid,
                                       [np.eye(3)], self.cfg())
        self.assertNotEqual(result.diagnostics['status'], 'accepted')
        self.assertFalse(result.valid.all())

    def test_identical_appearances_remain_ambiguous_at_scene_threshold(self):
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        source, _, refs, _ = scene()
        for i in (1, 2):
            source[:, 26+i*80:54+i*80] = source[:, 26:54]
        valid = np.ones(source.shape, np.uint8)
        result = match_ordered_objects([source], [refs], [valid], source, valid,
                                       [np.eye(3)], self.cfg(object_min_gap=.02))
        self.assertEqual(result.diagnostics['status'], 'ambiguous_identity')
        self.assertFalse(result.valid.any())

    def test_clipped_source_cannot_claim_unseen_object_extent_is_background(self):
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        source, _, refs, _ = scene('ABC', -43)
        target, truth, _, _ = scene()
        valid = np.ones(source.shape, np.uint8)
        result = match_ordered_objects([source], [refs], [valid], target, valid,
                                       [np.eye(3)], self.cfg())
        missing = (truth > 0) & (result.labels == 0)
        self.assertFalse(result.valid[missing].any())
        self.assertNotEqual(result.diagnostics['status'], 'accepted')
        # A complete second reference can supply the unseen extent.
        full_source, _, full_refs, _ = scene()
        complete = match_ordered_objects([source, full_source], [refs, full_refs], [valid, valid],
                                         target, valid, [np.eye(3), np.eye(3)], self.cfg())
        np.testing.assert_array_equal(complete.labels, truth)
        self.assertEqual(complete.diagnostics['status'], 'accepted')

    def test_ordered_instance_mode_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'semantic'):
            self.cfg(label_mode='instance').validate()

    def test_small_vertical_fragment_does_not_outrank_full_object(self):
        from periodic_template_matcher._internal.objects.matching import match_ordered_objects
        source, _, refs, _ = scene()
        source = np.tile(source, (6, 1))
        refs = [dict(ref, mask=np.tile(ref['mask'], (6, 1))) for ref in refs]
        target = source.copy()
        rng = np.random.default_rng(88)
        for i in range(3):
            x = 26+i*80
            target[:500, x:x+28] = np.clip(target[:500, x:x+28].astype(float)+rng.normal(0, 45, (500, 28)), 0, 255)
        valid = np.ones(source.shape, np.uint8)
        result = match_ordered_objects([source], [refs], [valid], target, valid, [np.eye(3)], self.cfg(max_side=600))
        self.assertEqual(result.diagnostics['status'], 'accepted')
        self.assertTrue(all(obj['visible_fraction'] > .8 for obj in result.diagnostics['objects']))

    def test_original_annotation_instances_and_config_routing(self):
        from periodic_template_matcher._internal.io.annotations import load_objects
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, labels, objects, _ = scene()
            shapes = []
            for i in (2, 0, 1):  # JSON drawing order is not left-to-right order.
                x = 26+i*80
                shapes.append(dict(label='2', shape_type='rectangle', points=[[x, 0], [x+27, 99]]))
            path = root/'labels.json'
            path.write_text(json.dumps(dict(imageHeight=100, imageWidth=240, shapes=shapes)), encoding='utf-8')
            spec = dict(labelme='labels.json', class_map={'2': 2}, object_order=['A', 'B', 'C'])
            refs = load_objects(spec, lambda p: root/p, image.shape, 2, ('A', 'B', 'C'))
            self.assertEqual([r['identity'] for r in refs], list('ABC'))
            for actual, expected in zip(refs, objects):
                np.testing.assert_array_equal(actual['mask'], expected['mask'])
            spec['object_order'] = ['A', 'A', 'C']
            with self.assertRaises(ValueError):
                load_objects(spec, lambda p: root/p, image.shape, 2, ('A', 'B', 'C'))

    def test_public_api_and_saved_identity_artifact(self):
        source, labels, refs, _ = scene()
        target, truth, _, ids = scene('CAB')
        cfg = self.cfg(min_template_gap=0, min_score=0, periodic_completion=True,
                       period_vectors=((80., 0.),))
        from unittest.mock import patch
        selected = dict(accepted=True, best_template_index=0, ranking=[0], status='accepted',
                        matches=[dict(template_index=0, match_score=90., matrix_template_to_target=np.eye(3).tolist())])
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selected):
            result = PeriodicTemplateMatcher(cfg).match([source], [labels], target, template_objects=[refs])
        np.testing.assert_array_equal(result.mapped_labels, truth)
        np.testing.assert_array_equal(result.object_identities, ids)
        with tempfile.TemporaryDirectory() as temp:
            path = PeriodicTemplateMatcher(cfg).save_result(result, temp)
            metadata = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(metadata['ordered_objects']['visible_order'], 'CAB')
            np.testing.assert_array_equal(read_image(Path(temp)/'object_identities.png', -1), ids)
            np.testing.assert_array_equal(read_image(Path(temp)/'base_labels.png', -1), result.base_labels)
            np.testing.assert_array_equal(read_image(Path(temp)/'base_valid.png', -1) > 0, result.base_valid)
            from periodic_template_matcher.app.data import load_result, render_view
            data = load_result(temp)
            np.testing.assert_array_equal(data.object_identities, ids)
            a = render_view(data, (240, 100), 1., (0., 0.), source='objects', object_identity=1, show_unknown=False)
            self.assertTrue(np.any(a[ids == 1] != data.image[ids == 1]))
            np.testing.assert_array_equal(a[ids != 1], data.image[ids != 1])

    def test_identity_evidence_can_resolve_only_a_geometric_template_tie(self):
        from unittest.mock import patch
        source, labels, refs, _ = scene()
        target, truth, _, _ = scene('CAB')
        selected = dict(accepted=False, best_template_index=0, ranking=[0], status='templates_too_close',
                        matches=[dict(template_index=0, match_score=90., geometry_accepted=True,
                                      matrix_template_to_target=np.eye(3).tolist())])
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selected):
            result = PeriodicTemplateMatcher(self.cfg()).match([source], [labels], target, template_objects=[refs])
        self.assertTrue(result.accepted)
        self.assertEqual(result.registration['status'], 'accepted_ordered_objects')
        np.testing.assert_array_equal(result.mapped_labels, truth)
        selected.update(accepted=False, status='low_match_score')
        selected['matches'][0]['geometry_accepted'] = False
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selected):
            result = PeriodicTemplateMatcher(self.cfg()).match([source], [labels], target, template_objects=[refs])
        self.assertFalse(result.accepted)
        self.assertIsNone(result.mapped_labels)

    def test_workflow_object_orders_and_stale_artifact_cleanup(self):
        from unittest.mock import patch
        source, _, _, _ = scene()
        target, truth, _, ids = scene('BCA')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_image(root/'source.png', source)
            write_image(root/'target.png', target)
            shapes = [dict(label='2', shape_type='rectangle', points=[[26+i*80, 0], [53+i*80, 99]]) for i in range(3)]
            (root/'labels.json').write_text(json.dumps(dict(imageHeight=100, imageWidth=240, shapes=shapes)), encoding='utf-8')
            spec = dict(target='target.png', templates=[dict(image='source.png', labelme='labels.json',
                        class_map={'2': 2}, object_order=list('ABC'))],
                        options=dict(objects=dict(ordered_object_class=2, object_min_score=.65),
                                     step1=dict(max_side=240)))
            path = root/'config.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            selected = dict(accepted=True, best_template_index=0, ranking=[0], status='accepted',
                            matches=[dict(template_index=0, match_score=90., geometry_accepted=True,
                                          matrix_template_to_target=np.eye(3).tolist())])
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selected):
                output = run_config(path, root/'out')
            self.assertEqual(output['ordered_objects']['visible_order'], 'BCA')
            np.testing.assert_array_equal(read_image(root/'out/mapped_labels.png', -1), truth)
            np.testing.assert_array_equal(read_image(root/'out/object_identities.png', -1), ids)
            spec['options'].pop('objects')
            path.write_text(json.dumps(spec), encoding='utf-8')
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selected):
                run_config(path, root/'out')
            self.assertFalse((root/'out/object_identities.png').exists())
            self.assertFalse((root/'out/base_labels.png').exists())
            self.assertFalse((root/'out/base_valid.png').exists())


if __name__ == '__main__':
    unittest.main()
