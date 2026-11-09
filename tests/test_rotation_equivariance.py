"""Quarter turns exchange identity-layout and same-identity repetition axes."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher, run_config
from periodic_template_matcher._internal.io.annotations import load_labels, load_objects
from periodic_template_matcher._internal.objects.completion import complete_object_support
from periodic_template_matcher._internal.objects.matching import ObjectPrediction, match_ordered_objects
from periodic_template_matcher.utils import read_image, write_image, warp_valid_mask
from tests import test_object_completion
from tests.test_ordered_objects import scene
from tests.test_periodic_completion import selection


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'periodic_template_matcher/config.sd1.labelme.opencv.json'


def rotation(shape, k):
    h, w = shape[:2]
    return (np.array([[0., -1, h-1], [1, 0, 0], [0, 0, 1]]) if k == -1 else
            np.array([[0., 1, 0], [-1, 0, w-1], [0, 0, 1]]))


def turn(a, k):
    return np.ascontiguousarray(np.rot90(a, k))


class RotationEquivarianceTests(unittest.TestCase):
    def test_both_quarter_turns_all_phases_and_clipped_edges(self):
        source, _, refs, _ = scene()
        valid = np.ones(source.shape, np.uint8)
        for k in (-1, 1):
            cycle = list('ABC' if k == -1 else 'CBA')
            cfg = MatcherConfig(ordered_object_class=2, object_order_axis='y', object_cycle=cycle,
                class_periodicity={'2': ['x']}, max_side=240, object_min_score=.65)
            cfg.validate()
            for order in ('ABC', 'BCA', 'CAB'):
                for offset in (-43, 0, 41):
                    with self.subTest(k=k, order=order, offset=offset):
                        target, truth, _, original_ids = scene(order, offset)
                        ids = np.zeros_like(original_ids)
                        for i, name in enumerate('ABC'):
                            ids[original_ids == i+1] = cycle.index(name)+1
                        result = match_ordered_objects([turn(source, k)],
                            [[dict(r, mask=turn(r['mask'], k)) for r in refs]], [turn(valid, k)],
                            turn(target, k), turn(valid, k), [np.eye(3)], cfg)
                        self.assertEqual(result.diagnostics['status'], 'accepted')
                        self.assertEqual(result.diagnostics['order_axis'], 'y')
                        self.assertEqual(result.diagnostics['visible_order'], order if k == -1 else order[::-1])
                        np.testing.assert_array_equal(result.labels, turn(truth, k))
                        np.testing.assert_array_equal(result.identities, turn(ids, k))
                        for obj in result.diagnostics['objects']:
                            self.assertIn('center_y', obj)
                            raw = turn(refs[obj['reference_index']]['mask'], k)
                            mask = warp_valid_mask(raw, np.asarray(obj['matrix_template_to_target']), result.labels.shape)
                            expected = result.identities == cycle.index(obj['identity'])+1
                            np.testing.assert_array_equal(mask, expected)

    def test_rotated_white_defect_support_and_transform_coordinates(self):
        for k in (-1, 1):
            with self.subTest(k=k):
                source, target, refs, truth, original = test_object_completion.ObjectCompletionTests().fixture()
                target[230:260, 26:54] = 255
                R = rotation(source.shape, k)
                info = deepcopy(original.diagnostics)
                for obj in info['objects']:
                    obj['matrix_template_to_target'] = (R @ np.asarray(obj['matrix_template_to_target']) @ np.linalg.inv(R)).tolist()
                    obj['supporting_transforms'] = [obj['matrix_template_to_target']]
                info['order_axis'] = 'y'
                cfg = MatcherConfig(max_side=300, ordered_object_class=2, object_order_axis='y',
                                    class_periodicity={'2': ['x']})
                prediction = ObjectPrediction(turn(original.labels, k), turn(original.identities, k),
                                              turn(original.valid, k), info)
                complete_object_support(prediction, [turn(source, k)],
                    [[dict(r, mask=turn(r['mask'], k)) for r in refs]], [turn(np.ones(source.shape, np.uint8), k)],
                    turn(target, k), turn(np.ones(target.shape, np.uint8), k),
                    [(R[:2, :2] @ np.array([0., 100.])).tolist()], cfg)
                np.testing.assert_array_equal(prediction.labels, turn(truth, k))
                self.assertTrue(prediction.valid.all())
                info = prediction.diagnostics['support_completion']
                self.assertEqual(info['repeat_axes'], ['x'])
                self.assertGreater(info['added_foreground_pixels'], 0)
                for copy in info['copies']:
                    self.assertAlmostEqual(copy['target_vector'][1], 0.)

    def test_nonperiodic_vertical_layout_does_not_copy_support(self):
        source, target, refs, _, original = test_object_completion.ObjectCompletionTests().fixture()
        R = rotation(source.shape, -1)
        info = deepcopy(original.diagnostics)
        info['order_axis'] = 'y'
        for obj in info['objects']:
            obj['matrix_template_to_target'] = (R @ np.asarray(obj['matrix_template_to_target']) @ np.linalg.inv(R)).tolist()
        prediction = ObjectPrediction(turn(original.labels, -1), turn(original.identities, -1),
                                      turn(original.valid, -1), info)
        before = prediction.valid.copy()
        cfg = MatcherConfig(max_side=300, ordered_object_class=2, object_order_axis='y', class_periodicity={'2': []})
        complete_object_support(prediction, [turn(source, -1)],
            [[dict(r, mask=turn(r['mask'], -1)) for r in refs]], [turn(np.ones(source.shape, np.uint8), -1)],
            turn(target, -1), turn(np.ones(target.shape, np.uint8), -1), [[100., 0.]], cfg)
        np.testing.assert_array_equal(prediction.valid, before)

    def test_vertical_workflow_rotates_shapes_roi_and_order(self):
        source, _, _, _ = scene()
        target, truth, _, ids = scene('CAB')
        for k in (-1, 1):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                R = rotation(source.shape, k)
                shapes = []
                for i in (2, 0, 1):
                    points = np.array([[26+80*i, 0, 1], [53+80*i, 99, 1]])
                    shapes.append(dict(label='2', shape_type='rectangle', points=(points @ R.T)[:, :2].tolist()))
                image, output = turn(source, k), turn(target, k)
                roi = np.ones(source.shape, np.uint8)
                roi[-4:] = 0
                write_image(root/'source.png', image)
                write_image(root/'target.png', output)
                write_image(root/'roi.png', turn(roi, k))
                (root/'labels.json').write_text(json.dumps(dict(imageHeight=image.shape[0], imageWidth=image.shape[1], shapes=shapes)), encoding='utf-8')
                cycle = list('ABC' if k == -1 else 'CBA')
                spec = dict(target='target.png', target_valid_mask='roi.png', templates=[dict(
                    image='source.png', labelme='labels.json', valid_mask='roi.png', class_map={'2': 2},
                    instance_layouts={'2': {'order_axis': 'y', 'order': cycle}})],
                    options=dict(classes={'2': dict(repeat_axes=['x'], instance_matching=dict(cycle=cycle))},
                                 step1=dict(max_side=240)))
                path = root/'config.json'
                path.write_text(json.dumps(spec), encoding='utf-8')
                with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, np.eye(3))):
                    result = run_config(path, root/'out')
                self.assertEqual(result['ordered_objects']['visible_order'], 'CAB' if k == -1 else 'BAC')
                self.assertEqual(result['config']['object_order_axis'], 'y')
                actual = read_image(root/'out/mapped_labels.png', -1)
                expected = turn(truth, k)
                expected[turn(roi, k) == 0] = 0
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(read_image(root/'out/label_valid.png', -1)[turn(roi, k) == 0].any())

    def test_axis_and_repetition_must_not_conflict(self):
        with self.assertRaisesRegex(ValueError, 'vertical'):
            MatcherConfig(ordered_object_class=2, object_order_axis='y', class_periodicity={'2': ['y']}).validate()
        with self.assertRaises(ValueError):
            MatcherConfig(object_order_axis='z').validate()

    def test_templates_must_agree_on_layout_axis(self):
        from periodic_template_matcher._internal.io.annotations import layout_axis
        specs = [dict(instance_layouts={'2': dict(order_axis=axis, order=list('ABC'))}) for axis in ('x', 'y')]
        with self.assertRaisesRegex(ValueError, 'same.*order_axis'):
            layout_axis(specs)

    def test_transpose_diagnostics_roundtrip_preserves_all_coordinates(self):
        from periodic_template_matcher._internal.objects.axes import transpose_diagnostics
        info = dict(order_axis='y', objects=[dict(center_y=53., horizontal_fraction=.7, vertical_fraction=.9,
            matrix_template_to_target=[[1., .01, 30.], [-.02, .99, -7.], [0., 0., 1.]],
            supporting_transforms=[np.eye(3).tolist()])], layout_spacings=[dict(top='A', bottom='B', target_px=80.)],
            support_completion=dict(repeat_axes=['x'], copies=[dict(target_vector=[80., 1.],
            regions=[dict(corrected_matrix=np.eye(3).tolist()), dict(corrected_matrix=None)])]))
        self.assertEqual(transpose_diagnostics(transpose_diagnostics(info)), info)

    def test_actual_sd1_templates_and_input_quarter_turns(self):
        if not CONFIG.exists():
            self.skipTest('SD-1 config not available')
        spec = json.loads(CONFIG.read_text(encoding='utf-8'))
        resolve = lambda value: (CONFIG.parent/value).resolve()
        if not all(resolve(s['image']).exists() and resolve(s['labelme']).exists() for s in spec['templates']):
            self.skipTest('Optional SD-1 source assets not available')
        cfg = replace(MatcherConfig.from_options(spec['options']), valid_height=None)
        if cfg.ordered_object_class != 2 or any(s['instance_layouts']['2']['order_axis'] != 'x' for s in spec['templates']):
            self.skipTest('This fixture expects the SD-1 horizontal baseline configuration')
        images = [read_image(resolve(s['image'])) for s in spec['templates']]
        target = read_image(resolve(spec['target']))
        labels = [load_labels(s, resolve, im.shape[:2]) for s, im in zip(spec['templates'], images)]
        refs = [load_objects(s, resolve, im.shape[:2], 2, cfg.object_cycle) for s, im in zip(spec['templates'], images)]
        valids = []
        for item, image in zip(spec['templates'], images):
            valid = np.zeros(image.shape[:2], np.uint8)
            valid[:item['valid_height']] = 1
            valids.append(valid)
        tv = np.zeros(target.shape[:2], np.uint8)
        tv[:spec['target_valid_height']] = 1
        baseline = PeriodicTemplateMatcher(cfg).match(images, labels, target, template_valids=valids,
                                                    target_valid=tv, template_objects=refs)
        self.assertTrue(baseline.accepted)
        if (Path(spec['target']).name == '18851_AFE55CS025A_TDAOI810_71_78.715_236.172_O_M_20260124_135701.jpg'
                and list(cfg.object_cycle) == list('ABCD') and cfg.object_visible_count == 3
                and [s['instance_layouts']['2']['order'] for s in spec['templates']]
                    == [list('ABC'), list('DAB')]):
            # User-confirmed identity regression: D is only a left-edge fragment.
            info = baseline.registration['ordered_objects']
            self.assertEqual(info['visible_order'], 'DAB')
            self.assertEqual(info['status'], 'accepted')
            self.assertLess(info['objects'][0]['center_x'], 0)
            self.assertTrue((baseline.object_identities[:, :100] == 4).any())
        for k, name in ((-1, 'cw'), (1, 'ccw')):
            with self.subTest(rotation=name):
                cycle = list(cfg.object_cycle) if k == -1 else list(reversed(cfg.object_cycle))
                rotated_cfg = replace(cfg, object_order_axis='y', object_cycle=cycle,
                    class_periodicity={c: ['y' if a == 'x' else 'x' for a in axes]
                                       for c, axes in cfg.class_periodicity.items()})
                prediction = PeriodicTemplateMatcher(rotated_cfg).match(
                    [turn(im, k) for im in images], [turn(l, k) for l in labels], turn(target, k),
                    template_valids=[turn(v, k) for v in valids], target_valid=turn(tv, k),
                    template_objects=[[dict(r, mask=turn(r['mask'], k)) for r in rr] for rr in refs])
                self.assertTrue(prediction.accepted)
                info = prediction.registration['ordered_objects']
                self.assertEqual(info['status'], 'accepted')
                self.assertEqual(info['order_axis'], 'y')
                old_order = baseline.registration['ordered_objects']['visible_order']
                self.assertEqual(info['visible_order'], old_order if k == -1 else old_order[::-1])
                roi = turn(tv, k) > 0
                self.assertTrue(prediction.label_valid[roi].all())
                self.assertFalse(prediction.label_valid[~roi].any())
                expected = turn(baseline.mapped_labels, k)
                # This is consistency against the baseline, NOT GT accuracy.
                # Full-resolution ECC and discrete peak/reference selection are
                # not exactly rotation-equivariant; exact masks are tested above.
                for class_id in (1, 2):
                    actual_class = prediction.mapped_labels == class_id
                    expected_class = expected == class_id
                    iou = (actual_class & expected_class).sum()/max(int((actual_class | expected_class).sum()), 1)
                    self.assertGreater(iou, .90, (name, class_id, iou))
                identities = np.zeros_like(baseline.object_identities)
                for i, identity in enumerate(cfg.object_cycle):
                    identities[baseline.object_identities == i+1] = cycle.index(identity)+1
                identities = turn(identities, k)
                for i in range(1, len(cycle)+1):
                    actual_id = prediction.object_identities == i
                    expected_id = identities == i
                    if not expected_id.any():
                        # A partial window excludes one identity; rotations must
                        # preserve its absence, rather than assign empty IoU=0.
                        self.assertFalse(actual_id.any(), (name, cycle[i-1]))
                        continue
                    iou = (actual_id & expected_id).sum()/max(int((actual_id | expected_id).sum()), 1)
                    self.assertGreater(iou, .85, (name, cycle[i-1], iou))


if __name__ == '__main__':
    unittest.main()
