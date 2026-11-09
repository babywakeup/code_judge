"""Class rules and template layouts explicitly share the same semantic class ID."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from periodic_template_matcher import MatcherConfig, run_config
from periodic_template_matcher._internal.io.annotations import load_objects
from periodic_template_matcher._internal.io.workflow import unique_templates
from periodic_template_matcher.utils import read_image, write_image
from tests.test_ordered_objects import scene
from tests.test_periodic_completion import selection


def class_options():
    return {'classes': {
        '1': {'repeat_axes': ['x', 'y']},
        '2': {'repeat_axes': ['y'], 'instance_matching': {
            'cycle': ['A', 'B', 'C'], 'min_score': .7, 'min_gap': .02,
            'min_visible': .15, 'candidates': 8, 'spacing_weight': .3}}}}


class ClassConfigSchemaTests(unittest.TestCase):
    def test_new_rules_normalize_to_legacy_runtime_config(self):
        new = MatcherConfig.from_options(class_options())
        old = MatcherConfig.from_options({
            'objects': {'ordered_object_class': 2, 'object_cycle': ['A', 'B', 'C'],
                        'object_min_score': .7, 'object_min_gap': .02, 'object_min_visible': .15,
                        'object_candidates': 8, 'object_spacing_weight': .3},
            'periodic': {'class_periodicity': {'1': ['x', 'y'], '2': ['y']}}})
        self.assertEqual(asdict(new), asdict(old))
        self.assertEqual(new.ordered_object_class, 2)
        self.assertEqual(new.periodic_axes(2), ('y',))

    def test_semantic_only_rules_disable_instance_matching(self):
        cfg = MatcherConfig.from_options({'classes': {'1': {'repeat_axes': []}}})
        self.assertIsNone(cfg.ordered_object_class)
        self.assertEqual(cfg.periodic_axes(1), ())

    def test_invalid_or_ambiguous_class_rules_are_rejected(self):
        invalid = [None, [], {}, {'01': {'repeat_axes': []}}, {'0': {'repeat_axes': []}},
                   {'1': {}}, {'1': {'repeat_axes': 'xy'}}, {'1': {'repeat_axes': ['z']}},
                   {'1': {'repeat_axes': [], 'unexpected': True}},
                   {'2': {'repeat_axes': ['y'], 'instance_matching': {}}},
                   {'2': {'repeat_axes': ['y'], 'instance_matching': None}},
                   {'2': {'repeat_axes': ['y'], 'instance_matching': {'cycle': ['A'], 'score': .5}}}]
        for classes in invalid:
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                MatcherConfig.from_options({'classes': classes})
        options = class_options()
        options['classes']['3'] = deepcopy(options['classes']['2'])
        with self.assertRaisesRegex(ValueError, 'one.*instance_matching'):
            MatcherConfig.from_options(options)
        for old in ({'objects': {'ordered_object_class': 2}}, {'object_min_score': .5},
                    {'periodic': {'class_periodicity': {'1': []}}}):
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, 'mix'):
                MatcherConfig.from_options(dict(class_options(), **old))

    def template(self, root, order='CAB'):
        source, _, refs, _ = scene(order)
        shapes = []
        for i in (2, 0, 1):
            x = 26+80*i
            shapes.append(dict(label='2', shape_type='rectangle', points=[[x, 0], [x+27, 99]]))
        (root/'labels.json').write_text(json.dumps(dict(imageHeight=100, imageWidth=240, shapes=shapes)), encoding='utf-8')
        spec = dict(image='source.png', labelme='labels.json', class_map={'2': 2},
                    instance_layouts={'2': {'order_axis': 'x', 'order': list(order)}})
        return source, refs, spec

    def test_layout_is_bound_to_class_and_axis_not_drawing_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, expected, spec = self.template(root)
            actual = load_objects(spec, lambda p: root/p, source.shape, 2, list('ABC'))
            self.assertEqual([r['identity'] for r in actual], list('CAB'))
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a['mask'], b['mask'])

    def test_unsupported_or_mismatched_layout_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, original = self.template(root)
            invalid = [None, {}, {'3': {'order_axis': 'x', 'order': list('ABC')}},
                       {'2': {'order_axis': 'z', 'order': list('ABC')}},
                       {'2': {'order': list('ABC')}},
                       {'2': {'order_axis': 'x', 'order': list('ABC'), 'oops': 1}}]
            for layouts in invalid:
                spec = dict(original, instance_layouts=layouts)
                with self.subTest(layouts=layouts), self.assertRaises(ValueError):
                    load_objects(spec, lambda p: root/p, source.shape, 2, list('ABC'))
            with self.assertRaisesRegex(ValueError, 'mix'):
                load_objects(dict(original, object_order=list('ABC')), lambda p: root/p, source.shape, 2, list('ABC'))

    def test_duplicate_image_different_layout_is_a_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, spec = self.template(root)
            write_image(root/'source.png', source)
            other = deepcopy(spec)
            other['instance_layouts']['2']['order'] = list('ABC')
            with self.assertRaisesRegex(ValueError, 'Duplicate template'):
                unique_templates([spec, other], lambda p: root/p)

    def test_new_workflow_is_pixel_exact_to_legacy_and_exports_readable_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, template = self.template(root, order='ABC')
            target, truth, _, ids = scene('CAB')
            write_image(root/'source.png', source)
            write_image(root/'target.png', target)
            options = class_options()
            options['step1'] = {'max_side': 240}
            spec = dict(target='target.png', templates=[template], options=options)
            path = root/'config.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, np.eye(3))):
                new = run_config(path, root/'new')
            self.assertEqual(new['class_rules'], options['classes'])
            self.assertEqual(new['template_instance_layouts'][0]['2']['order_axis'], 'x')
            self.assertEqual(new['ordered_objects']['class_id'], 2)
            self.assertEqual(new['ordered_objects']['order_axis'], 'x')
            self.assertEqual(new['ordered_objects']['visible_order'], 'CAB')
            np.testing.assert_array_equal(read_image(root/'new/mapped_labels.png', -1), truth)
            np.testing.assert_array_equal(read_image(root/'new/object_identities.png', -1), ids)
            template.pop('instance_layouts')
            template['object_order'] = list('ABC')
            options.pop('classes')
            options['objects'] = dict(ordered_object_class=2, object_min_gap=.02)
            options['periodic'] = dict(class_periodicity={'1': ['x', 'y'], '2': ['y']})
            path.write_text(json.dumps(spec), encoding='utf-8')
            with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, np.eye(3))):
                run_config(path, root/'old')
            for name in ('mapped_labels.png', 'label_valid.png', 'object_identities.png'):
                np.testing.assert_array_equal(read_image(root/'new'/name, -1), read_image(root/'old'/name, -1))

    def test_layout_without_enabled_instance_class_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, template = self.template(root)
            write_image(root/'source.png', source)
            spec = dict(target='source.png', templates=[template], options={'classes': {'2': {'repeat_axes': ['y']}}})
            path = root/'config.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'instance_matching'):
                run_config(path, root/'out')


if __name__ == '__main__':
    unittest.main()