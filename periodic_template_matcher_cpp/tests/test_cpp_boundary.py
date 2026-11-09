"""Validation and ownership checks at the native Python boundary."""
import unittest
import subprocess
import sys
import textwrap
import numpy as np
from periodic_template_matcher_cpp import _native as native


class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)

    def prepare(self, labels=None, **kw):
        return native.prepare_inputs([self.image], [labels], self.image, {}, **kw)

    def test_labels_noncontiguous_bool_and_owned_output(self):
        labels = np.zeros((64, 128), np.uint16)[:, ::2]
        labels[10:20, 10:20] = 255
        prepared = self.prepare(labels)
        self.assertEqual(prepared['labels'][0].dtype, np.uint16)
        np.testing.assert_array_equal(prepared['labels'][0], labels)
        labels[:] = 0
        self.assertEqual(int(prepared['labels'][0].max()), 255)
        boolean = self.prepare(self.image > 128)['labels'][0]
        self.assertEqual(boolean.dtype, np.uint16)

    def test_binary_lists_and_overlap_rejection(self):
        a = np.zeros_like(self.image)
        b = np.zeros_like(self.image)
        a[:20] = 255
        b[20:40] = 1
        actual = self.prepare([a, b])['labels'][0]
        self.assertTrue((actual[:20] == 1).all())
        self.assertTrue((actual[20:40] == 2).all())
        with self.assertRaises(ValueError):
            self.prepare([a, a])
        self.assertFalse(self.prepare([])['labels'][0].any())

    def test_region_and_annotation_are_separate(self):
        annotation = np.zeros_like(self.image)
        annotation[:8] = 1
        prepared = self.prepare(annotation_valids=[annotation], template_valid_heights=[32])
        self.assertEqual(int(prepared['valids'][0].sum()), 32 * 64)
        self.assertEqual(int(prepared['annotations'][0].sum()), 8 * 64)
        self.assertEqual(prepared['regions']['templates'][0]['valid_height'], 32)
        self.assertEqual(prepared['regions']['target']['height_source'], 'full_image')

    def test_invalid_inputs(self):
        for labels in (np.ones((64, 64), np.float32), np.full((64, 64), -1), np.full((64, 64), 65536)):
            with self.subTest(dtype=labels.dtype), self.assertRaises(ValueError):
                self.prepare(labels)
        for im in (self.image.astype(float), self.image[:31], np.zeros((64, 64, 2), np.uint8)):
            with self.assertRaises(ValueError):
                native.prepare_inputs([im], [None], self.image, {})
        with self.assertRaises(ValueError):
            self.prepare(template_valid_heights=[True])

    def test_unconvertible_valid_mask_raises_without_crashing(self):
        # An unsuccessful NumPy forcecast must never be dereferenced by C++.
        script = textwrap.dedent('''
            import numpy as np
            from periodic_template_matcher_cpp import _native as native
            image = np.ones((64, 64), np.uint8)
            try:
                native.prepare_inputs([image], [None], image, {},
                    template_valids=[np.full(image.shape, object(), dtype=object)])
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError('invalid numeric conversion was accepted')
        ''')
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_object_optional_matrices_and_mismatched_source_valids(self):
        script = textwrap.dedent('''
            import numpy as np
            from periodic_template_matcher_cpp import _native as native
            image = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
            valid = np.ones(image.shape, np.uint8)
            cfg = dict(ordered_object_class=2, object_order_axis='y')
            actual = native.match_ordered_objects([image], [[]], [valid], image, valid, None, cfg)
            assert actual['diagnostics']['status'] == 'insufficient_evidence'
            for source_valids, matrices in (([], [None]), ([valid], [])):
                try:
                    native.match_ordered_objects([image], [[]], source_valids,
                                                 image, valid, matrices, cfg)
                except ValueError:
                    pass
                else:
                    raise AssertionError('mismatched vector length was accepted')
            try:
                native.match_ordered_objects([image], None, [valid], image, valid, [None], cfg)
            except ValueError:
                pass
            else:
                raise AssertionError('missing references were accepted')
        ''')
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_invalid_percentile_is_a_python_exception(self):
        script = textwrap.dedent('''
            from periodic_template_matcher_cpp import _native as native
            for p in (-1.0, 101.0, float('nan'), float('inf')):
                try:
                    native.percentile([1.0, 2.0, 3.0], p)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid percentile was accepted')
        ''')
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_direct_feature_helpers_reject_mismatched_shapes(self):
        # periods() walks the feature dimensions while indexing the mask.
        # Reject before entering that loop in release builds too.
        with self.assertRaises(ValueError):
            native.periods(np.ones((64, 64), np.float32), np.ones((1, 1), np.uint8))
        valid_features = native.features(self.image, np.ones(self.image.shape, np.uint8))
        malformed = (*valid_features[:3], np.ones((1, 1), np.uint8))
        with self.assertRaises(ValueError):
            native.score(malformed, valid_features, np.eye(3), {})

    def test_invalid_config_errors_are_value_errors(self):
        for config in ([1], {'period_vectors': [['oops', 0]]}, {'min_score': float('nan')}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                native.validate_config(config)


if __name__ == '__main__':
    unittest.main()
