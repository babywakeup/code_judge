"""Saved prediction contracts and independent viewer rendering."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from periodic_template_matcher import MatchResult, PeriodicTemplateMatcher


class SavedResultTests(unittest.TestCase):
    def make_result(self, accepted=True):
        image = np.full((40, 50, 3), 120, np.uint8)
        valid = np.zeros((40, 50), bool)
        valid[5:35, 5:45] = True
        labels = np.zeros((40, 50), np.uint16)
        labels[10:20, 10:20] = 1001
        registration = dict(accepted=accepted, status='accepted' if accepted else 'low_match_score',
                            ranking=[0], best_template_index=0,
                            matches=[dict(match_score=90., matrix_template_to_target=np.eye(3).tolist())])
        return MatchResult(registration, image, image, image, valid,
                           labels if accepted else None, valid if accepted else None)

    def test_save_load_round_trip_preserves_ids_background_and_unknown(self):
        from periodic_template_matcher.app.data import load_result
        matcher = PeriodicTemplateMatcher()
        result = self.make_result()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = matcher.save_result(result, tmp)
            self.assertEqual(manifest, Path(tmp).resolve() / 'result.json')
            saved = load_result(manifest)
            np.testing.assert_array_equal(saved.image, result.input_image)
            np.testing.assert_array_equal(saved.labels, result.mapped_labels)
            np.testing.assert_array_equal(saved.valid, result.label_valid)
            self.assertEqual(saved.labels.dtype, np.uint16)
            self.assertEqual(saved.labels[15, 15], 1001)
            self.assertEqual(saved.labels[6, 6], 0)
            self.assertTrue(saved.valid[6, 6])
            self.assertFalse(saved.valid[0, 0])
            self.assertFalse((Path(tmp) / 'report.html').exists())

    def test_rejected_rerun_removes_old_prediction_but_preserves_other_files(self):
        from periodic_template_matcher.app.data import load_result
        matcher = PeriodicTemplateMatcher()
        with tempfile.TemporaryDirectory() as tmp:
            matcher.save_result(self.make_result(), tmp)
            keep = Path(tmp) / 'notes.txt'
            keep.write_text('keep')
            (Path(tmp) / 'report.html').write_text('old generated report')
            manifest = matcher.save_result(self.make_result(False), tmp)
            saved = load_result(manifest)
            self.assertIsNone(saved.labels)
            self.assertIsNone(saved.valid)
            self.assertEqual(saved.metadata['prediction_status'], 'match_rejected')
            self.assertFalse((Path(tmp) / 'mapped_labels.png').exists())
            self.assertFalse((Path(tmp) / 'report.html').exists())
            self.assertEqual(keep.read_text(), 'keep')

    def test_viewer_filters_classes_and_periodic_regions(self):
        from periodic_template_matcher.app.data import load_result, render_view
        result = self.make_result()
        result.label_periodic_valid = np.zeros((40, 50), bool)
        result.label_periodic_valid[10:20, 15:20] = True
        with tempfile.TemporaryDirectory() as tmp:
            manifest = PeriodicTemplateMatcher().save_result(result, tmp)
            saved = load_result(manifest)
            direct = render_view(saved, (50, 40), 1., (0., 0.), source='direct', show_unknown=False)
            periodic = render_view(saved, (50, 40), 1., (0., 0.), source='periodic', show_unknown=False)
            np.testing.assert_array_equal(direct[15, 17], saved.image[15, 17])
            self.assertFalse(np.array_equal(periodic[15, 17], saved.image[15, 17]))
            np.testing.assert_array_equal(periodic[15, 12], saved.image[15, 12])
            filtered = render_view(saved, (50, 40), 1., (0., 0.), class_id=2, show_unknown=False)
            np.testing.assert_array_equal(filtered, saved.image)

    def test_loader_rejects_mismatched_mask_size(self):
        from periodic_template_matcher.app.data import load_result
        from periodic_template_matcher.utils import write_image
        with tempfile.TemporaryDirectory() as tmp:
            manifest = PeriodicTemplateMatcher().save_result(self.make_result(), tmp)
            write_image(Path(tmp) / 'mapped_labels.png', np.zeros((2, 2), np.uint16))
            with self.assertRaisesRegex(ValueError, 'shape|size|dimensions'):
                load_result(manifest)

    def test_zoomed_display_and_pixel_lookup_use_identical_sampling(self):
        from periodic_template_matcher.app.data import class_color, load_result, render_view, source_pixel
        with tempfile.TemporaryDirectory() as tmp:
            saved = load_result(PeriodicTemplateMatcher().save_result(self.make_result(), tmp))
            view = render_view(saved, (100, 80), 2., (0., 0.), mode='mask')
            x, y = source_pixel(19, 20, 2., (0., 0.))
            self.assertEqual((x, y), (10, 10))
            self.assertEqual(int(saved.labels[y, x]), 1001)
            np.testing.assert_array_equal(view[20, 19], class_color(1001))
            self.assertNotEqual(class_color(1), class_color(257))


if __name__ == '__main__':
    unittest.main()
