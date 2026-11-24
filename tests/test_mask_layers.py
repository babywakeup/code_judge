"""Independent stage masks retain overlapping classes without changing fusion."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from periodic_template_matcher import PeriodicTemplateMatcher
from periodic_template_matcher._internal.objects.fusion import PredictionFusionStep
from periodic_template_matcher._internal.objects.steps import LocatedObjects
from periodic_template_matcher._internal.segmentation.transfer import LabelTransferStep
from periodic_template_matcher.app.data import class_color, load_result, render_view
from tests import test_match_pipeline as pipeline_tests, test_result_viewer as viewer_tests


class MaskLayerTests(unittest.TestCase):
    def fixture(self):
        case = pipeline_tests.MatchStepTests()
        case.setUp()
        case.config.ordered_object_class = 2
        case.config.object_visible_count = 1
        case.inputs.objects = [[]]
        LabelTransferStep().run(case.inputs, case.match, case.config)
        return case

    def test_direct_layer_excludes_independently_matched_class(self):
        case = self.fixture()
        result = case.match.result
        self.assertTrue((result.direct_labels == 1).any())
        self.assertFalse((result.direct_labels == 2).any())
        self.assertFalse((result.mapped_labels == 2).any())
        np.testing.assert_array_equal(result.direct_labels, result.mapped_labels)
        self.assertFalse(np.shares_memory(result.direct_labels, result.mapped_labels))
        before = result.direct_labels.copy()
        result.mapped_labels[:] = 0
        np.testing.assert_array_equal(result.direct_labels, before)

    def test_direct_exclusion_uses_configured_class_not_hardcoded_label2(self):
        case = self.fixture()
        case.config.ordered_object_class = 1
        before = case.inputs.labels[0].copy()
        transferred = LabelTransferStep().run(case.inputs, case.match, case.config)
        self.assertFalse((case.match.result.direct_labels == 1).any())
        self.assertTrue((case.match.result.direct_labels == 2).any())
        self.assertFalse((transferred.labels == 1).any())
        np.testing.assert_array_equal(case.inputs.labels[0], before)

    def test_semantic_only_matching_keeps_all_direct_classes(self):
        case = self.fixture()
        case.inputs.objects = None
        LabelTransferStep().run(case.inputs, case.match, case.config)
        self.assertTrue((case.match.result.direct_labels == 1).any())
        self.assertTrue((case.match.result.direct_labels == 2).any())
        np.testing.assert_array_equal(case.match.result.direct_labels, case.match.result.mapped_labels)

    def test_all_sources_do_not_recall_unlocated_label2_from_template(self):
        case = self.fixture()
        PredictionFusionStep().run(case.inputs, case.match,
            LocatedObjects(case.prediction(), case.inputs.template_valids), case.config)
        with tempfile.TemporaryDirectory() as tmp:
            data = load_result(PeriodicTemplateMatcher().save_result(case.match.result, tmp))
            for source in ('direct', 'all', 'fused', 'objects'):
                frame = render_view(data, (64, 64), 1., (0., 0.), mode='mask',
                                    source=source, class_id=2, show_unknown=False)
                self.assertFalse(np.all(frame == class_color(2), axis=-1).any(), source)

    def test_verified_object_keeps_both_layers_instead_of_destroying_class1(self):
        case = self.fixture()
        result = case.match.result
        pred = case.prediction()
        pred.labels[4:8, 4:8] = 2
        pred.identities[4:8, 4:8] = 1
        PredictionFusionStep().run(case.inputs, case.match,
            LocatedObjects(pred, case.inputs.template_valids), case.config)
        np.testing.assert_array_equal(result.mapped_labels[4:8, 4:8], 2)
        np.testing.assert_array_equal(result.direct_labels[4:8, 4:8], 1)
        np.testing.assert_array_equal(result.object_labels[4:8, 4:8], 2)
        np.testing.assert_array_equal(result.object_layer_identities[4:8, 4:8], 1)
        pred.labels[:] = 0
        self.assertTrue(result.object_labels[4:8, 4:8].all())

    def test_protected_base_does_not_destroy_independent_object_identity(self):
        case = self.fixture()
        case.config.object_visible_count = None
        pred = case.prediction()
        pred.labels[4:8, 4:8] = 2
        pred.identities[4:8, 4:8] = 1
        PredictionFusionStep().run(case.inputs, case.match,
            LocatedObjects(pred, case.inputs.template_valids), case.config)
        result = case.match.result
        np.testing.assert_array_equal(result.mapped_labels[4:8, 4:8], 1)
        self.assertFalse(result.object_identities[4:8, 4:8].any())
        np.testing.assert_array_equal(result.object_layer_identities[4:8, 4:8], 1)

    def saved_fixture(self):
        result = viewer_tests.SavedResultTests().make_result()
        shape = result.mapped_labels.shape
        result.direct_labels = result.mapped_labels.copy()
        result.direct_valid = result.label_valid.copy()
        result.direct_labels[10:20, 10:20] = 1
        result.completion_labels = np.zeros(shape, np.uint16)
        result.completion_valid = np.zeros(shape, bool)
        result.completion_labels[10:20, 15:25] = 1
        result.completion_valid[10:20, 15:25] = True
        result.object_labels = np.zeros(shape, np.uint16)
        result.object_layer_valid = np.zeros(shape, bool)
        result.object_labels[10:20, 10:25] = 2
        result.object_layer_valid[10:20, 10:25] = True
        result.object_layer_identities = result.object_labels//2
        result.mapped_labels[10:20, 10:25] = 2
        return result

    def test_export_load_and_render_preserve_overlapping_class_membership(self):
        result = self.saved_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            data = load_result(PeriodicTemplateMatcher().save_result(result, tmp))
            for name in ('direct_labels', 'direct_valid', 'completion_labels', 'completion_valid',
                         'object_labels', 'object_layer_valid', 'object_layer_identities'):
                np.testing.assert_array_equal(getattr(data, name), getattr(result, name))
            self.assertIn(1, data.class_ids)  # Absent from the exclusive final map.
            def view(source, class_id=None, identity=None):
                return render_view(data, (50, 40), 1., (0., 0.), mode='mask',
                                   source=source, class_id=class_id, object_identity=identity,
                                   show_unknown=False)
            np.testing.assert_array_equal(view('direct')[15, 17], class_color(1))
            np.testing.assert_array_equal(view('periodic')[15, 17], class_color(1))
            np.testing.assert_array_equal(view('objects')[15, 17], class_color(2))
            np.testing.assert_array_equal(view('fused')[15, 17], class_color(2))
            np.testing.assert_array_equal(view('all', 1)[15, 17], class_color(1))
            np.testing.assert_array_equal(view('all', 2)[15, 17], class_color(2))
            mean = ((np.asarray(class_color(1), float)+class_color(2))/2).astype(np.uint8)
            np.testing.assert_array_equal(view('all')[15, 17], mean)
            np.testing.assert_array_equal(view('objects', identity=1)[15, 17], class_color(2))
            self.assertFalse(np.array_equal(view('objects', identity=2)[15, 17], class_color(2)))

    def test_source_unknown_and_background_are_not_derived_from_final_validity(self):
        result = self.saved_fixture()
        result.label_valid[15, 17] = False
        result.mapped_labels[15, 17] = 0
        with tempfile.TemporaryDirectory() as tmp:
            data = load_result(PeriodicTemplateMatcher().save_result(result, tmp))
            self.assertTrue(data.direct_valid[15, 17])
            view = render_view(data, (50, 40), 1., (0., 0.), mode='mask', source='direct')
            np.testing.assert_array_equal(view[15, 17], class_color(1))

    def test_partial_or_malformed_layer_pair_rejected_before_export(self):
        for reason in ('missing_valid', 'wrong_size', 'wrong_label_dtype', 'wrong_valid_dtype'):
            result = self.saved_fixture()
            if reason == 'missing_valid': result.direct_valid = None
            if reason == 'wrong_size': result.direct_labels = result.direct_labels[:2]
            if reason == 'wrong_label_dtype': result.direct_labels = result.direct_labels.astype(np.uint8)
            if reason == 'wrong_valid_dtype': result.direct_valid = result.direct_valid.astype(np.uint8)
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                PeriodicTemplateMatcher().save_result(result, tmp)

    def test_loader_rejects_missing_layer_pair_and_escaping_layer_path(self):
        for reason in ('pair', 'escape'):
            with tempfile.TemporaryDirectory() as tmp:
                path = PeriodicTemplateMatcher().save_result(self.saved_fixture(), tmp)
                meta = json.loads(path.read_text(encoding='utf-8'))
                if reason == 'pair': del meta['artifacts']['direct_valid']
                else: meta['artifacts']['direct_labels'] = '../outside.png'
                path.write_text(json.dumps(meta), encoding='utf-8')
                with self.assertRaises(ValueError): load_result(path)

    def test_rejected_rerun_cleans_owned_layer_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            matcher = PeriodicTemplateMatcher()
            matcher.save_result(self.saved_fixture(), tmp)
            matcher.save_result(viewer_tests.SavedResultTests().make_result(False), tmp)
            self.assertFalse((Path(tmp)/'direct_labels.png').exists())
            self.assertFalse((Path(tmp)/'object_layer_identities.png').exists())

    def test_pipeline_captures_periodic_mask_before_fusion_and_late_repairs(self):
        from periodic_template_matcher._internal.pipeline import MatchPipeline
        from unittest.mock import Mock
        case = self.fixture()
        pred = case.prediction()
        pred.labels[20:24, 20:24] = 2
        pred.identities[20:24, 20:24] = 1
        located = LocatedObjects(pred, case.inputs.template_valids)
        pipeline = MatchPipeline(case.config, registration=Mock(run=Mock(return_value=case.match)),
                                 ordered_objects=Mock(run=Mock(return_value=located)),
                                 object_support=Mock())
        def periodic(*args):
            result = case.match.result
            result.label_periodic_valid = np.zeros(case.labels.shape, bool)
            result.label_periodic_valid[20:24, 20:24] = True
            result.mapped_labels[20:24, 20:24] = 1
        def late(result, *args, **kwargs):
            result.mapped_labels[30:34, 30:34] = 1
            result.label_periodic_valid[30:34, 30:34] = True
        pipeline.periodic_completion.run = periodic
        with patch('periodic_template_matcher._internal.pipeline.complete_periodic_instances', side_effect=late):
            result = pipeline.run(case.inputs)
        self.assertFalse(result.label_periodic_valid[20:24, 20:24].any())
        self.assertTrue(result.completion_valid[20:24, 20:24].all())
        np.testing.assert_array_equal(result.completion_labels[20:24, 20:24], 1)
        np.testing.assert_array_equal(result.mapped_labels[20:24, 20:24], 2)
        np.testing.assert_array_equal(result.completion_labels[30:34, 30:34], 1)

    def test_crop_restores_every_new_layer_to_original_coordinates(self):
        from periodic_template_matcher._internal.segmentation.crop import restore_crop
        from periodic_template_matcher._internal.segmentation.preparation import InputPreparationStep
        from periodic_template_matcher import MatcherConfig
        image = np.zeros((64, 64, 3), np.uint8)
        labels = np.zeros((64, 64), np.uint16)
        inputs = InputPreparationStep(MatcherConfig(valid_x_start=5, valid_x_end=55,
            valid_y_start=7, valid_y_end=47)).run([image], [labels], image)
        from periodic_template_matcher._internal.models import MatchResult
        result = MatchResult(dict(ranking=[0]), inputs.target, inputs.templates[0],
                             inputs.target.copy(), inputs.target_roi.copy())
        keys = ('direct_labels', 'direct_valid', 'completion_labels', 'completion_valid',
                'object_labels', 'object_layer_valid', 'object_layer_identities')
        for key in keys:
            setattr(result, key, np.ones(inputs.target.shape[:2],
                    bool if 'valid' in key else np.uint16))
        restore_crop(result, inputs)
        for key in keys:
            array = getattr(result, key)
            self.assertEqual(array.shape, (64, 64))
            self.assertTrue(array[7:47, 5:55].all())
            self.assertFalse(array[:7].any())
            self.assertFalse(array[:, :5].any())
            self.assertFalse(array[47:].any())

    def test_two_reported_tnxss2_images_preserve_all_stage_masks(self):
        import gc
        gc.collect()
        root = Path(__file__).resolve().parents[1]
        names = ('18851_AFE55CS01NA_TDAOI810_53_-505.410_-774.357_O_M_20260124_115739.jpg',
                 '18851_AFE55CS028A_TDAOI810_29_378.943_-176.033_O_L_20260124_141033.jpg')
        targets = [root/'source/SD-1/imgs/TNXSS2'/name for name in names]
        if not all(path.is_file() for path in targets):
            self.skipTest('Local SD-1 photographs are not installed')
        config = root/'periodic_template_matcher/config.sd1.labelme.opencv.json'
        spec = json.loads(config.read_text(encoding='utf-8'))
        for template in spec['templates']:
            for key in ('image', 'labelme'):
                template[key] = str((config.parent/template[key]).resolve())
        spec['target'] = str(targets[0])
        spec['options']['step1'].update(max_side=640, scale=None)
        spec['options']['step2']['template_workers'] = 1
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)/'config.json'
            snapshot.write_text(json.dumps(spec), encoding='utf-8')
            matcher = PeriodicTemplateMatcher(snapshot)
            for i, target in enumerate(targets):
                with self.subTest(target=target.name):
                    result = matcher.infer(target=target)
                    self.assertTrue(result.accepted)
                    self.assertFalse((result.direct_labels == 2).any())
                    self.assertFalse((result.completion_labels == 2).any())
                    self.assertTrue((result.object_labels == 2).any())
                    overlap = ((result.direct_labels == 1) & result.direct_valid
                               & (result.object_labels == 2) & result.object_layer_valid)
                    if i == 0:
                        self.assertGreater(int(overlap.sum()), 64)
                        self.assertTrue((result.mapped_labels[overlap] == 2).all())
                    else:
                        # This image does NOT reproduce the same overlap loss;
                        # preserve that counterevidence rather than inventing it.
                        self.assertFalse(overlap.any())
                    saved = load_result(matcher.save_result(result, Path(tmp)/str(i)))
                    np.testing.assert_array_equal(saved.direct_labels, result.direct_labels)
                    np.testing.assert_array_equal(saved.object_labels, result.object_labels)
                    evidence = overlap if overlap.any() else ((result.completion_labels == 1)
                                                             & result.completion_valid)
                    points = np.argwhere(evidence)
                    self.assertGreater(len(points), 64)
                    y, x = points[len(points)//2]
                    frame = render_view(saved, (1, 1), 1., (-float(x), -float(y)),
                                        mode='mask', class_id=1, source='all', show_unknown=False)
                    np.testing.assert_array_equal(frame[0, 0], class_color(1))
                    # Do not keep both full-resolution results and their loaded
                    # PNG copies alive while the next serial inference runs.
                    del result, saved, overlap, evidence, points, frame
                    gc.collect()


if __name__ == '__main__':
    unittest.main()