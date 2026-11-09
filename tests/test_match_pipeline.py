"""Step contracts and orchestration, independent of numerical match quality."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.pipeline import MatchPipeline
from periodic_template_matcher._internal.objects.matching import ObjectPrediction
from periodic_template_matcher._internal.models import RegisteredMatch
from periodic_template_matcher._internal.segmentation.evaluation import SegmentationEvaluationStep
from periodic_template_matcher._internal.objects.fusion import PredictionFusionStep
from periodic_template_matcher._internal.objects.steps import (
    LocatedObjects, ObjectSupportCompletionStep, OrderedObjectStep,
)
from periodic_template_matcher._internal.segmentation.preparation import InputPreparationStep
from periodic_template_matcher._internal.registration.step import RegistrationStep
from periodic_template_matcher._internal.segmentation.transfer import LabelTransferStep, PeriodicCompletionStep
from tests.test_matcher_api import selection


class MatchStepTests(unittest.TestCase):
    def setUp(self):
        self.config = MatcherConfig()
        self.image = np.zeros((64, 64, 3), np.uint8)
        self.labels = np.zeros((64, 64), np.uint16)
        self.labels[4:8, 4:8] = 1
        self.labels[10:14, 10:14] = 2
        self.preparation = InputPreparationStep(self.config)
        self.inputs = self.preparation.run([self.image], [self.labels], self.image)
        self.match = RegistrationStep(Mock(return_value=selection())).run(self.inputs, self.config)

    def prediction(self, status='accepted'):
        return ObjectPrediction(np.zeros_like(self.labels), np.zeros_like(self.labels),
                                np.ones_like(self.labels, bool), dict(status=status))

    def test_preparation_does_not_mutate_object_metadata_or_masks(self):
        self.config.ordered_object_class = 2
        self.config.object_cycle = ('A',)
        refs = [[dict(identity='A', mask=(self.labels == 2).astype(np.uint8))]]
        annotation = np.ones_like(self.labels, np.uint8)
        annotation[:2] = 0
        inputs = self.preparation.run([self.image], [self.labels], self.image,
                                      annotation_valids=[annotation], template_objects=refs)
        self.assertNotIn('background_known', refs[0][0])
        self.assertFalse(inputs.objects[0][0]['background_known'])
        self.assertTrue(inputs.template_valids[0][:2].all())
        np.testing.assert_array_equal(annotation[:2], 0)
        np.testing.assert_array_equal(inputs.labels[0], self.labels)

    def test_registration_resets_only_already_resolved_height_policy(self):
        self.config.valid_height = 40
        inputs = self.preparation.run([self.image], [self.labels], self.image)
        selector = Mock(return_value=selection())
        match = RegistrationStep(selector).run(inputs, self.config)
        resolved_config = selector.call_args.args[2]
        self.assertIsNone(resolved_config.valid_height)
        self.assertEqual(self.config.valid_height, 40)
        self.assertFalse(match.result.geometric_valid[40:].any())

    def test_direct_transfer_excludes_instances_without_mutating_sources(self):
        self.config.ordered_object_class = 2
        self.inputs.objects = [[]]
        before = self.labels.copy()
        transferred = LabelTransferStep().run(self.inputs, self.match, self.config)
        self.assertFalse((transferred.labels == 2).any())
        self.assertTrue((self.match.result.mapped_labels == 1).any())
        np.testing.assert_array_equal(self.labels, before)

    def test_periodic_step_is_gated_by_enable_flag_and_semantic_mode(self):
        transferred = LabelTransferStep().run(self.inputs, self.match, self.config)
        with patch('periodic_template_matcher._internal.segmentation.transfer.complete_periodic') as complete:
            for enabled, mode, expected_calls in ((False, 'semantic', 0), (True, 'instance', 0),
                                                  (True, 'semantic', 1)):
                self.config.periodic_completion, self.config.label_mode = enabled, mode
                PeriodicCompletionStep().run(self.inputs, self.match, transferred, self.config)
                self.assertEqual(complete.call_count, expected_calls)

    def test_identity_evidence_only_overrides_template_tie(self):
        self.inputs.objects = [[]]
        for status, accepted in (('templates_too_close', True), ('low_score', False)):
            metadata = selection(False)
            metadata['status'] = status
            match = RegistrationStep(Mock(return_value=metadata)).run(self.inputs, self.config)
            locator = Mock(return_value=self.prediction())
            located = OrderedObjectStep(locator).run(self.inputs, match, self.config)
            self.assertIsNotNone(located)
            self.assertEqual(match.result.accepted, accepted)
            if accepted:
                self.assertEqual(metadata['status'], 'accepted_ordered_objects')
                self.assertTrue(metadata['template_selection_ambiguous'])

    def test_support_uses_geometry_approval_and_original_pixel_vectors(self):
        matrix = np.array([[2., 0., 8.], [0., 3., 9.], [0., 0., 1.]])
        match = RegisteredMatch(self.match.result, 0, matrix)
        periodic = dict(copies=[dict(geometry_accepted=True, accepted=False)],
                        period_vectors=[[0., 4.]])
        match.result.registration['periodic_completion'] = periodic
        located = LocatedObjects(self.prediction(), self.inputs.template_valids)
        complete = Mock()
        step = ObjectSupportCompletionStep(complete)
        step.run(self.inputs, match, located, self.config)
        np.testing.assert_allclose(complete.call_args.args[6], [[0., 12.]])
        periodic['copies'] = [dict(geometry_accepted=False, accepted=True)]
        step.run(self.inputs, match, located, self.config)
        self.assertEqual(complete.call_count, 1)

    def test_support_accepts_legacy_copy_diagnostics(self):
        self.match.result.registration['periodic_completion'] = dict(
            copies=[dict(accepted=True)], period_vectors=[[0., 4.]])
        complete = Mock()
        ObjectSupportCompletionStep(complete).run(
            self.inputs, self.match, LocatedObjects(self.prediction(), self.inputs.template_valids), self.config)
        complete.assert_called_once()

    def test_fusion_preserves_base_foreground_and_records_independent_snapshot(self):
        LabelTransferStep().run(self.inputs, self.match, self.config)
        result = self.match.result
        before = result.mapped_labels.copy()
        prediction = self.prediction()
        prediction.labels[4:8, 4:8] = 2  # Conflict with existing class 1.
        prediction.labels[16:18, 16:18] = 2
        prediction.identities[prediction.labels > 0] = 1
        prediction.valid[:2] = False
        result.registration['ordered_objects'] = prediction.diagnostics
        result.label_periodic_valid = np.ones_like(self.labels, bool)
        result.registration['periodic_completion'] = {}
        PredictionFusionStep().run(self.inputs, self.match,
                                   LocatedObjects(prediction, self.inputs.template_valids))
        np.testing.assert_array_equal(result.base_labels, before)
        self.assertFalse(np.shares_memory(result.base_labels, result.mapped_labels))
        self.assertTrue((result.mapped_labels[4:8, 4:8] == 1).all())
        self.assertTrue((result.mapped_labels[16:18, 16:18] == 2).all())
        self.assertFalse(result.object_identities[4:8, 4:8].any())
        self.assertFalse(result.label_valid[:2].any())
        self.assertFalse(result.label_periodic_valid[16:18, 16:18].any())
        self.assertEqual(prediction.diagnostics['foreground_conflict_pixels'], 16)
        self.assertEqual(result.registration['periodic_completion']['remaining_unknown_pixels'], 128)

    def test_evaluation_keeps_ground_truth_roi_out_of_prepared_inference(self):
        evaluator = SegmentationEvaluationStep(self.config)
        gt_valid = np.zeros_like(self.labels)
        gt_valid[4:8, 4:8] = 1
        truth = evaluator.prepare(self.image, self.labels, gt_valid, None, None)
        LabelTransferStep().run(self.inputs, self.match, self.config)
        evaluated = evaluator.run(self.match.result, truth)
        self.assertEqual(evaluated.score, 100.)
        self.assertEqual(evaluated.metrics['evaluated_pixels'], 16)
        self.assertTrue(self.inputs.target_valid.all())


class PipelineOrderTests(unittest.TestCase):
    def setUp(self):
        self.config = MatcherConfig()
        image = np.zeros((64, 64, 3), np.uint8)
        labels = np.zeros((64, 64), np.uint16)
        self.inputs = InputPreparationStep(self.config).run([image], [labels], image)
        self.match = RegistrationStep(Mock(return_value=selection())).run(self.inputs, self.config)
        self.pipeline = MatchPipeline(self.config)
        self.calls = []
        for name in ('registration', 'ordered_objects', 'label_transfer',
                     'periodic_completion', 'object_support', 'fusion'):
            step = Mock()
            step.run.side_effect = lambda *args, name=name: self.calls.append(name)
            setattr(self.pipeline, name, step)
        self.pipeline.registration.run.side_effect = lambda *args: self.record('registration', self.match)

    def record(self, name, value):
        self.calls.append(name)
        return value

    def test_full_execution_order(self):
        located = Mock()
        self.pipeline.ordered_objects.run.side_effect = lambda *args: self.record('ordered_objects', located)
        self.assertIs(self.pipeline.run(self.inputs), self.match.result)
        self.assertEqual(self.calls, ['registration', 'ordered_objects', 'label_transfer',
                                      'periodic_completion', 'object_support', 'fusion'])

    def test_missing_geometry_stops_before_object_matching(self):
        self.match.matrix = None
        self.pipeline.run(self.inputs)
        self.assertEqual(self.calls, ['registration'])

    def test_rejection_or_missing_labels_stops_after_object_evidence(self):
        for accepted, labels in ((False, self.inputs.labels[0]), (True, None)):
            self.calls.clear()
            self.match.result.registration['accepted'] = accepted
            self.inputs.labels[0] = labels
            self.pipeline.run(self.inputs)
            self.assertEqual(self.calls, ['registration', 'ordered_objects'])

    def test_object_evidence_can_open_label_transfer_gate(self):
        self.match.result.registration['accepted'] = False

        def locate(*args):
            self.match.result.registration['accepted'] = True
            return self.record('ordered_objects', Mock())

        self.pipeline.ordered_objects.run.side_effect = locate
        self.pipeline.run(self.inputs)
        self.assertIn('label_transfer', self.calls)

    def test_absent_object_branch_skips_support_and_fusion(self):
        self.pipeline.ordered_objects.run.side_effect = lambda *args: self.record('ordered_objects', None)
        self.pipeline.run(self.inputs)
        self.assertEqual(self.calls, ['registration', 'ordered_objects', 'label_transfer', 'periodic_completion'])


if __name__ == '__main__':
    unittest.main()