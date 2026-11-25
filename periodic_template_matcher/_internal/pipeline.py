"""Explicit inference order. This module owns control flow, not image algorithms."""

import numpy as np

from ..config import MatcherConfig
from .models import MatchResult, PreparedInputs
from .objects.fusion import PredictionFusionStep
from .objects.steps import ObjectSupportCompletionStep, OrderedObjectStep
from .registration.step import RegistrationStep
from .segmentation.instances import complete_periodic_instances
from .segmentation.occlusion import LatticeOcclusionCompletionStep
from .segmentation.reference_instances import complete_direct_references
from .segmentation.transfer import LabelTransferStep, PeriodicCompletionStep


class MatchPipeline:
    """Run stateless steps against per-call data; no file IO or ground truth.

    Identity matching precedes the label acceptance gate because strong identity
    evidence can resolve ``templates_too_close``. Support completion follows base
    periodic completion because it requires the latter's verified geometry.
    """

    def __init__(self, config: MatcherConfig, *,
                 registration: RegistrationStep | None = None,
                 ordered_objects: OrderedObjectStep | None = None,
                 object_support: ObjectSupportCompletionStep | None = None) -> None:
        self.config = config
        self.registration = registration if registration is not None else RegistrationStep()
        self.ordered_objects = ordered_objects if ordered_objects is not None else OrderedObjectStep()
        self.label_transfer = LabelTransferStep()
        self.periodic_completion = PeriodicCompletionStep()
        self.object_support = object_support if object_support is not None else ObjectSupportCompletionStep()
        self.occlusion_completion = LatticeOcclusionCompletionStep()
        self.fusion = PredictionFusionStep()

    def run(self, inputs: PreparedInputs) -> MatchResult:
        match = self.registration.run(inputs, self.config)
        partial_cycle = self.config.object_visible_count is not None and inputs.objects is not None
        if match.matrix is None and not partial_cycle:
            return match.result

        located = self.ordered_objects.run(inputs, match, self.config)
        if not match.result.accepted or inputs.labels[match.template_index] is None:
            if (partial_cycle and located is not None
                    and located.prediction.diagnostics['status'] in ('accepted', 'partial_reference')):
                # Local object evidence does not authorize base semantic transfer
                # or claims about background/other classes outside the objects.
                result, prediction = match.result, located.prediction
                result.registration.update(base_registration_accepted=result.accepted,
                    base_registration_status=result.registration['status'],
                    accepted=True, status='accepted_objects_only', prediction_scope='ordered_objects')
                result.mapped_labels = prediction.labels.copy()
                result.label_valid = (prediction.labels > 0) & inputs.target_roi
                result.object_identities = prediction.identities.copy()
                result.object_valid = result.label_valid.copy()
                result.object_labels = np.where(inputs.target_roi, prediction.labels, 0).astype(np.uint16)
                result.object_layer_valid = (prediction.labels > 0) & inputs.target_roi
                result.object_layer_identities = np.where(result.object_layer_valid,
                                                          prediction.identities, 0).astype(np.uint16)
                result.base_labels = np.zeros_like(prediction.labels)
                result.base_valid = np.zeros_like(result.label_valid)
            return match.result

        transferred = self.label_transfer.run(inputs, match, self.config)
        result = match.result
        complete_direct_references(result, inputs, match.template_index, transferred, self.config)
        self.periodic_completion.run(inputs, match, transferred, self.config)
        # Lattice-derived poses become available only after periodic verification.
        complete_direct_references(result, inputs, match.template_index, transferred, self.config, phase_only=True)
        if located is not None:
            self.fusion.capture_base(result)
        alternates = (dict(labels=inputs.labels[index], annotation_valid=inputs.annotation_valids[index],
                           source_valid=inputs.template_valids[index], template_index=index,
                           source_image=inputs.templates[index])
                      for index in result.registration['ranking']
                      if index != match.template_index and inputs.labels[index] is not None)
        complete_periodic_instances(result, inputs.target_roi, self.config,
            reference=dict(labels=transferred.labels, annotation_valid=transferred.annotation_valid,
                           source_valid=inputs.template_valids[match.template_index],
                           template_index=match.template_index), alternate_references=alternates)
        if located is not None:
            self.object_support.run(inputs, match, located, self.config)
        self.occlusion_completion.run(inputs, result, self.config,
                                      located.prediction if located is not None else None)
        if transferred.supplemental_background is not None:
            background = transferred.supplemental_background & (result.mapped_labels == 0)
            result.direct_valid |= background
            result.label_valid |= background
            periodic = result.registration['periodic_completion']
            missing = int((inputs.target_roi & ~result.label_valid).sum())
            periodic.update(supplemental_background_pixels=int(background.sum()),
                            remaining_unknown_pixels=missing,
                            label_coverage_roi=float(result.label_valid.sum()/inputs.target_roi.sum()),
                            status=periodic['status'] if missing else 'complete')
        if result.label_periodic_valid is not None and result.completion_valid is None:
            result.completion_valid = result.label_periodic_valid.copy()
            result.completion_labels = np.where(result.completion_valid, result.mapped_labels, 0).astype(np.uint16)
        if located is not None:
            self.fusion.run(inputs, match, located, self.config)
        return result
