"""Explicit inference order. This module owns control flow, not image algorithms."""

import numpy as np

from ..config import MatcherConfig
from .models import MatchResult, PreparedInputs
from .objects.fusion import PredictionFusionStep
from .objects.steps import ObjectSupportCompletionStep, OrderedObjectStep
from .registration.step import RegistrationStep
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
                result.base_labels = np.zeros_like(prediction.labels)
                result.base_valid = np.zeros_like(result.label_valid)
            return match.result

        transferred = self.label_transfer.run(inputs, match, self.config)
        self.periodic_completion.run(inputs, match, transferred, self.config)
        if located is not None:
            self.object_support.run(inputs, match, located, self.config)
            self.fusion.run(inputs, match, located, self.config)
        return match.result
