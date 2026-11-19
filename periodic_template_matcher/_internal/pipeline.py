"""Explicit inference order. This module owns control flow, not image algorithms."""

import cv2
import numpy as np

from ..config import MatcherConfig
from .models import MatchResult, PreparedInputs
from .objects.fusion import PredictionFusionStep
from .objects.steps import ObjectSupportCompletionStep, OrderedObjectStep
from .registration.step import RegistrationStep
from .reference.step import NormalReferenceStep
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
        self.normal_reference = NormalReferenceStep()

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
                self._object_reference(inputs, match, located)
                self.normal_reference.run(inputs, result)
            return match.result

        transferred = self.label_transfer.run(inputs, match, self.config)
        self.periodic_completion.run(inputs, match, transferred, self.config)
        if located is not None:
            self.object_support.run(inputs, match, located, self.config)
            self.fusion.run(inputs, match, located, self.config)
            self._object_reference(inputs, match, located)
        self.normal_reference.run(inputs, match.result)
        return match.result

    @staticmethod
    def _object_reference(inputs, match, located):
        result = match.result
        # Global alignment cannot place independently localized objects. Remove
        # their old source footprints before pasting verified instance geometry.
        if inputs.objects is not None:
            source_masks = {}
            for piece in result.reference_pieces:
                index = piece.template_index
                if index not in source_masks:
                    mask = np.zeros(inputs.templates[index].shape[:2], np.uint8)
                    for obj in inputs.objects[index]:
                        mask |= (obj['mask'] > 0).astype(np.uint8)
                    source_masks[index] = mask
                footprint = cv2.warpPerspective(source_masks[index].astype(np.float32), piece.matrix,
                    (inputs.target.shape[1], inputs.target.shape[0])) > 0
                piece.mask &= ~footprint
        for piece in located.prediction.reference_pieces:
            # Only actual instance foreground can override base reference geometry.
            positive = (result.object_identities > 0) & (located.prediction.labels > 0)
            piece.mask &= positive & inputs.target_roi
            if piece.mask.any():
                result.reference_pieces.append(piece)
