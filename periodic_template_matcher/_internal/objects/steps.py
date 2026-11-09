"""Steps 3 and 6: independent identity matching and verified support completion."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...config import MatcherConfig
from ...utils.types import Mask
from ..models import PreparedInputs, RegisteredMatch
from .completion import complete_object_support
from .matching import ObjectPrediction, match_ordered_objects


@dataclass
class LocatedObjects:
    """Prediction and the exact source evidence masks reused for support filling."""

    prediction: ObjectPrediction
    source_valids: list[Mask]


class OrderedObjectStep:
    """Locate identities independently; strong evidence may resolve a template tie."""

    def __init__(self, locator: Callable[..., ObjectPrediction] = match_ordered_objects) -> None:
        self.locator = locator

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            config: MatcherConfig) -> LocatedObjects | None:
        if inputs.objects is None or config.label_mode != 'semantic':
            return None
        # Appearance evidence must satisfy both the image ROI and annotation ROI.
        source_valids = [
            image_valid if annotation_valid is None
            else image_valid * (annotation_valid > 0).astype(np.uint8)
            for image_valid, annotation_valid
            in zip(inputs.template_valids, inputs.annotation_valids)
        ]
        result = match.result
        registration = result.registration
        # Rejected geometry contributes no reference, even when selection succeeds.
        matrices = [
            candidate.get('matrix_template_to_target')
            if candidate.get('geometry_accepted', result.accepted) else None
            for candidate in registration['matches']
        ]
        prediction = self.locator(inputs.templates, inputs.objects, source_valids, inputs.target,
                                  inputs.target_valid, matrices, config)
        registration['ordered_objects'] = prediction.diagnostics
        if (not result.accepted and registration['status'] == 'templates_too_close'
                and prediction.diagnostics['status'] == 'accepted'):
            registration.update(accepted=True, status='accepted_ordered_objects',
                                template_selection_ambiguous=True)
        return LocatedObjects(prediction, source_valids)


class ObjectSupportCompletionStep:
    """Use approved lattice geometry, never unconditional periodic instance copies."""

    def __init__(self, completer: Callable[..., None] = complete_object_support) -> None:
        self.completer = completer

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            located: LocatedObjects, config: MatcherConfig) -> None:
        periodic = match.result.registration.get('periodic_completion', {})
        # Geometry approval differs from label-copy acceptance. The object kernel
        # independently validates every actual support fill, including occlusion.
        if not any(c.get('geometry_accepted', c.get('accepted')) for c in periodic.get('copies', [])):
            return
        assert match.matrix is not None
        target_vectors = [(match.matrix[:2, :2] @ np.asarray(v)).tolist()
                          for v in periodic.get('period_vectors', [])]
        self.completer(located.prediction, inputs.templates, inputs.objects, located.source_valids,
                       inputs.target, inputs.target_valid, target_vectors, config)
