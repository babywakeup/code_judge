"""Steps 3 and 6: independent identity matching and verified support completion."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...config import MatcherConfig
from ...utils.types import Mask
from ..models import PreparedInputs, RegisteredMatch
from .completion import complete_object_support, support_vectors
from .matching import ObjectPrediction, match_ordered_objects
from .retry import retry_cropped_identity


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

    def __init__(self, completer: Callable[..., None] = complete_object_support, *,
                 locator: Callable[..., ObjectPrediction] = match_ordered_objects) -> None:
        self.completer = completer
        self.locator = locator

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            located: LocatedObjects, config: MatcherConfig) -> None:
        periodic = match.result.registration.get('periodic_completion', {})
        # Geometry approval differs from label-copy acceptance. The object kernel
        # independently validates every actual support fill, including occlusion.
        approved = [c for c in periodic.get('copies', []) if c.get('geometry_accepted', c.get('accepted'))]
        lattice = periodic.get('lattice', []) if config.shared_periods else []
        if approved:
            assert match.matrix is not None
            source_vectors = periodic.get('period_vectors', [])
            target_vectors = [(match.matrix[:2, :2] @ np.asarray(v)).tolist() for v in source_vectors]
        else:
            if located.prediction.diagnostics.get('status') not in ('accepted', 'partial_reference'):
                return
            # A global semantic copy can fail even when the target repeats
            # accurately. Already located identities can use that independent
            # period as a proposal and still validate their actual local fills.
            lattice = [axis for axis in lattice if axis.get('verified')
                       and axis.get('target_verification', {}).get('verified')
                       and support_vectors([axis['target_vector']], config,
                                           source_vectors=[axis['source_vector']])]
            if not lattice:
                return
            source_vectors = [axis['source_vector'] for axis in lattice]
            target_vectors = [axis['target_vector'] for axis in lattice]
            located.prediction.diagnostics['support_geometry_source'] = 'verified_target_period'
        if lattice and len(lattice) == len(target_vectors):
            target_vectors = [axis['target_vector'] for axis in lattice]
            # Keep source-axis provenance: a source-y period may be diagonal in
            # the target after affine registration. This is geometry, not a new identity.
            located.prediction.diagnostics['periodic_lattice'] = lattice
        if (config.object_visible_count is not None
                and located.prediction.diagnostics.get('status') == 'ambiguous_identity'
                and (inputs.original_templates is not None or inputs.original_target is not None)):
            # An approved copy in the order direction cannot justify a perpendicular retry.
            shifts = [np.asarray(copy['offset']) @ target_vectors for copy in approved
                      if target_vectors and len(copy.get('offset', ())) == len(target_vectors)]
            source_shifts = [np.asarray(copy['offset']) @ source_vectors for copy in approved
                             if source_vectors and len(copy.get('offset', ())) == len(source_vectors)]
            if support_vectors(shifts, config, source_vectors=source_shifts if lattice else None):
                located.prediction = retry_cropped_identity(inputs, match, located, config, self.locator)
                match.result.registration['ordered_objects'] = located.prediction.diagnostics
                if lattice:
                    located.prediction.diagnostics['periodic_lattice'] = lattice
        if lattice:
            # Source vectors belong to this exact selected template, not every
            # alternate reference with the same semantic identity.
            located.prediction.diagnostics['periodic_lattice_template_index'] = match.template_index
        self.completer(located.prediction, inputs.templates, inputs.objects, located.source_valids,
                       inputs.target, inputs.target_valid, target_vectors, config)
