"""Step 7: combine base semantics and instance predictions without losing unknowns."""

import numpy as np

from ...config import MatcherConfig
from ..models import PreparedInputs, RegisteredMatch
from .steps import LocatedObjects


class PredictionFusionStep:
    """Snapshot base output and fuse with the configured identity precedence."""

    def run(self, inputs: PreparedInputs, match: RegisteredMatch,
            located: LocatedObjects, config: MatcherConfig | None = None) -> None:
        result = match.result
        prediction = located.prediction
        assert result.mapped_labels is not None and result.label_valid is not None
        # This snapshot includes base periodic fill, but excludes object fusion.
        result.base_labels = result.mapped_labels.copy()
        result.base_valid = result.label_valid.copy()
        foreground = result.base_valid & (result.base_labels > 0)
        overlap = (prediction.labels > 0) & foreground
        prediction.diagnostics['foreground_conflict_pixels'] = int(overlap.sum())
        identity_verified = (config is not None and config.object_visible_count is not None
                             and prediction.diagnostics.get('status') in ('accepted', 'partial_reference'))
        # Base periodic registration can have the right texture but the wrong
        # identity phase. A verified instance must retain its complete foreground.
        protected = np.zeros_like(foreground) if identity_verified else foreground
        positive = (prediction.labels > 0) & ~protected & inputs.target_roi
        prediction.diagnostics['instance_priority_pixels'] = int((positive & foreground).sum())
        result.mapped_labels[positive] = prediction.labels[positive]
        result.object_identities = np.where(positive, prediction.identities, 0).astype(np.uint16)
        result.object_valid = prediction.valid
        # Periodic background alone cannot prove that the instance class is absent.
        result.label_valid = ((result.label_valid & prediction.valid)
                              | positive | foreground) & inputs.target_roi
        result.mapped_labels[~result.label_valid] = 0
        if result.label_periodic_valid is not None:
            result.label_periodic_valid &= result.label_valid & ~positive
        if 'periodic_completion' in result.registration:
            info = result.registration['periodic_completion']
            missing = int((inputs.target_roi & ~result.label_valid).sum())
            info.update(status='complete' if not missing else 'incomplete',
                        label_coverage_roi=float(result.label_valid.sum() / inputs.target_roi.sum()),
                        remaining_unknown_pixels=missing)
