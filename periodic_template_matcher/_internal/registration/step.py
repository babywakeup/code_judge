"""Step 2: select a source and expose direct image geometry, not labels."""

from collections.abc import Callable
from dataclasses import asdict, replace

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import warp_valid_mask
from ...utils.types import Diagnostics
from ..models import MatchResult, PreparedInputs, RegisteredMatch
from .core import select_templates


class RegistrationStep:
    """Own template selection and direct image/ROI projection.

    The selector is injectable for isolated orchestration tests. All numerical
    registration, backend selection and ranking stay in the registration kernel.
    """

    def __init__(self, selector: Callable[..., Diagnostics] = select_templates) -> None:
        self.selector = selector

    def run(self, inputs: PreparedInputs, config: MatcherConfig) -> RegisteredMatch:
        # InputPreparationStep resolved heights already. Do not exclude rows twice.
        registration_config = replace(config, valid_height=None)
        registration = self.selector(inputs.templates, inputs.target, registration_config,
                                     inputs.template_valids, inputs.target_valid, config.label_mode)
        registration.update(config=asdict(config), valid_regions=inputs.regions)
        index = registration['ranking'][0]
        source = inputs.templates[index]
        matrix = registration['matches'][index].get('matrix_template_to_target')
        result = MatchResult(registration, inputs.target, source, None,
                             np.zeros(inputs.target.shape[:2], bool))
        if matrix is None:
            return RegisteredMatch(result, index, None)
        matrix = np.asarray(matrix, np.float64)
        result.warped_template = cv2.warpPerspective(
            source, matrix, (inputs.target.shape[1], inputs.target.shape[0]))
        result.geometric_valid = warp_valid_mask(inputs.template_valids[index], matrix, inputs.target.shape)
        result.geometric_valid &= inputs.target_roi
        return RegisteredMatch(result, index, matrix)