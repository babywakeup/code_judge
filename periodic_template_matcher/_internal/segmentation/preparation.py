"""Step 1: array validation, label normalization, ROI and object references."""

from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, normalize_masks, validate_image
from ...utils.types import Diagnostics, Image, LabelInput, Mask
from ..models import PreparedInputs, PreparedTarget, PreparedTemplates


class InputPreparationStep:
    """Resolve input policy once, before any registration or appearance scoring."""

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config

    @staticmethod
    def validate_masks(values: Sequence[Mask | None] | None, images: Sequence[Image],
                       name: str) -> Sequence[Mask | None]:
        if values is None:
            return [None] * len(images)
        if not isinstance(values, Sequence) or len(values) != len(images):
            raise ValueError(f'{name} must have one entry per template')
        for mask, image in zip(values, images):
            if mask is not None and (not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]):
                raise ValueError(f'{name} entries must match image height and width')
        return values

    def prepare_regions(self, templates: Sequence[Image], target: Image,
                        valids: Sequence[Mask | None], target_valid: Mask | None,
                        template_valid_heights: Sequence[int | None] | None,
                        target_valid_height: int | None) -> tuple[list[Mask], Mask, Diagnostics]:
        """Per-image heights override config; no height means full image. Masks intersect the ROI."""
        masks, regions = self._template_regions(templates, valids, template_valid_heights)
        target_policy = ValidRegion.from_config(self.config, target_valid_height)
        target_mask = target_policy.build_mask(target, target_valid)
        return masks, target_mask, dict(templates=regions, target=target_policy.describe(target, target_mask))

    def _template_policies(self, count: int, heights: Sequence[int | None] | None) -> list[ValidRegion]:
        if heights is None:
            heights = [None] * count
        if not isinstance(heights, Sequence) or len(heights) != count:
            raise ValueError('template_valid_heights must have one entry per template')
        return [ValidRegion.from_config(self.config, height) for height in heights]

    def _template_regions(self, templates: Sequence[Image], valids: Sequence[Mask | None],
                          heights: Sequence[int | None] | None) -> tuple[list[Mask], list[Diagnostics]]:
        policies = self._template_policies(len(templates), heights)
        masks = [policy.build_mask(image, mask) for policy, image, mask in zip(policies, templates, valids)]
        regions = [policy.describe(image, mask) for policy, image, mask in zip(policies, templates, masks)]
        return masks, regions

    def prepare_objects(self, references: Sequence[Sequence[Diagnostics]] | None,
                        templates: Sequence[Image], valids: Sequence[Mask],
                        annotations: Sequence[Mask | None]) -> list[list[Diagnostics]] | None:
        """Copy reference metadata; never attach derived state to caller objects."""
        if self.config.ordered_object_class is None:
            return None
        if references is None or len(references) != len(templates):
            raise ValueError('ordered matching requires template_objects for every template')
        objects = []
        for refs, image, valid, annotation in zip(references, templates, valids, annotations):
            names = [ref.get('identity') for ref in refs]
            if self.config.object_visible_count is not None:
                if (any(not isinstance(name, str) for name in names)
                        or len(set(names)) != len(names) or not set(names) <= set(self.config.object_cycle)):
                    raise ValueError('Partial template references must have unique configured identities')
            elif len(names) != len(self.config.object_cycle) or set(names) != set(self.config.object_cycle):
                raise ValueError('Each template must provide each ordered identity exactly once')
            prepared = []
            background_known = annotation is None or np.array_equal((annotation > 0) & (valid > 0), valid > 0)
            for ref in refs:
                mask = ref.get('mask')
                if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2] or not (mask > 0).any():
                    raise ValueError('Object reference mask must be nonempty and source-sized')
                prepared.append(dict(ref, background_known=background_known))
            if self.config.object_visible_count is not None and prepared:
                coordinate = 1 if self.config.object_order_axis == 'x' else 0
                def center(ref):
                    values = np.where(ref['mask'] > 0)[coordinate]
                    return (int(values.min()) + int(values.max())) / 2
                order = [ref['identity'] for ref in sorted(prepared, key=center)]
                cycle = list(self.config.object_cycle)
                if order not in [(cycle[i:] + cycle[:i])[:len(order)] for i in range(len(cycle))]:
                    raise ValueError('Partial template references must form a contiguous cyclic window along order_axis')
            objects.append(prepared)
        return objects

    def run(self, templates: Sequence[Image], masks: Sequence[LabelInput | None],
            target: Image, *, template_valids: Sequence[Mask | None] | None = None,
            annotation_valids: Sequence[Mask | None] | None = None,
            target_valid: Mask | None = None,
            template_valid_heights: Sequence[int | None] | None = None,
            target_valid_height: int | None = None,
            template_objects: Sequence[Sequence[Diagnostics]] | None = None) -> PreparedInputs:
        self.config.validate()
        sources = self.prepare_templates(templates, masks, template_valids=template_valids,
                                         annotation_valids=annotation_valids,
                                         template_valid_heights=template_valid_heights,
                                         template_objects=template_objects)
        return self.combine(sources, self.prepare_target(target, target_valid, target_valid_height))

    @staticmethod
    def _validate_image(image: Image) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError('Images must be numpy arrays; use read_image for file paths')
        validate_image(image)

    def prepare_target(self, image: Image, valid: Mask | None,
                       valid_height: int | None) -> PreparedTarget:
        self._validate_image(image)
        self.validate_masks([valid], [image], 'target_valid')
        policy = ValidRegion.from_config(self.config, valid_height)
        cropped, mask, _ = self._working_region(image, valid, policy)
        original = image if cropped.shape != image.shape else None
        return PreparedTarget(cropped, mask, policy.describe(image, mask), original)

    def _working_region(self, image: Image, valid: Mask | None,
                        policy: ValidRegion) -> tuple[Image, Mask, tuple[slice, slice]]:
        config = self.config
        if not (config.valid_x_start or config.valid_y_start
                or config.valid_x_end is not None or config.valid_y_end is not None):
            # Legacy height-only callers use the original canvas, including narrow valid bands.
            return image, policy.build_mask(image, valid), (slice(None), slice(None))
        region = policy.slices(image)
        cropped = image[region]
        if cropped.shape == image.shape:
            cropped = image
        mask = ValidRegion().build_mask(cropped, None if valid is None else valid[region])
        return cropped, mask, region

    def prepare_templates(self, templates: Sequence[Image], masks: Sequence[LabelInput | None], *,
                          template_valids: Sequence[Mask | None] | None = None,
                          annotation_valids: Sequence[Mask | None] | None = None,
                          template_valid_heights: Sequence[int | None] | None = None,
                          template_objects: Sequence[Sequence[Diagnostics]] | None = None) -> PreparedTemplates:
        if not isinstance(templates, Sequence) or not templates:
            raise ValueError('templates must be a nonempty list of images')
        for image in templates:
            self._validate_image(image)
        if not isinstance(masks, Sequence) or len(masks) != len(templates):
            raise ValueError('masks must have one entry per template (None is allowed)')
        labels = [None if mask is None else normalize_masks(mask, image.shape[:2], f'masks[{i}]')
                  for i, (mask, image) in enumerate(zip(masks, templates))]
        valids = self.validate_masks(template_valids, templates, 'template_valids')
        annotations = self.validate_masks(annotation_valids, templates, 'annotation_valids')
        policies = self._template_policies(len(templates), template_valid_heights)
        images, cropped_valids, regions, slices = [], [], [], []
        for image, mask, policy in zip(templates, valids, policies):
            cropped, valid, region = self._working_region(image, mask, policy)
            images.append(cropped)
            cropped_valids.append(valid)
            regions.append(policy.describe(image, valid))
            slices.append(region)
        cropped_annotations = [None if a is None else a[region] for a, region in zip(annotations, slices)]
        objects = self.prepare_objects(template_objects, templates, cropped_valids, cropped_annotations)
        if objects is not None:
            for refs, region in zip(objects, slices):
                for ref in refs:
                    ref['mask'] = ref['mask'][region]
        cropped_labels = [None if label is None else label[region] for label, region in zip(labels, slices)]
        originals = templates if any(cropped is not image for cropped, image in zip(images, templates)) else None
        return PreparedTemplates(images, cropped_labels, cropped_valids, cropped_annotations, regions,
                                 objects, originals)

    @staticmethod
    def combine(sources: PreparedTemplates, target: PreparedTarget) -> PreparedInputs:
        regions = deepcopy(dict(templates=sources.regions, target=target.region))
        return PreparedInputs(sources.images, sources.labels, target.image, sources.valids,
                              sources.annotation_valids, target.valid, regions, sources.objects,
                              sources.original_images, target.original_image)
