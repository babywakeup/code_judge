"""Transformers training with bounded-memory paired-image segmentation metrics."""

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Trainer


def prediction_masks(logits, labels):
    """Keep masks, not full logits, and align fixed-size outputs to the labels."""
    if isinstance(logits, tuple):
        logits = logits[0]
    if labels is not None and logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits.float(), size=labels.shape[-2:],
                               mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


class SegmentationMetrics:
    """Accumulate only a confusion matrix across gathered evaluation batches."""

    def __init__(self, num_labels, ignore_index=255):
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    def __call__(self, prediction, compute_result=False):
        def as_numpy(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

        labels = as_numpy(prediction.label_ids).reshape(-1)
        masks = as_numpy(prediction.predictions).reshape(-1)
        valid = (labels != self.ignore_index) & (labels >= 0) & (labels < self.num_labels)
        valid &= (masks >= 0) & (masks < self.num_labels)
        encoded = labels[valid] * self.num_labels + masks[valid]
        self.confusion += np.bincount(encoded, minlength=self.num_labels ** 2).reshape(self.confusion.shape)
        if not compute_result:
            return {}
        diagonal = self.confusion.diagonal().astype(np.float64)
        union = self.confusion.sum(axis=0) + self.confusion.sum(axis=1) - diagonal
        iou = diagonal / np.maximum(union, 1)
        metrics = {
            "foreground_miou": float(iou[1:].mean() if self.num_labels > 1 else iou.mean()),
            "pixel_accuracy": float(diagonal.sum() / max(self.confusion.sum(), 1)),
        }
        self.confusion.fill(0)
        return metrics


class SceneChangeTrainer(Trainer):
    """Use HF optimization/checkpoints/AMP with the model's paired-image loss.

    The model owns weighted CE and ignored-label semantics. Evaluation gathers
    per-batch masks across processes and reduces them to a confusion matrix,
    rather than retaining dataset-sized full-resolution logits in memory.
    """

    def __init__(self, model, args, **kwargs):
        args.label_names = ["labels"]
        args.remove_unused_columns = False  # Preserve both image inputs, also under PEFT.
        args.batch_eval_metrics = True
        kwargs.setdefault("compute_metrics", SegmentationMetrics(
            model.config.num_labels, model.config.loss_ignore_index,
        ))
        kwargs.setdefault("preprocess_logits_for_metrics", prediction_masks)
        super().__init__(model=model, args=args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss).all():
            raise RuntimeError("Scene-change model returned a non-finite loss")
        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        ignored = list(dict.fromkeys([*(ignore_keys or []), "hidden_states"]))
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignored)
