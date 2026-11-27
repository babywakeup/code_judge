"""CLI options and explicit model/dataset construction for debugging."""

import argparse
from pathlib import Path

from transformers import Dinov2Config

from ..data import CroppedPSCD, VLCMUCD
from ..models.scene_change import SceneChangeConfig, SceneChangeForSemanticSegmentation, SceneChangeProcessor
from ..models.assets import HF_CACHE_ROOT, resolve_local_backbone


DEFAULT_BACKBONE = HF_CACHE_ROOT / "facebook/dinov2-small"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train scene-change segmentation with a local DINOv2 backbone.")
    parser.add_argument("--dataset", choices=("vl-cmu-cd", "pscd"), default="vl-cmu-cd")
    parser.add_argument("--data-root", type=Path, help="Dataset root; omitted uses segment.data defaults")
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE,
                        help="Local backbone (default: ../../hf_cache/facebook/dinov2-small)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-model", type=Path, help="Complete saved scene-change model, not an adapter")
    source.add_argument("--random-backbone", action="store_true", help="Tiny offline pipeline check only")
    parser.add_argument("--smoke", action="store_true", help="Synthetic forward/backward check, no dataset needed")
    parser.add_argument("--smoke-model-size", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--local-files-only", action="store_true", help="Compatibility flag; loading is always offline")
    parser.add_argument("--image-size", type=int, help="Optional square crop, a multiple of the backbone patch size")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Trainer manages placement; use CUDA_VISIBLE_DEVICES to select GPUs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Optimizer updates; 0 means no step limit")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path, help="Restore Trainer model/optimizer/scheduler/RNG state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene-change-demo"))
    return parser.parse_args(argv)


def make_model_and_processor(args):
    checkpoint = args.resume_from_checkpoint or args.init_model
    if checkpoint:
        model = SceneChangeForSemanticSegmentation.from_pretrained(checkpoint, local_files_only=True)
        processor = SceneChangeProcessor.from_pretrained(checkpoint, local_files_only=True)
        if args.unfreeze_backbone:
            model.set_backbone_trainable(True)
    else:
        options = dict(classifier_relu=False, freeze_backbone=not args.unfreeze_backbone)
        if args.random_backbone:
            backbone = Dinov2Config(hidden_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, image_size=56, patch_size=14)
            model = SceneChangeForSemanticSegmentation(SceneChangeConfig(
                backbone_config=backbone, dropout=0, **options,
            ))
        else:
            model = SceneChangeForSemanticSegmentation.from_backbone_pretrained(
                str(resolve_local_backbone(args.backbone)),
                backbone_load_kwargs={"local_files_only": True}, **options,
            )
        processor = SceneChangeProcessor(patch_size=model.config.backbone_config.patch_size)

    if args.image_size is not None:
        patch = model.config.backbone_config.patch_size
        if args.image_size < patch or args.image_size % patch:
            raise ValueError(f"image-size must be a positive multiple of patch size {patch}")
        # Preserve normalization, mask mapping and ignore-index settings on reload.
        processor.size = {"height": args.image_size, "width": args.image_size}
    return model, processor


def make_datasets(args, processor):
    dataset_type = VLCMUCD if args.dataset == "vl-cmu-cd" else CroppedPSCD
    train_dataset = dataset_type(args.data_root, mode="train", processor=processor, hflip_prob=args.hflip_prob)
    eval_dataset = dataset_type(args.data_root, mode="val", processor=processor)
    if not len(train_dataset) or not len(eval_dataset):
        raise ValueError("Train and validation datasets must both be nonempty")
    return train_dataset, eval_dataset
