"""Synthetic forward/backward check, called by ``python -m segment.demo --smoke``."""

import torch
from transformers import Dinov2Config, set_seed

from ..models.scene_change import SceneChangeConfig, SceneChangeForSemanticSegmentation, SceneChangeProcessor
from ..training.config import make_model_and_processor, parse_args
from ..training.runner import resolve_device


def run_smoke(args):
    set_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = resolve_device(args.device)
    if args.init_model or args.resume_from_checkpoint:
        model, processor = make_model_and_processor(args)
    else:
        backbone = (Dinov2Config(hidden_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, image_size=56)
                    if args.smoke_model_size == "tiny" else SceneChangeConfig().backbone_config)
        model = SceneChangeForSemanticSegmentation(SceneChangeConfig(
            backbone_config=backbone, classifier_relu=False, dropout=0,
        ))
        processor = SceneChangeProcessor(patch_size=backbone.patch_size)

    image_size = args.image_size if args.image_size is not None else 56
    patch = model.config.backbone_config.patch_size
    if image_size < patch or image_size % patch:
        raise ValueError(f"image-size must be a positive multiple of patch size {patch}")
    model = model.to(device).train()
    before = torch.randint(0, 256, (3, image_size, image_size), dtype=torch.uint8)
    after = before.clone()
    labels = torch.zeros(image_size, image_size, dtype=torch.long)
    start, end = image_size // 4, image_size * 3 // 4
    after[:, start:end, start:end] = 255 - after[:, start:end, start:end]
    labels[start:end, start:end] = 1
    inputs = processor(before, after, segmentation_maps=labels).to(device)
    output = model(**inputs)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("Smoke check returned an invalid training loss")
    output.loss.backward()
    if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters()):
        raise RuntimeError("Smoke check found no nonzero gradients")
    print(f"Smoke check passed; loss={output.loss.item():.6f}, logits={tuple(output.logits.shape)}")
    return output


if __name__ == "__main__":
    run_smoke(parse_args())
