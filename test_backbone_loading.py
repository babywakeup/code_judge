"""Load real local backbones and check pretrained/random initialization offline.

Run directly from the repository root. Defaults test both initialization policies
for cached DINOv2-small, DINOv2-base and DINOv3-ViT-B/16. No data or downloads are
needed. Use --backbone to select other local DINOv2 sizes, and --roundtrip to also
save/reload complete task checkpoints (requires temporary disk space).
"""

import argparse
import gc
import json
import tempfile
from unittest.mock import patch

from safetensors import safe_open
import torch
from transformers import AutoModel

from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation
from segment.models.assets import resolve_local_backbone


DEFAULT_BACKBONES = (
    "facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov3-vitb16-pretrain-lvd1689m",
)


def checkpoint_shards(source):
    """Return safetensors files and their keys without loading a second model."""
    single = source / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as reader:
            return {single: list(reader.keys())}
    index = source / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"This verification script requires local safetensors weights: {source}")
    shards = {}
    for name, filename in json.loads(index.read_text())["weight_map"].items():
        shards.setdefault(source / filename, []).append(name)
    return shards


def verify_pretrained_weights(model, source):
    """Compare every backbone tensor exactly with the local checkpoint."""
    actual = model.backbone.state_dict()
    checked = set()
    for filename, names in checkpoint_shards(source).items():
        with safe_open(filename, framework="pt", device="cpu") as reader:
            for name in names:
                # Official 4.x DINOv3 checkpoints used layer.*, renamed by
                # Transformers 5.x to model.layer.* during native loading.
                target = ("model." + name if model.config.backbone_config.model_type == "dinov3_vit"
                          and name.startswith("layer.") else name)
                if target not in actual:
                    raise AssertionError(f"Unexpected source tensor: {name}")
                if not torch.equal(actual[target].cpu(), reader.get_tensor(name)):
                    raise AssertionError(f"Backbone tensor differs from pretrained checkpoint: {name}")
                checked.add(target)
    if checked != set(actual):
        raise AssertionError(f"Unchecked backbone tensors: {sorted(set(actual) - checked)}")
    return len(checked)


def verify_random_patch_weights(model, source):
    """Ensure random construction did not copy the checkpoint's patch kernel."""
    name = ("embeddings.patch_embeddings.weight" if model.config.backbone_config.model_type == "dinov3_vit"
            else "embeddings.patch_embeddings.projection.weight")
    for filename, names in checkpoint_shards(source).items():
        if name in names:
            with safe_open(filename, framework="pt", device="cpu") as reader:
                if torch.equal(model.get_input_embeddings().weight.cpu(), reader.get_tensor(name)):
                    raise AssertionError("Random backbone unexpectedly matches pretrained patch weights")
            return
    raise AssertionError(f"Patch weights not found: {name}")


def run_case(source, pretrained, *, device, roundtrip):
    config = SceneChangeConfig(
        backbone_name_or_path=source, use_pretrained_backbone=pretrained,
        classifier_relu=False, dropout=0,
    )
    if pretrained:
        model = SceneChangeForSemanticSegmentation.from_config(config).eval()
        verified = verify_pretrained_weights(model, source)
    else:
        # A false flag must not invoke a weight loader, not merely yield a
        # different parameter after loading and then reinitializing the model.
        with patch.object(AutoModel, "from_pretrained", side_effect=AssertionError("Random mode loaded weights")):
            model = SceneChangeForSemanticSegmentation.from_config(config).eval()
        verify_random_patch_weights(model, source)
        verified = 0
    model.to(device)
    patch_size = config.backbone_config.patch_size
    images = torch.randn(1, 3, patch_size * 3, patch_size * 4, device=device)
    labels = torch.zeros(1, patch_size * 3, patch_size * 4, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(images, images.flip(-1), labels=labels, output_hidden_states=True)
        expected = (1, config.num_labels, patch_size * 3, patch_size * 4)
        if tuple(output.logits.shape) != expected or not torch.isfinite(output.loss):
            raise AssertionError(f"Invalid output/loss: {output.logits.shape}, {output.loss}")
        for facet in ("query", "key", "value", "token"):
            features = model.get_image_features(images, facet=facet)
            if tuple(features.shape) != (1, 12, config.backbone_config.hidden_size):
                raise AssertionError(f"Invalid {facet} patch grid: {features.shape}")
            if not torch.isfinite(features).all():
                raise AssertionError(f"Non-finite {facet} features")

    if roundtrip:
        with tempfile.TemporaryDirectory(prefix="scene-change-backbone-") as directory:
            model.save_pretrained(directory)
            with patch.object(AutoModel, "from_pretrained", side_effect=AssertionError("Reload read external weights")):
                restored = SceneChangeForSemanticSegmentation.from_pretrained(directory, local_files_only=True)
            restored.to(device).eval()
            with torch.no_grad():
                torch.testing.assert_close(output.logits, restored(images, images.flip(-1)).logits, rtol=0, atol=0)
    mode = "pretrained" if pretrained else "random"
    print(f"PASS {source.name}: mode={mode}, type={config.backbone_config.model_type}, "
          f"hidden={config.backbone_config.hidden_size}, patch={patch_size}, "
          f"logits={tuple(output.logits.shape)}, exact_weight_tensors={verified}, "
          f"loss={output.loss.item():.6f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", nargs="+", default=DEFAULT_BACKBONES, help="Local backbone paths or facebook/ IDs")
    parser.add_argument("--mode", choices=("both", "pretrained", "random"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--roundtrip", action="store_true", help="Also save/reload full task models")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    modes = (True, False) if args.mode == "both" else (args.mode == "pretrained",)
    count = 0
    for name in args.backbone:
        source = resolve_local_backbone(name)
        for pretrained in modes:
            run_case(source, pretrained, device=args.device, roundtrip=args.roundtrip)
            count += 1
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    print(f"All {count} local backbone loading checks passed.")


if __name__ == "__main__":
    main()
