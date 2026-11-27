"""Convert local reference DINOv2 attention checkpoints into an HF directory.

Run as ``python -m segment.tools.convert_checkpoint --help``. Loading legacy
pickle objects is disabled unless explicitly enabled for a trusted file.
"""

import argparse
import re
from pathlib import Path

import torch
from transformers import Dinov2Config

from ..models.scene_change import SceneChangeConfig, SceneChangeForSemanticSegmentation, SceneChangeImageProcessor
from ..models.assets import resolve_local_backbone


BACKBONE_SPECS = {
    "dinov2_vits14": ("facebook/dinov2-small", 384, 12, 6, False),
    "dinov2_vitb14": ("facebook/dinov2-base", 768, 12, 12, False),
    "dinov2_vitl14": ("facebook/dinov2-large", 1024, 24, 16, False),
    "dinov2_vitg14": ("facebook/dinov2-giant", 1536, 40, 24, True),
}


def config_from_reference(options, backbone_config=None):
    name = options.get("name")
    if name not in {"dino2 + cross_attention", "dino2 + single_cross_attention1"}:
        raise ValueError(f"Unsupported reference architecture: {name!r}; only DINOv2 attention variants are supported")
    backbone_name = options.get("dino-model", "dinov2_vits14")
    if backbone_name not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported DINOv2 backbone: {backbone_name}")
    source, dim, layers, heads, swiglu = BACKBONE_SPECS[backbone_name]
    if backbone_config is None:
        backbone_config = Dinov2Config(
            hidden_size=dim, num_hidden_layers=layers, num_attention_heads=heads,
            image_size=518, use_swiglu_ffn=swiglu,
        )
    config = SceneChangeConfig(
        backbone_config=backbone_config,
        feature_layer=options.get("layer1", layers - 1),
        feature_layer_pair=options.get("layer2"),
        facet=options.get("facet1", "query"), facet_pair=options.get("facet2", "query"),
        num_attention_heads=options.get("num-heads", 1),
        num_attention_blocks=options.get("num-blocks", 1) if name.endswith("+ cross_attention") else 1,
        dropout=options.get("dropout-rate", 0.1),
        attention_mode="legacy_self" if name.endswith("+ cross_attention") else "single_cross",
        output_size=[options.get("target-shp-row", 504), options.get("target-shp-col", 504)],
        freeze_backbone=options.get("freeze-dino", True),
        unfreeze_backbone_last_n_layers=options.get("unfreeze-dino-last-n-layer", 0),
        classifier_relu=True,
        backbone_interpolation="reference",
    )
    config.backbone_pretrained_source = source
    config.reference_model_name = name
    return config


def convert_reference_state_dict(state_dict):
    """Translate fused MHA/QKV projections and deduplicate the shared backbone.

    Supports trainable-only and complete state_dict files. Unsupported names
    fail explicitly rather than disappearing from the converted checkpoint.
    """
    converted = {}

    def put(name, value):
        if name in converted and not torch.equal(converted[name], value):
            raise ValueError(f"Reference branches have different shared weights for {name}")
        converted[name] = value

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint state {key} is not a tensor")
        while key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("conv1."):
            put("fusion." + key[len("conv1."):], value)
            continue
        if key.startswith("conv2."):
            put("classifier." + key[len("conv2."):], value)
            continue
        match = re.fullmatch(r"ca0s\.(\d+)\.cross_att_([12])\.(.+)", key)
        single_match = re.fullmatch(r"ca\.(.+)", key)
        if match or single_match:
            prefix = (f"interactions.{match[1]}.branch{match[2]}." if match
                      else "interactions.0.branch1.")
            suffix = match[3] if match else single_match[1]
            mha = "multi_head_self_attention."
            if suffix.startswith(mha + "in_proj_"):
                kind = suffix[len(mha + "in_proj_"):]
                if kind not in {"weight", "bias"} or value.shape[0] % 3:
                    raise ValueError(f"Malformed fused MHA projection: {key}")
                for projection, tensor in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                    put(prefix + projection + "." + kind, tensor.contiguous())
            else:
                put(prefix + suffix.removeprefix(mha), value)
            continue
        match = re.fullmatch(r"backbone\.dino_[12]\.dino\.(.+)", key)
        if match:
            suffix = match[1]
            direct = {
                "cls_token": "embeddings.cls_token",
                "mask_token": "embeddings.mask_token",
                "pos_embed": "embeddings.position_embeddings",
                "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
                "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
                "norm.weight": "layernorm.weight", "norm.bias": "layernorm.bias",
            }
            if suffix in direct:
                put("backbone." + direct[suffix], value)
                continue
            block = re.fullmatch(r"blocks\.(\d+)\.(.+)", suffix)
            if block:
                prefix = f"backbone.encoder.layer.{block[1]}."
                component = block[2]
                if component in {"attn.qkv.weight", "attn.qkv.bias"}:
                    kind = component.rsplit(".", 1)[-1]
                    for projection, tensor in zip(("query", "key", "value"), value.chunk(3, dim=0)):
                        put(prefix + "attention.attention." + projection + "." + kind, tensor.contiguous())
                else:
                    for old, new in (("attn.proj.", "attention.output.dense."),
                                     ("ls1.gamma", "layer_scale1.lambda1"),
                                     ("ls2.gamma", "layer_scale2.lambda1"),
                                     ("mlp.w12.", "mlp.weights_in."),
                                     ("mlp.w3.", "mlp.weights_out.")):
                        if component.startswith(old):
                            component = new + component[len(old):]
                            break
                    put(prefix + component, value)
                continue
        raise ValueError(f"Unsupported reference state key: {key}")
    return converted


def load_reference_weights(model, state_dict, *, backbone_is_pretrained=False):
    """Reject incomplete heads and missing unfrozen backbone parameters."""
    converted = convert_reference_state_dict(state_dict)
    expected = model.state_dict()
    missing = set(expected) - set(converted)
    unexpected = set(converted) - set(expected)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise ValueError(f"Incompatible checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    trainable = {key for key, parameter in model.named_parameters() if parameter.requires_grad}
    if missing & trainable:
        raise ValueError(f"Checkpoint omits trainable backbone parameters: {sorted(missing & trainable)}")
    if missing and not backbone_is_pretrained:
        raise ValueError("Reference checkpoint omits frozen backbone weights; load a pretrained backbone first")
    model.load_state_dict(converted, strict=not missing)
    return sorted(missing)


def convert_checkpoint(checkpoint_path, output_dir, *, backbone_source=None,
                       local_files_only=True, allow_unsafe_pickle=False):
    """Convert offline; local_files_only is retained for existing Python callers."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=not allow_unsafe_pickle)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError("Expected a reference checkpoint containing model and args.model")
    config = config_from_reference(checkpoint["args"]["model"])
    # Training-only checkpoints omit frozen backbone weights. Restore them from
    # the corresponding local foundation model, never random initialization.
    source = resolve_local_backbone(backbone_source or config.backbone_pretrained_source)
    model = SceneChangeForSemanticSegmentation.from_backbone_pretrained(
        str(source), config=config, backbone_load_kwargs={"local_files_only": True},
    )
    load_reference_weights(model, checkpoint["model"], backbone_is_pretrained=True)
    optimizer_options = checkpoint["args"].get("optimizer", {})
    if optimizer_options.get("loss-weight"):
        model.config.class_weights = [0.025, 0.975]
    processor = SceneChangeImageProcessor(
        size={"height": config.output_size[0], "width": config.output_size[1]},
        patch_size=config.backbone_config.patch_size,
    )
    output_dir = Path(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    return model, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local reference .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output HF model directory")
    parser.add_argument("--backbone", help="Local HF DINOv2 directory (default: ../../hf_cache/facebook/<variant>)")
    parser.add_argument("--local-files-only", action="store_true", help="Compatibility flag; always offline")
    parser.add_argument("--allow-unsafe-pickle", action="store_true",
                        help="Only enable for trusted checkpoints requiring legacy pickle objects")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output, backbone_source=args.backbone,
                       local_files_only=args.local_files_only, allow_unsafe_pickle=args.allow_unsafe_pickle)
    print(f"Saved complete Transformers model and processor to {args.output}")


if __name__ == "__main__":
    main()
