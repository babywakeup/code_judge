"""Optional PEFT helpers; importing the model family does not require peft."""

import re


def get_lora_config(model, *, r=8, lora_alpha=16, lora_dropout=0.05,
                    adapt_backbone=False, **kwargs):
    """Create a LoRA config for interaction projections and the change head."""
    from peft import LoraConfig

    if any(name in kwargs for name in ("task_type", "target_modules", "modules_to_save")):
        raise ValueError("Use a custom LoraConfig to override the task/target/head policy")
    targets = []
    if model.config.backbone_config.model_type == "dinov3_vit":
        backbone_pattern = r"backbone\.model\.layer\.(\d+)\.attention\.(q_proj|k_proj|v_proj)"
        facet_names = {"q_proj": "query", "k_proj": "key", "v_proj": "value"}
    else:
        backbone_pattern = r"backbone\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)"
        facet_names = {"query": "query", "key": "key", "value": "value"}
    reachable = set()
    for selected, facet in ((model.config.feature_layer, model.config.facet),
                            (model.config.feature_layer_pair, model.config.facet_pair)):
        selected %= model.config.backbone_config.num_hidden_layers
        for index in range(selected + 1):
            facets = ("query", "key", "value") if index < selected or facet == "token" else (facet,)
            reachable.update((index, value) for value in facets)
    for name, _ in model.named_modules():
        if name.startswith("interactions.") and name.rsplit(".", 1)[-1] in {
            "q_proj", "k_proj", "v_proj", "out_proj",
        }:
            targets.append(name)
        if adapt_backbone:
            match = re.fullmatch(backbone_pattern, name)
            if match and (int(match[1]), facet_names[match[2]]) in reachable:
                targets.append(name)
    if not targets:
        raise ValueError("No supported attention projections were found")
    return LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=targets, modules_to_save=["fusion", "classifier"], **kwargs,
    )


__all__ = ["get_lora_config"]
