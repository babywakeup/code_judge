"""Declarative configuration for DINOv2/DINOv3 paired-image segmentation."""

from pathlib import Path

from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, Dinov2Config, PreTrainedConfig


@strict
class SceneChangeConfig(PreTrainedConfig):
    """Model defaults, nested backbone configuration and architecture validation."""

    model_type = "scene_change"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_name_or_path: str | Path | None = None
    use_pretrained_backbone: bool = False
    backbone_config: dict | PreTrainedConfig | None = None
    feature_layer: int = -1
    feature_layer_pair: int | None = None
    facet: str = "query"
    facet_pair: str = "query"
    num_attention_heads: int = 1
    num_attention_blocks: int = 1
    dropout: float | int = 0.1
    attention_mode: str = "cross"
    decoder_hidden_size: int | None = None
    output_size: list[int] | tuple[int, int] | None = None
    classifier_relu: bool = True
    freeze_backbone: bool = True
    unfreeze_backbone_last_n_layers: int = 0
    loss_ignore_index: int = 255
    class_weights: list[float | int] | tuple[float | int, ...] | None = None
    initializer_range: float | int = 0.02
    backbone_interpolation: str = "hf"

    def __post_init__(self, **kwargs):
        if self.backbone_name_or_path is not None:
            self.backbone_name_or_path = str(Path(self.backbone_name_or_path).expanduser().resolve())
        if self.backbone_config is None and self.backbone_name_or_path:
            if not Path(self.backbone_name_or_path).is_dir():
                raise FileNotFoundError(f"Local backbone directory not found: {self.backbone_name_or_path}")
            self.backbone_config = AutoConfig.from_pretrained(self.backbone_name_or_path, local_files_only=True)
        elif self.backbone_config is None:
            self.backbone_config = Dinov2Config(
                hidden_size=384, num_hidden_layers=12, num_attention_heads=6, image_size=518,
            )
        elif isinstance(self.backbone_config, dict):
            values = dict(self.backbone_config)
            self.backbone_config = AutoConfig.for_model(values.pop("model_type", "dinov2"), **values)
        if self.feature_layer_pair is None:
            self.feature_layer_pair = self.feature_layer
        if self.decoder_hidden_size is None:
            self.decoder_hidden_size = self.backbone_config.hidden_size
        if self.output_size is not None:
            self.output_size = list(self.output_size)
        if self.class_weights is not None:
            self.class_weights = list(self.class_weights)

        num_labels = kwargs.pop("num_labels", 2)
        if self.id2label is None:
            self.id2label = ({0: "unchanged", 1: "changed"} if num_labels == 2 else
                             {i: f"LABEL_{i}" for i in range(num_labels)})
        else:
            num_labels = len(self.id2label)
        super().__post_init__(num_labels=num_labels, **kwargs)

    def validate_architecture(self):
        backbone = self.backbone_config
        if backbone.model_type not in {"dinov2", "dinov3_vit"}:
            raise ValueError("Supported backbones are dinov2 and dinov3_vit (without task heads)")
        if self.use_pretrained_backbone and not self.backbone_name_or_path:
            raise ValueError("use_pretrained_backbone requires backbone_name_or_path")
        if backbone.model_type != "dinov2" and self.backbone_interpolation == "reference":
            raise ValueError("reference interpolation is only supported for DINOv2")
        if backbone.num_channels != 3:
            raise ValueError("The paired RGB processor requires a three-channel backbone")
        if not isinstance(backbone.patch_size, int) or backbone.patch_size < 1:
            raise ValueError("Only square, positive integer patch_size is supported")
        if self.attention_mode not in {"cross", "legacy_self", "single_cross"}:
            raise ValueError("attention_mode must be cross, legacy_self or single_cross")
        if self.backbone_interpolation not in {"hf", "reference"}:
            raise ValueError("backbone_interpolation must be hf or reference")
        if any(facet not in {"query", "key", "value", "token"} for facet in (self.facet, self.facet_pair)):
            raise ValueError("facets must be query, key, value or token")
        if not all(-backbone.num_hidden_layers <= layer < backbone.num_hidden_layers
                   for layer in (self.feature_layer, self.feature_layer_pair)):
            raise ValueError("feature layer is outside the backbone")
        if self.num_attention_heads < 1 or backbone.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_blocks < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid number of attention blocks or dropout")
        if not 0 <= self.unfreeze_backbone_last_n_layers <= backbone.num_hidden_layers:
            raise ValueError("Invalid unfreeze_backbone_last_n_layers")
        if self.num_labels < 2:
            raise ValueError("Use at least two classes for cross-entropy segmentation")
        if self.output_size is not None and (len(self.output_size) != 2 or min(self.output_size) < 1):
            raise ValueError("output_size must contain two positive dimensions")
        if self.decoder_hidden_size < 1:
            raise ValueError("decoder_hidden_size must be positive")
        if self.class_weights is not None:
            if len(self.class_weights) != self.num_labels or any(x <= 0 for x in self.class_weights):
                raise ValueError("class_weights must have one positive weight per class")


__all__ = ["SceneChangeConfig"]
