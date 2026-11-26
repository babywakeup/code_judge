"""Serializable configuration for paired-image DINOv2 change detection."""

from transformers import Dinov2Config, PretrainedConfig


class SceneChangeConfig(PretrainedConfig):
    model_type = "scene_change"
    sub_configs = {"backbone_config": Dinov2Config}

    def __init__(
        self,
        backbone_config=None,
        feature_layer=-1,
        feature_layer_pair=None,
        facet="query",
        facet_pair="query",
        num_attention_heads=1,
        num_attention_blocks=1,
        dropout=0.1,
        attention_mode="cross",
        decoder_hidden_size=None,
        output_size=None,
        classifier_relu=True,
        freeze_backbone=True,
        unfreeze_backbone_last_n_layers=0,
        num_labels=2,
        loss_ignore_index=255,
        class_weights=None,
        initializer_range=0.02,
        backbone_interpolation="hf",
        **kwargs,
    ):
        if "id2label" in kwargs and kwargs["id2label"] is not None:
            num_labels = len(kwargs["id2label"])
        if backbone_config is None:
            backbone_config = Dinov2Config(
                hidden_size=384, num_hidden_layers=12,
                num_attention_heads=6, image_size=518,
            )
        elif isinstance(backbone_config, dict):
            backbone_config = Dinov2Config(**backbone_config)
        if not isinstance(backbone_config, Dinov2Config):
            raise TypeError("backbone_config must be a Dinov2Config or dictionary")
        if backbone_config.num_channels != 3:
            raise ValueError("The paired RGB processor requires a three-channel backbone")
        if isinstance(backbone_config.patch_size, (list, tuple)):
            raise ValueError("Only square, integer DINOv2 patch_size is supported")
        if attention_mode not in {"cross", "legacy_self", "single_cross"}:
            raise ValueError("attention_mode must be cross, legacy_self or single_cross")
        if backbone_interpolation not in {"hf", "reference"}:
            raise ValueError("backbone_interpolation must be hf or reference")
        for value in (facet, facet_pair):
            if value not in {"query", "key", "value", "token"}:
                raise ValueError("facets must be query, key, value or token")
        n_layers = backbone_config.num_hidden_layers
        feature_layer_pair = feature_layer if feature_layer_pair is None else feature_layer_pair
        if not all(-n_layers <= layer < n_layers for layer in (feature_layer, feature_layer_pair)):
            raise ValueError("feature layer is outside the backbone")
        if num_attention_heads < 1 or backbone_config.hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_blocks < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid number of attention blocks or dropout")
        if not 0 <= unfreeze_backbone_last_n_layers <= n_layers:
            raise ValueError("Invalid unfreeze_backbone_last_n_layers")
        if num_labels < 2:
            raise ValueError("Use at least two classes for cross-entropy segmentation")
        if output_size is not None:
            if len(output_size) != 2 or any(int(x) < 1 for x in output_size):
                raise ValueError("output_size must contain two positive dimensions")
            output_size = list(output_size)
        if decoder_hidden_size is not None and decoder_hidden_size < 1:
            raise ValueError("decoder_hidden_size must be positive")
        if class_weights is not None:
            if len(class_weights) != num_labels or any(x <= 0 for x in class_weights):
                raise ValueError("class_weights must have one positive weight per class")
            class_weights = list(class_weights)
        # Preserve id2label/label2id provided by from_pretrained.
        kwargs.setdefault("id2label", {0: "unchanged", 1: "changed"} if num_labels == 2 else
                          {i: f"LABEL_{i}" for i in range(num_labels)})
        super().__init__(num_labels=num_labels, **kwargs)
        self.backbone_config = backbone_config
        self.feature_layer = feature_layer
        self.feature_layer_pair = feature_layer_pair
        self.facet = facet
        self.facet_pair = facet_pair
        self.num_attention_heads = num_attention_heads
        self.num_attention_blocks = num_attention_blocks
        self.dropout = dropout
        self.attention_mode = attention_mode
        self.decoder_hidden_size = decoder_hidden_size or backbone_config.hidden_size
        self.output_size = output_size
        self.classifier_relu = classifier_relu
        self.freeze_backbone = freeze_backbone
        self.unfreeze_backbone_last_n_layers = unfreeze_backbone_last_n_layers
        self.loss_ignore_index = loss_ignore_index
        self.class_weights = class_weights
        self.initializer_range = initializer_range
        self.backbone_interpolation = backbone_interpolation
