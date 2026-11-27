"""DINOv2/DINOv3 change segmentation with Transformers-style components.

Architecture adapted from Chad Lin's Robust-Scene-Change-Detection (MIT).
See segment/LICENSE and segment/THIRD_PARTY_NOTICES.md.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_scene_change import SceneChangeConfig


@dataclass
class SceneChangeOutput(ModelOutput):
    """Pixel logits and optional loss/normalized pre-interaction patch features."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, torch.Tensor] | None = None


def scene_change_loss(logits, labels, *, ignore_index=255, class_weights=None):
    """Weighted pixel CE, resizing logits only; all-ignored batches yield zero."""
    if labels.ndim != 3 or labels.shape[0] != logits.shape[0] or labels.is_floating_point():
        raise ValueError("labels must be an integer B,H,W tensor of class IDs")
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    labels = labels.to(device=logits.device, dtype=torch.long)
    if not torch.any(labels != ignore_index):
        return logits.sum() * 0.0
    weights = logits.new_tensor(class_weights) if class_weights is not None else None
    return F.cross_entropy(logits, labels, weight=weights, ignore_index=ignore_index)


class SceneChangeAttentionBlock(nn.Module):
    """Reference post-norm MHA/FFN with PEFT-targetable projections."""

    def __init__(self, config):
        super().__init__()
        dim = config.backbone_config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.attention_dropout = config.dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(config.dropout),
        )

    def forward(self, query, context):
        batch = query.shape[0]

        def split(tensor):
            return tensor.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            split(self.q_proj(query)), split(self.k_proj(context)), split(self.v_proj(context)),
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, query.shape[1], -1)
        hidden = self.layer_norm1(query + self.dropout(self.out_proj(attention)))
        return self.layer_norm2(hidden + self.feed_forward(hidden))


class SceneChangeInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.attention_mode
        self.branch1 = SceneChangeAttentionBlock(config)
        if self.mode != "single_cross":
            self.branch2 = SceneChangeAttentionBlock(config)

    def forward(self, x, y):
        if self.mode == "single_cross":
            return self.branch1(x, y), y
        if self.mode == "legacy_self":
            # Preserve the reference double K/V exchange, which is self-attention.
            return self.branch1(x, x), self.branch2(y, y)
        return self.branch1(x, y), self.branch2(y, x)


class SceneChangePreTrainedModel(PreTrainedModel):
    """Shared HF metadata, initialization and saved backbone-freezing policy."""

    config_class = SceneChangeConfig
    base_model_prefix = "backbone"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2Layer", "DINOv3ViTLayer", "SceneChangeInteraction"]
    accepts_loss_kwargs = False

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        embeddings = self.backbone.get_input_embeddings()
        return getattr(embeddings, "projection", embeddings)

    @property
    def backbone_layers(self):
        """Native encoder layers without registering duplicate parameter paths."""
        if self.config.backbone_config.model_type == "dinov3_vit":
            return self.backbone.model.layer
        return self.backbone.encoder.layer

    def set_backbone_trainable(self, trainable=False, last_n_layers=0):
        """Call before adding adapters; do not freeze newly inserted LoRA weights."""
        layers = self.backbone_layers
        if not 0 <= last_n_layers <= len(layers):
            raise ValueError("last_n_layers is outside the backbone")
        self.backbone.requires_grad_(trainable)
        for layer in layers[len(layers) - last_n_layers:]:
            layer.requires_grad_(True)
        self.config.freeze_backbone = not trainable
        self.config.unfreeze_backbone_last_n_layers = last_n_layers

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Complete task checkpoints already contain the backbone. Never reopen
        # its original directory, even if the saved initialization flag is true.
        kwargs["_load_backbone_weights"] = False
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        # HF materialization may reset requires_grad; restore optimizer groups.
        model.set_backbone_trainable(
            not model.config.freeze_backbone, model.config.unfreeze_backbone_last_n_layers,
        )
        return result


class SceneChangeForSemanticSegmentation(SceneChangePreTrainedModel):
    """Shared DINOv2/DINOv3 features, paired interactions and a pixel-change head."""

    def __init__(self, config, *, _load_backbone_weights=True, backbone_load_kwargs=None):
        super().__init__(config)
        if config.use_pretrained_backbone and _load_backbone_weights:
            source = Path(config.backbone_name_or_path)
            if not source.is_dir():
                raise FileNotFoundError(f"Local backbone directory not found: {source}")
            load_kwargs = dict(backbone_load_kwargs or {})
            load_kwargs.update(local_files_only=True, output_loading_info=True)
            self.backbone, loading_info = AutoModel.from_pretrained(
                str(source), config=config.backbone_config, **load_kwargs,
            )
            # Do not silently leave a partly random backbone after loading an
            # incompatible source/config or ignore_mismatched_sizes=True.
            if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
                raise ValueError(f"Incomplete pretrained backbone: {loading_info}")
        else:
            self.backbone = AutoModel.from_config(config.backbone_config)
        self.interactions = nn.ModuleList(
            SceneChangeInteraction(config) for _ in range(config.num_attention_blocks)
        )
        dim = config.backbone_config.hidden_size
        single = config.attention_mode == "single_cross"
        self.fusion = nn.Conv2d(
            dim if single else dim * 2, config.decoder_hidden_size,
            kernel_size=3, padding=0 if single else 1,
        )
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        self.gradient_checkpointing = False
        self.post_init()
        self.set_backbone_trainable(
            not config.freeze_backbone, config.unfreeze_backbone_last_n_layers,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        """Build the task model, honoring its backbone initialization policy."""
        return cls._from_config(config, **kwargs)

    @classmethod
    def from_backbone_pretrained(cls, pretrained_model_name_or_path, *, config=None,
                                 backbone_load_kwargs=None, **config_kwargs):
        """Convenience entry point using the same config-driven loading policy."""
        if config is None:
            config = SceneChangeConfig(
                backbone_name_or_path=str(pretrained_model_name_or_path),
                use_pretrained_backbone=True, **config_kwargs,
            )
        elif config_kwargs:
            raise ValueError("Pass config or config keyword arguments, not both")
        else:
            config = deepcopy(config)
            config.backbone_name_or_path = str(Path(pretrained_model_name_or_path).expanduser().resolve())
            config.use_pretrained_backbone = True
        model = cls.from_config(config, backbone_load_kwargs=backbone_load_kwargs)
        model.config.backbone_pretrained_source = str(pretrained_model_name_or_path)
        return model

    def _embed_images(self, pixels):
        if self.config.backbone_interpolation == "hf":
            return self.backbone.embeddings(pixels)
        # Retain reference torch.hub DINOv2's 0.1 bicubic interpolation offset.
        embeddings = self.backbone.embeddings
        tokens = embeddings.patch_embeddings(pixels)
        tokens = torch.cat((embeddings.cls_token.expand(pixels.shape[0], -1, -1), tokens), dim=1)
        position = embeddings.position_embeddings
        if tokens.shape[1] != position.shape[1] or pixels.shape[-2] != pixels.shape[-1]:
            grid = int((position.shape[1] - 1) ** 0.5)
            patches = position[:, 1:].float().reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
            patch = self.config.backbone_config.patch_size
            patches = F.interpolate(
                patches, scale_factor=((pixels.shape[-2] // patch + 0.1) / grid,
                                       (pixels.shape[-1] // patch + 0.1) / grid),
                mode="bicubic", align_corners=False, antialias=False,
            )
            position = torch.cat((position[:, :1].float(), patches.flatten(2).transpose(1, 2)), dim=1)
            position = position.to(tokens.dtype)
        return embeddings.dropout(tokens + position)

    def get_image_features(self, pixel_values, layer_index=None, facet=None):
        """Return normalized patch Q/K/V or tokens, without hooks or a task head.

        Defaults match the first image branch. Q/K are linear projections before
        RoPE in DINOv3, matching native projection hooks. Earlier encoder layers
        use native RoPE attention. CLS/register tokens are excluded. Dynamically
        trainable LoRA projections remain in the autograd graph.
        """
        layers = self.backbone_layers
        dinov3 = self.config.backbone_config.model_type == "dinov3_vit"
        layer_index = self.config.feature_layer if layer_index is None else layer_index
        facet = self.config.facet if facet is None else facet
        if not -len(layers) <= layer_index < len(layers):
            raise ValueError("feature layer is outside the backbone")
        if facet not in {"query", "key", "value", "token"}:
            raise ValueError("facet must be query, key, value or token")
        layer_index %= len(layers)
        frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if frozen:
            self.backbone.eval()
        with torch.no_grad() if frozen else nullcontext():
            if dinov3:
                # Match native forward: RoPE and patch embeddings must see the
                # same input dtype, including when model weights are BF16/FP16.
                pixel_values = pixel_values.to(self.get_input_embeddings().weight.dtype)
            tokens = self._embed_images(pixel_values)
            layer_kwargs = {"position_embeddings": self.backbone.rope_embeddings(pixel_values)} if dinov3 else {}
            for layer in layers[:layer_index]:
                tokens = layer(tokens, **layer_kwargs)
                if isinstance(tokens, tuple):
                    tokens = tokens[0]
            layer = layers[layer_index]
            if facet == "token":
                features = layer(tokens, **layer_kwargs)
                if isinstance(features, tuple):
                    features = features[0]
            else:
                attention = layer.attention if dinov3 else layer.attention.attention
                projection = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[facet] if dinov3 else facet
                features = getattr(attention, projection)(layer.norm1(tokens))
            prefix_tokens = 1 + getattr(self.config.backbone_config, "num_register_tokens", 0)
            return F.normalize(features[:, prefix_tokens:], dim=-1)

    def _validate_inputs(self, pixel_values, pixel_values_pair):
        if pixel_values.ndim != 4 or pixel_values_pair.shape != pixel_values.shape:
            raise ValueError("Image pairs must have identical B,C,H,W shapes")
        _, channels, height, width = pixel_values.shape
        patch = self.config.backbone_config.patch_size
        if channels != 3 or height < patch or width < patch or height % patch or width % patch:
            raise ValueError(f"RGB input height and width must be positive multiples of {patch}")
        return height, width

    def _decode_features(self, x, y, image_size):
        fused = x if self.config.attention_mode == "single_cross" else torch.cat((x, y), dim=-1)
        patch = self.config.backbone_config.patch_size
        fused = fused.transpose(1, 2).reshape(x.shape[0], -1, image_size[0] // patch, image_size[1] // patch)
        if self.config.attention_mode == "single_cross" and min(fused.shape[-2:]) < 3:
            raise ValueError("single_cross requires at least a 3x3 patch grid")
        logits = self.classifier(F.relu(self.fusion(fused)))
        if self.config.classifier_relu:
            logits = F.relu(logits)
        return F.interpolate(logits, size=self.config.output_size or image_size, mode="bilinear", align_corners=False)

    def forward(self, pixel_values, pixel_values_pair, labels=None,
                output_hidden_states=None, return_dict=None, num_items_in_batch=None):
        """Compare paired B,3,H,W images and return B,num_labels,H,W logits."""
        image_size = self._validate_inputs(pixel_values, pixel_values_pair)
        output_hidden_states = (self.config.output_hidden_states if output_hidden_states is None
                                else output_hidden_states)
        return_dict = self.config.return_dict if return_dict is None else return_dict
        x = self.get_image_features(pixel_values)
        y = self.get_image_features(pixel_values_pair, self.config.feature_layer_pair, self.config.facet_pair)
        hidden_states = (x, y) if output_hidden_states else None
        for interaction in self.interactions:
            if self.gradient_checkpointing and self.training:
                x, y = self._gradient_checkpointing_func(interaction, x, y)
            else:
                x, y = interaction(x, y)
        logits = self._decode_features(x, y, image_size)
        loss = None if labels is None else scene_change_loss(
            logits, labels, ignore_index=self.config.loss_ignore_index, class_weights=self.config.class_weights,
        )
        output = SceneChangeOutput(loss=loss, logits=logits, hidden_states=hidden_states)
        return output if return_dict else output.to_tuple()


__all__ = ["SceneChangePreTrainedModel", "SceneChangeForSemanticSegmentation", "SceneChangeOutput"]
