"""Transformers-native adaptation of the reference DINOv2 attention model.

Architecture adapted from Chad Lin's Robust-Scene-Change-Detection (MIT).
See LICENSE and THIRD_PARTY_NOTICES.md in this package.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Model, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_scene_change import SceneChangeConfig


@dataclass
class SceneChangeOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, torch.Tensor] | None = None


class SceneChangeAttentionBlock(nn.Module):
    """Original post-norm MHA/FFN block with PEFT-targetable projections."""

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

        def split(x):
            return x.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            split(self.q_proj(query)), split(self.k_proj(context)), split(self.v_proj(context)),
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, query.shape[1], -1)
        x = self.layer_norm1(query + self.dropout(self.out_proj(attention)))
        return self.layer_norm2(x + self.feed_forward(x))


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
            # The original apply_cross_attention and TwoCrossAttention swap K/V
            # twice. This mode intentionally reproduces the resulting self-attention.
            return self.branch1(x, x), self.branch2(y, y)
        return self.branch1(x, y), self.branch2(y, x)


class SceneChangeForSemanticSegmentation(PreTrainedModel):
    config_class = SceneChangeConfig
    base_model_prefix = "backbone"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2Layer", "SceneChangeInteraction"]
    # Loss uses a per-image pixel mean, not Trainer's item-count normalization.
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Dinov2Model(config.backbone_config)
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

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.backbone.embeddings.patch_embeddings.projection

    def set_backbone_trainable(self, trainable=False, last_n_layers=0):
        """Call BEFORE adding adapters; never freeze newly inserted LoRA weights."""
        layers = self.backbone.encoder.layer
        if not 0 <= last_n_layers <= len(layers):
            raise ValueError("last_n_layers is outside the backbone")
        self.backbone.requires_grad_(trainable)
        if last_n_layers:
            for layer in layers[-last_n_layers:]:
                layer.requires_grad_(True)
        self.config.freeze_backbone = not trainable
        self.config.unfreeze_backbone_last_n_layers = last_n_layers

    @classmethod
    def from_backbone_pretrained(cls, pretrained_model_name_or_path, *, config=None,
                                 backbone_load_kwargs=None, **config_kwargs):
        """Initialize a random change head on explicitly loaded DINOv2 weights.

        In contrast, from_pretrained loads a COMPLETE saved scene-change model.
        Ordinary construction and complete-model loading never download a backbone.
        """
        backbone = Dinov2Model.from_pretrained(
            pretrained_model_name_or_path, **(backbone_load_kwargs or {}),
        )
        if config is None:
            config = SceneChangeConfig(backbone_config=backbone.config, **config_kwargs)
        elif config_kwargs:
            raise ValueError("Pass config or config keyword arguments, not both")
        model = cls(config)
        # strict loading prevents quietly accepting a different architecture.
        model.backbone.load_state_dict(backbone.state_dict(), strict=True)
        model.config.backbone_pretrained_source = str(pretrained_model_name_or_path)
        return model

    def _embed_images(self, pixels):
        if self.config.backbone_interpolation == "hf":
            return self.backbone.embeddings(pixels)
        # Reference torch.hub DINOv2 uses bicubic scale_factor with a 0.1
        # offset; modern HF uses an exact target size. Preserve the former
        # for converted checkpoints instead of silently changing features.
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
            patches = patches.flatten(2).transpose(1, 2)
            position = torch.cat((position[:, :1].float(), patches), dim=1).to(tokens.dtype)
        return embeddings.dropout(tokens + position)

    def _extract_features(self, pixels, layer_index, facet):
        """Read normalized patch Q/K/V or post-block tokens without mutable hooks.

        Only blocks through the chosen layer are evaluated. The selected projection
        is called normally, so injected LoRA modules remain in the autograd graph.
        """
        layers = self.backbone.encoder.layer
        layer_index %= len(layers)
        frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if frozen:
            self.backbone.eval()
        with torch.no_grad() if frozen else nullcontext():
            tokens = self._embed_images(pixels)
            for layer in layers[:layer_index]:
                tokens = layer(tokens)
                # Transformers 4.x returns a tuple; 5.x returns a tensor.
                if isinstance(tokens, tuple):
                    tokens = tokens[0]
            layer = layers[layer_index]
            if facet == "token":
                features = layer(tokens)
                if isinstance(features, tuple):
                    features = features[0]
            else:
                projection = getattr(layer.attention.attention, facet)
                features = projection(layer.norm1(tokens))
            return F.normalize(features[:, 1:], dim=-1)

    def forward(self, pixel_values, pixel_values_pair, labels=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        """Inputs: two B,3,H,W tensors; logits: B,num_labels,H,W.

        labels is B,H,W with integer class IDs (255 ignored by default). Logits
        follow config.output_size or, by default, the processed input resolution.
        """
        if kwargs:
            # Trainer versions may pass this loss bookkeeping argument.
            kwargs.pop("num_items_in_batch", None)
            if kwargs:
                raise TypeError(f"Unexpected model arguments: {sorted(kwargs)}")
        if pixel_values.ndim != 4 or pixel_values_pair.shape != pixel_values.shape:
            raise ValueError("Image pairs must have identical B,C,H,W shapes")
        batch, channels, height, width = pixel_values.shape
        patch = self.config.backbone_config.patch_size
        if channels != 3 or height < patch or width < patch or height % patch or width % patch:
            raise ValueError(f"RGB input height and width must be positive multiples of {patch}")
        output_hidden_states = (self.config.output_hidden_states if output_hidden_states is None
                                else output_hidden_states)
        return_dict = self.config.return_dict if return_dict is None else return_dict
        x = self._extract_features(pixel_values, self.config.feature_layer, self.config.facet)
        y = self._extract_features(pixel_values_pair, self.config.feature_layer_pair, self.config.facet_pair)
        hidden_states = (x, y) if output_hidden_states else None
        for interaction in self.interactions:
            if self.gradient_checkpointing and self.training:
                x, y = self._gradient_checkpointing_func(interaction, x, y)
            else:
                x, y = interaction(x, y)
        fused = x if self.config.attention_mode == "single_cross" else torch.cat((x, y), dim=-1)
        fused = fused.transpose(1, 2).reshape(batch, -1, height // patch, width // patch)
        if self.config.attention_mode == "single_cross" and min(fused.shape[-2:]) < 3:
            raise ValueError("single_cross requires at least a 3x3 patch grid")
        logits = self.classifier(F.relu(self.fusion(fused)))
        if self.config.classifier_relu:
            logits = F.relu(logits)
        logits = F.interpolate(logits, size=self.config.output_size or (height, width),
                               mode="bilinear", align_corners=False)
        loss = None
        if labels is not None:
            if labels.ndim != 3 or labels.shape[0] != batch or labels.is_floating_point():
                raise ValueError("labels must be an integer B,H,W tensor of class IDs")
            loss_logits = logits
            if logits.shape[-2:] != labels.shape[-2:]:
                loss_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            weights = (logits.new_tensor(self.config.class_weights)
                       if self.config.class_weights is not None else None)
            labels = labels.to(device=logits.device, dtype=torch.long)
            if torch.any(labels != self.config.loss_ignore_index):
                loss = F.cross_entropy(loss_logits, labels, weight=weights,
                                       ignore_index=self.config.loss_ignore_index)
            else:
                loss = logits.sum() * 0.0
        if not return_dict:
            output = (logits,) + ((hidden_states,) if hidden_states is not None else ())
            return (loss,) + output if loss is not None else output
        return SceneChangeOutput(loss=loss, logits=logits, hidden_states=hidden_states)
