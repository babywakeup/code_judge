"""Tiny offline backbone tests; no cached weights or GPU are required."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub.errors import StrictDataclassClassValidationError
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForSemanticSegmentation, DINOv3ViTConfig, Dinov2Config

from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation
from segment.models.scene_change.peft_utils import get_lora_config


def tiny_backbone(kind):
    options = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4)
    if kind == "dinov2":
        return Dinov2Config(image_size=56, **options)
    return DINOv3ViTConfig(image_size=64, patch_size=16, intermediate_size=64,
                         num_register_tokens=4, pos_embed_rescale=None, **options)


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_config_initialization_switch_and_exact_pretrained_weights(self):
        for kind in ("dinov2", "dinov3_vit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as source:
                native = AutoModel.from_config(tiny_backbone(kind))
                native.save_pretrained(source)
                config = SceneChangeConfig(backbone_name_or_path=source, use_pretrained_backbone=True)
                # Exercise the standard HF Auto entry point as well as our public factory.
                model = AutoModelForSemanticSegmentation.from_config(config)
                self.assertEqual(config.backbone_config.model_type, kind)
                for name, value in native.state_dict().items():
                    torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)
                config.use_pretrained_backbone = False
                with patch.object(AutoModel, "from_pretrained", side_effect=AssertionError("No weight loading")):
                    random_model = SceneChangeForSemanticSegmentation.from_config(config)
                self.assertFalse(torch.equal(model.get_input_embeddings().weight,
                                             random_model.get_input_embeddings().weight))

    def test_complete_checkpoint_is_independent_of_original_backbone(self):
        for kind in ("dinov2", "dinov3_vit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                source, task = Path(directory) / "backbone", Path(directory) / "task"
                AutoModel.from_config(tiny_backbone(kind)).save_pretrained(source)
                config = SceneChangeConfig(backbone_name_or_path=source, use_pretrained_backbone=True, dropout=0)
                model = SceneChangeForSemanticSegmentation.from_config(config).eval()
                model.save_pretrained(task)
                shutil.rmtree(source)
                with patch.object(AutoModel, "from_pretrained", side_effect=AssertionError("No external weights")):
                    restored = SceneChangeForSemanticSegmentation.from_pretrained(task, local_files_only=True).eval()
                self.assertTrue(restored.config.use_pretrained_backbone)
                self.assertEqual(restored.config.backbone_name_or_path, str(source))
                patch_size = config.backbone_config.patch_size
                images = torch.randn(1, 3, patch_size * 3, patch_size * 4)
                with torch.no_grad():
                    torch.testing.assert_close(model(images, images).logits, restored(images, images).logits,
                                               rtol=0, atol=0)

    def test_dinov3_facets_match_native_hooks_and_exclude_register_tokens(self):
        config = SceneChangeConfig(backbone_config=tiny_backbone("dinov3_vit"), dropout=0)
        model = SceneChangeForSemanticSegmentation(config).eval()
        images = torch.randn(1, 3, 48, 64)
        layer = model.backbone_layers[1]
        for facet in ("query", "key", "value", "token"):
            with self.subTest(facet=facet):
                projection = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}
                module = layer if facet == "token" else getattr(layer.attention, projection[facet])
                captures = []
                handle = module.register_forward_hook(lambda m, i, output: captures.append(output))
                try:
                    with torch.no_grad():
                        model.backbone(images)
                finally:
                    handle.remove()
                actual = model.get_image_features(images, 1, facet)
                self.assertEqual(tuple(actual.shape), (1, 12, 32))
                torch.testing.assert_close(actual, F.normalize(captures[0][:, 5:], dim=-1), rtol=0, atol=0)

    def test_dinov3_lora_gradients_checkpointing_and_reachable_targets(self):
        from peft import get_peft_model

        model = SceneChangeForSemanticSegmentation(SceneChangeConfig(
            backbone_config=tiny_backbone("dinov3_vit"), feature_layer=0, feature_layer_pair=1,
            facet="query", facet_pair="query", classifier_relu=False, dropout=0,
        ))
        lora = get_lora_config(model, adapt_backbone=True, r=2)
        self.assertIn("backbone.model.layer.0.attention.v_proj", lora.target_modules)
        self.assertIn("backbone.model.layer.1.attention.q_proj", lora.target_modules)
        self.assertNotIn("backbone.model.layer.1.attention.v_proj", lora.target_modules)
        model = get_peft_model(model, lora)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        images = torch.randn(1, 3, 48, 64)
        output = model(images, images, labels=torch.zeros(1, 48, 64, dtype=torch.long))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for prefix in ("backbone.", "interactions."):
            gradients = [value.grad for name, value in model.named_parameters()
                         if prefix in name and "lora_B" in name]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients), prefix)

    def test_dinov3_freezing_policy_survives_reload(self):
        for options in ({}, {"unfreeze_backbone_last_n_layers": 1}, {"freeze_backbone": False}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                model = SceneChangeForSemanticSegmentation(SceneChangeConfig(
                    backbone_config=tiny_backbone("dinov3_vit"), **options,
                ))
                policy = {name: p.requires_grad for name, p in model.named_parameters()}
                model.save_pretrained(directory)
                restored = SceneChangeForSemanticSegmentation.from_pretrained(directory, local_files_only=True)
                self.assertEqual(policy, {name: p.requires_grad for name, p in restored.named_parameters()})

    def test_dinov3_bfloat16_matches_native_projection_with_float_inputs(self):
        model = SceneChangeForSemanticSegmentation(SceneChangeConfig(
            backbone_config=tiny_backbone("dinov3_vit"), dropout=0,
        )).to(dtype=torch.bfloat16).eval()
        images = torch.randn(1, 3, 48, 64)
        captures = []
        handle = model.backbone_layers[1].attention.q_proj.register_forward_hook(
            lambda m, i, output: captures.append(output),
        )
        try:
            with torch.no_grad():
                model.backbone(images)
        finally:
            handle.remove()
        features = model.get_image_features(images, 1, "query")
        self.assertEqual(features.dtype, torch.bfloat16)
        torch.testing.assert_close(features, F.normalize(captures[0][:, 5:], dim=-1), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(model(images, images).logits).all())

    def test_invalid_backbone_policy_and_unsupported_architectures(self):
        for options in ({"use_pretrained_backbone": True},
                        {"backbone_config": {"model_type": "vit"}},
                        {"backbone_config": tiny_backbone("dinov3_vit"), "backbone_interpolation": "reference"}):
            with self.subTest(options=options), self.assertRaises(StrictDataclassClassValidationError):
                SceneChangeConfig(**options)

    def test_incomplete_pretrained_backbone_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            native = AutoModel.from_config(tiny_backbone("dinov2"))
            native.save_pretrained(directory)
            incompatible = tiny_backbone("dinov2")
            incompatible.num_hidden_layers = 1
            config = SceneChangeConfig(backbone_config=incompatible, backbone_name_or_path=directory,
                                       use_pretrained_backbone=True)
            with self.assertRaisesRegex(ValueError, "Incomplete pretrained backbone"):
                SceneChangeForSemanticSegmentation.from_config(config)

    def test_dinov2_all_standard_sizes_can_be_built_on_meta_device(self):
        for size, dim, depth, heads in (("small", 384, 12, 6), ("base", 768, 12, 12),
                                        ("large", 1024, 24, 16), ("giant", 1536, 40, 24)):
            with self.subTest(size=size), torch.device("meta"):
                backbone = Dinov2Config(hidden_size=dim, num_hidden_layers=depth, num_attention_heads=heads,
                                         use_swiglu_ffn=size == "giant")
                model = SceneChangeForSemanticSegmentation.from_config(SceneChangeConfig(backbone_config=backbone))
                self.assertEqual(model.fusion.in_channels, dim * 2)
                self.assertEqual(len(model.backbone_layers), depth)
                self.assertEqual(model.get_input_embeddings().weight.device.type, "meta")

    def test_dinov3_remote_code_without_segment_or_backbone_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source, task = Path(directory) / "backbone", Path(directory) / "task"
            AutoModel.from_config(tiny_backbone("dinov3_vit")).save_pretrained(source)
            model = SceneChangeForSemanticSegmentation.from_config(SceneChangeConfig(
                backbone_name_or_path=source, use_pretrained_backbone=True,
            ))
            model.save_pretrained(task)
            shutil.rmtree(source)
            code = """
import sys
import torch
from transformers import AutoModelForSemanticSegmentation
assert 'segment' not in sys.modules
model = AutoModelForSemanticSegmentation.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
assert model.config.backbone_config.model_type == 'dinov3_vit'
assert model.config.use_pretrained_backbone
assert model.get_image_features(torch.randn(1, 3, 48, 64)).shape == (1, 12, 32)
"""
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-c", code, str(task)], cwd=task, env=env,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
