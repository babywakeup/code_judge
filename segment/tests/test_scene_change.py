"""No network, datasets, pretrained weights, or GPU are needed for these tests."""

import importlib.util
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from huggingface_hub.errors import StrictDataclassClassValidationError
import torch
from torch.nn import functional as F
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModelForSemanticSegmentation,
    AutoProcessor, Dinov2Config, Dinov2Model, Trainer, TrainingArguments,
)

from segment import (
    SceneChangeConfig, SceneChangeForSemanticSegmentation, SceneChangePreTrainedModel, SceneChangeProcessor,
)
from segment.models.scene_change.modeling_scene_change import scene_change_loss
from segment.tools.convert_checkpoint import (
    config_from_reference, convert_checkpoint, convert_reference_state_dict, load_reference_weights,
)
from segment.models.scene_change.peft_utils import get_lora_config


def tiny_config(**kwargs):
    options = {"num_attention_heads": 4, "dropout": 0, "classifier_relu": False}
    options.update(kwargs)
    return SceneChangeConfig(
        backbone_config=Dinov2Config(hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=4, image_size=56),
        **options,
    )


def reference_head(model):
    state = {"module.conv1." + k: v.clone() for k, v in model.fusion.state_dict().items()}
    state.update({"module.conv2." + k: v.clone() for k, v in model.classifier.state_dict().items()})
    for index, interaction in enumerate(model.interactions):
        for branch_index in (1,) if interaction.mode == "single_cross" else (1, 2):
            branch = getattr(interaction, f"branch{branch_index}")
            prefix = ("module.ca." if interaction.mode == "single_cross" else
                      f"module.ca0s.{index}.cross_att_{branch_index}.")
            for kind in ("weight", "bias"):
                state[prefix + "multi_head_self_attention.in_proj_" + kind] = torch.cat(
                    [getattr(getattr(branch, projection), kind).detach().clone()
                     for projection in ("q_proj", "k_proj", "v_proj")], dim=0,
                )
            for key, value in branch.state_dict().items():
                if key.startswith(("q_proj.", "k_proj.", "v_proj.")):
                    continue
                state[prefix + ("multi_head_self_attention." if key.startswith("out_proj.") else "") + key] = value.clone()
    return state


class SceneChangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_named_model_family_exports_and_base(self):
        from segment.models import SceneChangeConfig as exported_config
        from segment.models.scene_change import SceneChangeForSemanticSegmentation as exported_model

        self.assertIs(exported_config, SceneChangeConfig)
        self.assertIs(exported_model, SceneChangeForSemanticSegmentation)
        self.assertTrue(issubclass(exported_model, SceneChangePreTrainedModel))
        self.assertEqual(exported_config.model_type, "scene_change")
        self.assertTrue(exported_model.__module__.startswith("segment.models.scene_change."))
        self.assertFalse((Path(__file__).resolve().parents[1] / "model").exists())

    def test_declarative_config_defaults_legacy_dict_and_validation(self):
        config = tiny_config(output_size=(28, 42), class_weights=(1, 2))
        self.assertTrue(is_dataclass(config))
        self.assertIn("backbone_config", {field.name for field in fields(config)})
        self.assertEqual(config.feature_layer_pair, config.feature_layer)
        self.assertEqual(config.decoder_hidden_size, 32)
        self.assertEqual(config.output_size, [28, 42])
        self.assertEqual(config.class_weights, [1, 2])
        self.assertEqual(config.id2label, {0: "unchanged", 1: "changed"})
        legacy = SceneChangeConfig.from_dict({
            "model_type": "scene_change", "backbone_config": config.backbone_config.to_dict(),
            "num_labels": 3, "class_weights": [1, 2, 3], "dropout": 0,
            "backbone_pretrained_source": "local-test-model",
        })
        self.assertEqual(legacy.num_labels, 3)
        self.assertEqual(legacy.backbone_config.hidden_size, 32)
        self.assertEqual(legacy.backbone_pretrained_source, "local-test-model")
        for invalid in ({"dropout": 1}, {"feature_layer": 2}, {"output_size": [0, 28]},
                        {"num_labels": 1}, {"class_weights": [1]}):
            with self.subTest(invalid=invalid), self.assertRaises(StrictDataclassClassValidationError):
                tiny_config(**invalid)

    def test_public_image_features_and_invalid_selection(self):
        model = SceneChangeForSemanticSegmentation(tiny_config()).eval()
        pixels = torch.rand(1, 3, 42, 56)
        with torch.no_grad():
            features = model.get_image_features(pixels)
            output = model(pixels, pixels, output_hidden_states=True)
        torch.testing.assert_close(features, output.hidden_states[0], rtol=0, atol=0)
        self.assertEqual(tuple(features.shape), (1, 12, 32))
        with self.assertRaisesRegex(ValueError, "outside"):
            model.get_image_features(pixels, layer_index=2)
        with self.assertRaisesRegex(ValueError, "facet"):
            model.get_image_features(pixels, facet="invalid")

    def test_loss_helper_matches_weighted_ce_and_ignored_gradients(self):
        logits = torch.randn(2, 3, 14, 21, requires_grad=True)
        labels = torch.randint(0, 3, (2, 28, 42))
        labels[:, 0] = 255
        loss = scene_change_loss(logits, labels, class_weights=[1, 2, 3])
        reference = F.cross_entropy(
            F.interpolate(logits, size=(28, 42), mode="bilinear", align_corners=False),
            labels, weight=logits.new_tensor([1, 2, 3]), ignore_index=255,
        )
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        logits.grad = None
        ignored = scene_change_loss(logits, torch.full_like(labels, 255))
        self.assertEqual(ignored.item(), 0)
        ignored.backward()
        self.assertEqual(logits.grad.count_nonzero().item(), 0)

    def test_model_output_tuple_contract(self):
        model = SceneChangeForSemanticSegmentation(tiny_config()).eval()
        images = (torch.rand(1, 3, 28, 28), torch.rand(1, 3, 28, 28))
        for with_labels in (False, True):
            for hidden in (False, True):
                inputs = {"labels": torch.zeros(1, 28, 28, dtype=torch.long)} if with_labels else {}
                with torch.no_grad():
                    result = model(*images, **inputs, return_dict=True, output_hidden_states=hidden)
                    result_tuple = model(*images, **inputs, return_dict=False, output_hidden_states=hidden,
                                         num_items_in_batch=1)
                torch.testing.assert_close(result.to_tuple(), result_tuple, rtol=0, atol=0)

    def test_forward_loss_and_frozen_backbone(self):
        model = SceneChangeForSemanticSegmentation(tiny_config())
        batch = {"pixel_values": torch.rand(2, 3, 42, 56), "pixel_values_pair": torch.rand(2, 3, 42, 56),
                 "labels": torch.randint(0, 2, (2, 42, 56))}
        output = model(**batch, output_hidden_states=True)
        self.assertEqual(tuple(output.logits.shape), (2, 2, 42, 56))
        self.assertEqual(tuple(output.hidden_states[0].shape), (2, 12, 32))
        output.loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.interactions.parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertFalse(model.backbone.training)
        self.assertEqual(len(model(**batch, return_dict=False)), 2)

    def test_facets_match_backbone_projection_hooks(self):
        model = SceneChangeForSemanticSegmentation(tiny_config()).eval()
        pixels = torch.rand(1, 3, 42, 56)
        layer = model.backbone.encoder.layer[1]
        for facet in ("query", "key", "value", "token"):
            captures = []
            module = layer if facet == "token" else getattr(layer.attention.attention, facet)
            handle = module.register_forward_hook(lambda m, i, o: captures.append(o))
            try:
                with torch.no_grad():
                    model.backbone(pixels)
            finally:
                handle.remove()
            expected = F.normalize(captures[0][:, 1:], dim=-1)
            torch.testing.assert_close(model.get_image_features(pixels, 1, facet), expected)

    def test_save_load_auto_and_no_backbone_download(self):
        model = SceneChangeForSemanticSegmentation(tiny_config()).eval()
        processor = SceneChangeProcessor(size=42)
        inputs = processor(np.zeros((49, 49, 3), np.uint8), np.ones((49, 49, 3), np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            processor.save_pretrained(directory)
            self.assertTrue((Path(directory) / "model.safetensors").exists())
            self.assertIsInstance(AutoConfig.from_pretrained(directory), SceneChangeConfig)
            self.assertIsInstance(AutoProcessor.from_pretrained(directory), SceneChangeProcessor)
            self.assertIsInstance(AutoImageProcessor.from_pretrained(directory), SceneChangeProcessor)
            restored = AutoModelForSemanticSegmentation.from_pretrained(directory, local_files_only=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits, rtol=0, atol=0)
            saved = json.loads((Path(directory) / "config.json").read_text())
            self.assertIn("AutoModelForSemanticSegmentation", saved["auto_map"])

    def test_remote_code_without_importing_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            SceneChangeForSemanticSegmentation(tiny_config()).save_pretrained(directory)
            SceneChangeProcessor().save_pretrained(directory)
            code = """
import sys
from transformers import AutoModelForSemanticSegmentation, AutoProcessor
assert 'segment' not in sys.modules
model = AutoModelForSemanticSegmentation.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
processor = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
assert model.config.model_type == 'scene_change'
assert processor.model_input_names == ['pixel_values', 'pixel_values_pair']
"""
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-c", code, directory], cwd=directory,
                                    env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_config_round_trip_multiclass(self):
        config = tiny_config(num_labels=3, class_weights=[1, 2, 3], output_size=[28, 42])
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = SceneChangeConfig.from_pretrained(directory)
        self.assertEqual(restored.num_labels, 3)
        self.assertEqual(restored.class_weights, [1, 2, 3])
        self.assertIsInstance(restored.backbone_config, Dinov2Config)
        self.assertEqual(restored.backbone_config.hidden_size, 32)

    def test_processor_crop_mask_and_rescaling(self):
        processor = SceneChangeProcessor(mask_value_map={0: 0, 255: 1})
        image = np.full((45, 59, 3), 255, dtype=np.uint8)
        mask = np.zeros((45, 59), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        batch = processor(image, image, segmentation_maps=mask)
        self.assertEqual(tuple(batch.pixel_values.shape), (1, 3, 42, 56))
        self.assertEqual(batch.pixel_values.max().item(), 1)
        self.assertEqual(batch.labels.dtype, torch.long)
        expected = torch.from_numpy(mask[2:44, 2:58] == 255).long()
        torch.testing.assert_close(batch.labels[0], expected)
        for singleton_mask in (mask[..., None], mask[None]):
            torch.testing.assert_close(processor(image, image, segmentation_maps=singleton_mask).labels, batch.labels)
        floats = processor(torch.ones(3, 42, 56), torch.ones(3, 42, 56))
        self.assertEqual(floats.pixel_values.min().item(), 1)
        as_numpy = processor(image, image, return_tensors="np")
        self.assertIsInstance(as_numpy.pixel_values, np.ndarray)
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            loaded = SceneChangeProcessor.from_pretrained(directory)
            self.assertEqual(loaded.mask_value_map, {0: 0, 255: 1})
            torch.testing.assert_close(loaded(image, image, segmentation_maps=mask).labels, batch.labels)

    def test_processor_padding_and_batch_validation(self):
        processor = SceneChangeProcessor(size=56)
        images = torch.ones(2, 3, 28, 42)
        masks = torch.zeros(2, 28, 42, dtype=torch.long)
        data = processor(images, images, segmentation_maps=masks)
        self.assertEqual(tuple(data.labels.shape), (2, 56, 56))
        self.assertEqual(data.labels[0, 0, 0].item(), 255)
        self.assertEqual(data.labels[0, 20, 20].item(), 0)
        with self.assertRaises(ValueError):
            processor(images, images[:1])
        with self.assertRaises(ValueError):
            SceneChangeProcessor()(list(images) + [torch.ones(3, 42, 42)],
                                   list(images) + [torch.ones(3, 42, 42)])

    def test_postprocess_and_ignored_loss(self):
        model = SceneChangeForSemanticSegmentation(tiny_config(output_size=[28, 28]))
        inputs = (torch.rand(1, 3, 42, 56), torch.rand(1, 3, 42, 56))
        output = model(*inputs, labels=torch.full((1, 42, 56), 255))
        self.assertEqual(output.loss.item(), 0)
        output.loss.backward()
        maps = SceneChangeProcessor().post_process_semantic_segmentation(output, [(51, 67)])
        self.assertEqual(tuple(maps[0].shape), (51, 67))

    def test_gradient_checkpointing_and_unfreezing(self):
        model = SceneChangeForSemanticSegmentation(tiny_config(freeze_backbone=False, facet="token", facet_pair="token"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        output = model(torch.rand(1, 3, 42, 42), torch.rand(1, 3, 42, 42), labels=torch.ones(1, 42, 42, dtype=torch.long))
        output.loss.backward()
        self.assertIsNotNone(model.get_input_embeddings().weight.grad)
        model.gradient_checkpointing_disable()
        model.set_backbone_trainable(False, last_n_layers=1)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.encoder.layer[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.encoder.layer[1].parameters()))

    def test_reference_attention_equivalence(self):
        reference_file = Path(__file__).resolve().parents[2] / "参考资料/Robust-Scene-Change-Detection/src/robust_scene_change_detect/models/transformer.py"
        if not reference_file.exists():
            self.skipTest("Reference checkout is not available")
        spec = importlib.util.spec_from_file_location("reference_transformer", reference_file)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        model = SceneChangeForSemanticSegmentation(tiny_config(attention_mode="legacy_self", num_attention_blocks=2)).eval()
        state = reference_head(model)
        inputs = (torch.rand(1, 3, 42, 56), torch.rand(1, 3, 42, 56))
        with torch.no_grad():
            x = model.get_image_features(inputs[0], -1, "query")
            y = model.get_image_features(inputs[1], -1, "query")
            for index in range(2):
                block = reference.TwoCrossAttention(32, 4, 0, 32, 4, 0).eval()
                prefix = f"module.ca0s.{index}."
                block.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
                x, y = block(x, y, y, y, x, x)
            features = torch.cat((x, y), dim=-1).transpose(1, 2).reshape(1, 64, 3, 4)
            expected = model.classifier(F.relu(model.fusion(features)))
            expected = F.interpolate(expected, size=(42, 56), mode="bilinear", align_corners=False)
            torch.testing.assert_close(model(*inputs).logits, expected, rtol=1e-5, atol=1e-6)

    def test_reference_mapping_strictness(self):
        model = SceneChangeForSemanticSegmentation(tiny_config(attention_mode="legacy_self"))
        state = reference_head(model)
        with self.assertRaisesRegex(ValueError, "frozen backbone"):
            load_reference_weights(model, state)
        clone = SceneChangeForSemanticSegmentation(tiny_config(attention_mode="legacy_self"))
        clone.backbone.load_state_dict(model.backbone.state_dict())
        load_reference_weights(clone, state, backbone_is_pretrained=True)
        for key in model.state_dict():
            torch.testing.assert_close(model.state_dict()[key], clone.state_dict()[key])
        partial = dict(state)
        partial.pop("module.conv2.bias")
        with self.assertRaises(ValueError):
            load_reference_weights(clone, partial, backbone_is_pretrained=True)
        state["module.unsupported.weight"] = torch.ones(1)
        with self.assertRaises(ValueError):
            convert_reference_state_dict(state)

    def test_reference_backbone_mapping(self):
        qkv = torch.randn(96, 32)
        state = {"module.backbone.dino_1.dino.blocks.0.attn.qkv.weight": qkv,
                 "module.backbone.dino_2.dino.blocks.0.attn.qkv.weight": qkv.clone(),
                 "module.backbone.dino_1.dino.blocks.0.ls1.gamma": torch.ones(32),
                 "module.backbone.dino_1.dino.pos_embed": torch.ones(1, 17, 32)}
        mapped = convert_reference_state_dict(state)
        torch.testing.assert_close(mapped["backbone.encoder.layer.0.attention.attention.query.weight"], qkv[:32])
        self.assertIn("backbone.encoder.layer.0.layer_scale1.lambda1", mapped)
        self.assertIn("backbone.embeddings.position_embeddings", mapped)
        state["module.backbone.dino_2.dino.blocks.0.attn.qkv.weight"] = qkv + 1
        with self.assertRaisesRegex(ValueError, "different shared weights"):
            convert_reference_state_dict(state)
        config = config_from_reference({"name": "dino2 + cross_attention", "layer1": 1}, tiny_config().backbone_config)
        self.assertEqual(config.attention_mode, "legacy_self")
        self.assertEqual(config.backbone_interpolation, "reference")

    def test_full_reference_backbone_state_round_trip(self):
        model = SceneChangeForSemanticSegmentation(tiny_config(attention_mode="legacy_self"))
        state = reference_head(model)
        direct = {
            "embeddings.cls_token": "cls_token", "embeddings.mask_token": "mask_token",
            "embeddings.position_embeddings": "pos_embed",
            "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
            "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
            "layernorm.weight": "norm.weight", "layernorm.bias": "norm.bias",
        }
        for key, value in model.backbone.state_dict().items():
            if key in direct:
                state["module.backbone.dino_1.dino." + direct[key]] = value.clone()
        for index, layer in enumerate(model.backbone.encoder.layer):
            prefix = f"module.backbone.dino_1.dino.blocks.{index}."
            for kind in ("weight", "bias"):
                state[prefix + "attn.qkv." + kind] = torch.cat([
                    getattr(getattr(layer.attention.attention, facet), kind).detach().clone()
                    for facet in ("query", "key", "value")
                ])
            for key, value in layer.state_dict().items():
                if key.startswith("attention.attention."):
                    continue
                for new, old in (("attention.output.dense.", "attn.proj."),
                                 ("layer_scale1.lambda1", "ls1.gamma"),
                                 ("layer_scale2.lambda1", "ls2.gamma")):
                    if key.startswith(new):
                        key = old + key[len(new):]
                        break
                state[prefix + key] = value.clone()
        clone = SceneChangeForSemanticSegmentation(tiny_config(attention_mode="legacy_self"))
        self.assertEqual(load_reference_weights(clone, state), [])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)

    def test_reference_checkpoint_conversion_end_to_end(self):
        options = {"name": "dino2 + cross_attention", "dino-model": "dinov2_vits14",
                   "layer1": 1, "num-heads": 1, "num-blocks": 1,
                   "dropout-rate": 0, "target-shp-row": 42, "target-shp-col": 56}
        config = config_from_reference(options)
        model = SceneChangeForSemanticSegmentation(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            backbone_path, output_path = directory / "backbone", directory / "converted"
            model.backbone.save_pretrained(backbone_path)
            checkpoint_path = directory / "reference.pth"
            torch.save({"model": reference_head(model),
                        "args": {"model": options, "optimizer": {"loss-weight": True}}}, checkpoint_path)
            converted, processor = convert_checkpoint(
                checkpoint_path, output_path, backbone_source=backbone_path, local_files_only=True,
            )
            self.assertEqual(converted.config.class_weights, [0.025, 0.975])
            self.assertEqual(processor.size, {"height": 42, "width": 56})
            inputs = (torch.rand(1, 3, 42, 56), torch.rand(1, 3, 42, 56))
            with torch.no_grad():
                torch.testing.assert_close(model(*inputs).logits, converted.eval()(*inputs).logits, rtol=0, atol=0)
            restored = SceneChangeForSemanticSegmentation.from_pretrained(output_path, local_files_only=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(converted(*inputs).logits, restored(*inputs).logits, rtol=0, atol=0)

    def test_lora_targets_only_reachable_backbone_projections(self):
        model = SceneChangeForSemanticSegmentation(tiny_config(feature_layer=0, feature_layer_pair=1))
        config = get_lora_config(model, adapt_backbone=True)
        self.assertIn("backbone.encoder.layer.0.attention.attention.value", config.target_modules)
        self.assertIn("backbone.encoder.layer.1.attention.attention.query", config.target_modules)
        self.assertNotIn("backbone.encoder.layer.1.attention.attention.value", config.target_modules)
        self.assertIsNone(config.task_type)
        self.assertEqual(config.modules_to_save, ["fusion", "classifier"])

    def test_invalid_model_inputs_and_config(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "divisible"):
            tiny_config(num_attention_heads=3)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "facets"):
            tiny_config(facet="invalid")
        model = SceneChangeForSemanticSegmentation(tiny_config())
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 29, 28), torch.rand(1, 3, 29, 28))
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 28, 28), torch.rand(1, 3, 28, 28), labels=torch.zeros(1, 28, 28))

    def test_local_pretrained_backbone_and_single_cross(self):
        config = tiny_config(attention_mode="single_cross")
        with tempfile.TemporaryDirectory() as directory:
            backbone = Dinov2Model(config.backbone_config)
            backbone.save_pretrained(directory)
            model = SceneChangeForSemanticSegmentation.from_backbone_pretrained(
                directory, config=config, backbone_load_kwargs={"local_files_only": True},
            )
        torch.testing.assert_close(model.backbone.get_input_embeddings().projection.weight,
                                   backbone.get_input_embeddings().projection.weight)
        self.assertEqual(tuple(model(torch.rand(1, 3, 42, 42), torch.rand(1, 3, 42, 42)).logits.shape), (1, 2, 42, 42))
        load_reference_weights(model, reference_head(model), backbone_is_pretrained=True)

    def test_peft_adapter_backward_save_reload_merge(self):
        from peft import PeftModel, get_peft_model
        base = SceneChangeForSemanticSegmentation(tiny_config(facet="token", facet_pair="token"))
        inputs = {"pixel_values": torch.rand(1, 3, 42, 42), "pixel_values_pair": torch.rand(1, 3, 42, 42),
                  "labels": torch.randint(0, 2, (1, 42, 42))}
        with tempfile.TemporaryDirectory() as directory:
            base_path, adapter_path = Path(directory) / "base", Path(directory) / "adapter"
            base.save_pretrained(base_path)
            wrapped = get_peft_model(base, get_lora_config(base, r=2, lora_alpha=4, adapt_backbone=True))
            wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            wrapped(**inputs).loss.backward()
            for prefix in ("backbone.", "interactions."):
                grads = [p.grad for name, p in wrapped.named_parameters() if prefix in name and "lora_B" in name]
                self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), prefix)
            optimizer = torch.optim.SGD([p for p in wrapped.parameters() if p.requires_grad], lr=0.01)
            optimizer.step()
            wrapped.eval()
            with torch.no_grad():
                expected = wrapped(**inputs).logits
            wrapped.save_pretrained(adapter_path)
            restored = PeftModel.from_pretrained(SceneChangeForSemanticSegmentation.from_pretrained(base_path), adapter_path).eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(**inputs).logits)
                merged = restored.merge_and_unload().eval()
                torch.testing.assert_close(expected, merged(**inputs).logits, rtol=1e-4, atol=1e-6)

    def test_transformers_trainer_cpu_step(self):
        from peft import get_peft_model
        model = SceneChangeForSemanticSegmentation(tiny_config())
        model = get_peft_model(model, get_lora_config(model, r=2))
        examples = [{"pixel_values": torch.rand(3, 42, 42), "pixel_values_pair": torch.rand(3, 42, 42),
                     "labels": torch.zeros(42, 42, dtype=torch.long)} for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                                     per_device_train_batch_size=2, report_to=[], save_strategy="no",
                                     disable_tqdm=True, label_names=["labels"])
            trainer = Trainer(model=model, args=args, train_dataset=examples)
            result = trainer.train()
            self.assertEqual(result.global_step, 1)
            self.assertTrue(np.isfinite(result.training_loss))


if __name__ == "__main__":
    unittest.main()
