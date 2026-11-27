"""Offline Trainer, CLI layout, metrics and resumable training regressions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import Dinov2Config, EvalPrediction, Trainer, TrainingArguments

from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation, SceneChangeProcessor
from segment.demo import main
from segment.models.assets import resolve_local_backbone
from segment.models.scene_change.peft_utils import get_lora_config
from segment.training import SceneChangeTrainer
from segment.training.config import make_model_and_processor, parse_args
from segment.training.trainer import SegmentationMetrics, prediction_masks


def tiny_model(**kwargs):
    return SceneChangeForSemanticSegmentation(SceneChangeConfig(
        backbone_config=Dinov2Config(hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=4, image_size=56),
        classifier_relu=False, dropout=0, **kwargs,
    ))


def examples():
    samples = []
    for _ in range(3):
        labels = torch.zeros(28, 28, dtype=torch.long)
        labels[7:21, 7:21] = 1
        labels[0] = 255
        samples.append({"pixel_values": torch.rand(3, 28, 28),
                        "pixel_values_pair": torch.rand(3, 28, 28), "labels": labels})
    return samples


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_root_has_only_public_entry_files(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual({p.name for p in root.glob("*.py")}, {"__init__.py", "demo.py"})

    def test_local_asset_resolution_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "facebook/dinov2-small"
            local.mkdir(parents=True)
            with patch("segment.models.assets.HF_CACHE_ROOT", root), patch("segment.models.assets.REPO_ROOT", root):
                self.assertEqual(resolve_local_backbone("facebook/dinov2-small"), local)
                self.assertEqual(resolve_local_backbone("dinov2_vits14"), local)
                self.assertEqual(resolve_local_backbone(local), local)
                self.assertEqual(resolve_local_backbone("./facebook/dinov2-small"), local)
                with self.assertRaises(FileNotFoundError):
                    resolve_local_backbone("facebook/missing")

    def test_smoke_does_not_require_cached_backbone(self):
        with patch("segment.training.config.resolve_local_backbone", side_effect=AssertionError("Must stay offline")):
            output = main(["--smoke", "--device", "cpu"])
        self.assertEqual(tuple(output.logits.shape), (1, 2, 56, 56))
        self.assertTrue(torch.isfinite(output.loss))

    def test_streamed_metrics_ignore_padding_and_reset(self):
        metric = SegmentationMetrics(2)
        self.assertEqual(metric(EvalPrediction(
            predictions=torch.tensor([0, 1, 0, 1, 0]), label_ids=torch.tensor([0, 1, 1, 255, -100]),
        )), {})
        result = metric(EvalPrediction(predictions=np.array([1, 1]), label_ids=np.array([0, 1])),
                        compute_result=True)
        self.assertAlmostEqual(result["foreground_miou"], 0.5)
        self.assertAlmostEqual(result["pixel_accuracy"], 0.6)
        self.assertFalse(metric.confusion.any())
        ignored = metric(EvalPrediction(predictions=np.array([1]), label_ids=np.array([255])),
                         compute_result=True)
        self.assertEqual(ignored, {"foreground_miou": 0.0, "pixel_accuracy": 0.0})

    def test_prediction_masks_resize_logits_and_drop_extra_outputs(self):
        logits = torch.zeros(1, 2, 14, 14)
        logits[:, 1] = 1
        masks = prediction_masks((logits, torch.ones(1)), torch.zeros(1, 28, 28, dtype=torch.long))
        self.assertEqual(tuple(masks.shape), (1, 28, 28))
        self.assertTrue(masks.eq(1).all())

    def test_subclass_train_evaluate_and_predict(self):
        self.assertTrue(issubclass(SceneChangeTrainer, Trainer))
        model = tiny_model(output_size=[14, 14], output_hidden_states=True)
        samples = examples()
        before = model.classifier.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                                     per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                     gradient_accumulation_steps=2, gradient_checkpointing=True,
                                     gradient_checkpointing_kwargs={"use_reentrant": False},
                                     report_to=[], save_strategy="no", disable_tqdm=True)
            trainer = SceneChangeTrainer(model=model, args=args, train_dataset=samples, eval_dataset=samples)
            self.assertEqual(trainer.label_names, ["labels"])
            self.assertTrue(trainer.args.batch_eval_metrics)
            result = trainer.train()
            self.assertEqual(result.global_step, 1)
            self.assertFalse(torch.equal(before, model.classifier.weight))
            first = trainer.evaluate()
            second = trainer.evaluate()
            self.assertAlmostEqual(first["eval_loss"], second["eval_loss"])
            self.assertAlmostEqual(first["eval_foreground_miou"], second["eval_foreground_miou"])
            prediction = trainer.predict(samples)
            self.assertEqual(prediction.predictions.shape, (3, 28, 28))
            self.assertIn("test_pixel_accuracy", prediction.metrics)

    def test_peft_trainer_keeps_both_images_and_labels(self):
        from peft import get_peft_model

        model = tiny_model()
        model = get_peft_model(model, get_lora_config(model, r=2))
        samples = examples()
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                                     report_to=[], save_strategy="no", disable_tqdm=True)
            trainer = SceneChangeTrainer(model=model, args=args, train_dataset=samples, eval_dataset=samples)
            self.assertEqual(trainer.train().global_step, 1)
            self.assertTrue(np.isfinite(trainer.evaluate()["eval_loss"]))

    def test_nonfinite_loss_fails_loudly(self):
        model = tiny_model()
        with torch.no_grad():
            model.classifier.weight.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[])
            trainer = SceneChangeTrainer(model=model, args=args)
            batch = {key: value[None] for key, value in examples()[0].items()}
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                trainer.compute_loss(model, batch)

    def test_reloaded_processor_keeps_settings_with_new_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            tiny_model().save_pretrained(directory)
            SceneChangeProcessor(do_normalize=True, mask_value_map={0: 0, 255: 1}).save_pretrained(directory)
            args = parse_args(["--init-model", directory, "--image-size", "28"])
            _, processor = make_model_and_processor(args)
            self.assertEqual(processor.size, {"height": 28, "width": 28})
            self.assertTrue(processor.do_normalize)
            self.assertEqual(processor.mask_value_map, {0: 0, 255: 1})

    def test_reload_preserves_optimizer_parameter_groups(self):
        for frozen, last_n in ((True, 0), (True, 1), (False, 0)):
            with self.subTest(frozen=frozen, last_n=last_n), tempfile.TemporaryDirectory() as directory:
                model = tiny_model(freeze_backbone=frozen, unfreeze_backbone_last_n_layers=last_n)
                expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                model.save_pretrained(directory)
                restored, _ = SceneChangeForSemanticSegmentation.from_pretrained(
                    directory, local_files_only=True, output_loading_info=True,
                )
                actual = {name for name, parameter in restored.named_parameters() if parameter.requires_grad}
                self.assertEqual(expected, actual)

    def test_demo_trains_saves_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "data", Path(directory) / "output"
            for split, name in (("train", "001_1_00_0.png"), ("train", "013_1_00_0.png"),
                                ("test", "000_1_00_0.png")):
                for folder in ("t0", "t1", "mask"):
                    target = root / split / folder
                    target.mkdir(parents=True, exist_ok=True)
                    shape = (28, 28) if folder == "mask" else (28, 28, 3)
                    array = np.zeros(shape, dtype=np.uint8)
                    array[7:21, 7:21] = 255
                    Image.fromarray(array).save(target / name)
            argv = ["--random-backbone", "--device", "cpu", "--batch-size", "1",
                    "--data-root", str(root), "--output-dir", str(output)]
            with patch("segment.training.config.resolve_local_backbone", side_effect=AssertionError("No download")):
                result = main([*argv, "--max-steps", "1"])
                self.assertTrue(np.isfinite(result["validation"]["eval_loss"]))
                for name in ("model.safetensors", "preprocessor_config.json", "trainer_state.json",
                             "train_results.json", "eval_results.json", "checkpoint-1/optimizer.pt"):
                    self.assertTrue((output / name).is_file(), name)
                self.assertEqual(json.loads((output / "trainer_state.json").read_text())["global_step"], 1)
                # Resume is self-contained: it must not reload the original foundation model.
                resume_argv = [arg for arg in argv if arg != "--random-backbone"]
                main([*resume_argv, "--max-steps", "2", "--resume-from-checkpoint", str(output / "checkpoint-1")])
                self.assertEqual(json.loads((output / "trainer_state.json").read_text())["global_step"], 2)
                self.assertTrue((output / "checkpoint-2/scheduler.pt").is_file())
            restored = SceneChangeForSemanticSegmentation.from_pretrained(output, local_files_only=True)
            self.assertEqual(restored.config.backbone_config.hidden_size, 64)


if __name__ == "__main__":
    unittest.main()
