"""Build the training components; let Transformers manage the training loop."""

import torch
from transformers import TrainingArguments, set_seed

from .config import make_datasets, make_model_and_processor
from .trainer import SceneChangeTrainer


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return torch.device(name)


def make_training_arguments(args):
    if args.batch_size < 1 or args.epochs < 1 or args.workers < 0 or args.max_steps < 0:
        raise ValueError("batch-size/epochs must be positive; workers/max-steps cannot be negative")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid learning-rate, weight-decay, or gradient-accumulation-steps")
    if not 0 <= args.hflip_prob <= 1:
        raise ValueError("hflip-prob must be in [0,1]")
    device = resolve_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("fp16/bf16 autocast requires CUDA in this demo")
    return TrainingArguments(
        output_dir=str(args.output_dir), use_cpu=device.type == "cpu",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs, max_steps=args.max_steps or -1,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        optim="adamw_torch", lr_scheduler_type="constant",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=args.amp == "fp16", bf16=args.amp == "bf16",
        dataloader_num_workers=args.workers, dataloader_pin_memory=device.type == "cuda",
        dataloader_persistent_workers=args.workers > 0,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        logging_steps=10, report_to=[], seed=args.seed, data_seed=args.seed,
    )


def train(args):
    """Train/evaluate/save using SceneChangeTrainer, including resumable checkpoints."""
    training_args = make_training_arguments(args)
    set_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, processor = make_model_and_processor(args)
    train_dataset, eval_dataset = make_datasets(args, processor)
    trainer = SceneChangeTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, processing_class=processor,
    )
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
    )
    metrics = trainer.evaluate()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return {"train": result.metrics, "validation": metrics}
