# official_train.py
# -*- coding: utf-8 -*-
"""
Official LoRA fine-tuning script for sequence regression/ranking with 🤗 Transformers Trainer.
- Uses only `transformers.Trainer` (built-in Accelerate) to safely support multi-GPU training.
- Supports LoRA + (optional) k-bit loading via bitsandbytes.
- Provides multiple loss functions (mse, l1, smoothl1, bce, focalmse).
- Computes MSE/MAE/NDCG@k metrics; saves the best model by lowest MSE.
- Clean, open-source friendly with clear comments.

Recommended multi-GPU launch (DDP):
  torchrun --nproc_per_node=8 official_train.py --checkpoint <hf-model> --data_path <train.csv> --test_data_path <test.csv>

Or with Accelerate launcher:
  accelerate launch --num_processes 8 official_train.py --checkpoint <hf-model> --data_path <train.csv> --test_data_path <test.csv>

Notes:
- DO NOT call `accelerator.prepare(...)` yourself. Trainer already handles Accelerate.
- Avoid passing a `device_map` for DDP replicas (let Trainer place modules on each GPU).
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split

from sklearn.metrics import ndcg_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Project-specific dataset. Expected to return dicts with "input_ids", "attention_mask", "labels"
from NAIDv1.dataset import TextDataset

# Make tokenizers deterministic & avoid tokenizer multithreading overhead spam
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------
# Metrics
# ---------------------------
def ndcg_at_k(predictions, labels, k: int = 20) -> float:
    """
    Compute NDCG@k for a single list of predictions/labels.
    If len(predictions) < k, returns -1.0 as sentinel.
    """
    if len(predictions) < k:
        return -1.0
    # sklearn expects shape (n_samples, n_labels); we feed as single-sample
    return ndcg_score([labels], [predictions], k=k)


def compute_metrics(eval_pred):
    """
    eval_pred: Tuple(np.ndarray predictions, np.ndarray labels) from Trainer.
    We convert to torch tensors for loss calcs, then back to numpy for NDCG.
    """
    preds, labels = eval_pred
    preds_t = torch.as_tensor(preds).squeeze()
    labels_t = torch.as_tensor(labels).squeeze()

    mse = nn.MSELoss()(preds_t, labels_t).item()
    mae = nn.L1Loss()(preds_t, labels_t).item()

    ndcg = ndcg_at_k(
        preds_t.detach().cpu().numpy(),
        labels_t.detach().cpu().numpy(),
        k=20,
    )
    return {"mse": mse, "mae": mae, "ndcg": ndcg}


# ---------------------------
# Losses
# ---------------------------
class FocalMSELoss(nn.Module):
    """
    Template.
    A simple focal MSE variant:
    loss = (|y - y_hat|^gamma) * (y - y_hat)^2
    """
    def __init__(self, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        base = diff.pow(2)
        mod = diff.abs().pow(self.gamma)
        loss = base * mod
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss_fn(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smoothl1":
        return nn.SmoothL1Loss()
    if name == "bce":
        # For BCE we assume labels in [0,1] and model outputs raw logits -> use BCEWithLogitsLoss
        return nn.BCEWithLogitsLoss()
    if name == "focalmse":
        return FocalMSELoss(gamma=1.0)
    raise ValueError(f"Unknown loss_func: {name}")


# ---------------------------
# Custom Trainer (to support custom loss)
# ---------------------------
class RegressionTrainer(Trainer):
    """
    Override compute_loss to support custom losses (mse/l1/smoothl1/bce/focalmse).
    - If num_labels == 1 and loss is mse/l1/smoothl1/focalmse: we assume a regression task and
      model outputs logits of shape (batch, 1). We'll squeeze to (batch,).
    - For bce: we'll also squeeze and apply BCEWithLogits on raw logits.
    """

    def __init__(self, *args, loss_func: str = "mse", **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_name = loss_func
        self._loss_fn = build_loss_fn(loss_func)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        # Squeeze last dim for single-label regression/binary
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        if self._loss_name == "bce":
            # BCEWithLogits expects float targets in {0,1}
            loss = self._loss_fn(logits, labels.float())
        else:
            loss = self._loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Utils
# ---------------------------
def save_args_to_json(args: argparse.Namespace, file_path: str) -> None:
    args_dict = vars(args)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)


def build_lora_model(
    base_model: nn.Module,
    load_in_8bit: bool,
    target_modules: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> nn.Module:
    """
    Wrap base model with PEFT-LoRA. If using k-bit (8-bit) loading, call prepare_model_for_kbit_training first.
    """
    # Prepare for k-bit if requested
    if load_in_8bit:
        base_model = prepare_model_for_kbit_training(base_model)

    if target_modules:
        target_list = [m.strip() for m in target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_list,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )
    else:
        # Apply to default modules inferred by PEFT
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------
# Main
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for sequence regression/ranking using 🤗 Trainer (multi-GPU safe)."
    )

    # Core paths
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Hugging Face model id or local path")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Training CSV path")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Held-out test CSV path (not used for param search)")
    parser.add_argument("--runs_dir", type=str, default=None,
                        help="Output dir for logs and checkpoints (default: ./official_runs/<timestamp>)")

    # Training config
    parser.add_argument("--total_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=5e-5,
                        help="Used only if --learning_rate is not set: lr = base_lr * (effective_batch/256)")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if supported")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if supported")

    # Task / labels
    parser.add_argument("--num_labels", type=int, default=1,
                        help="For regression use 1. For BCE you still set 1 and provide labels in {0,1}.")
    parser.add_argument("--loss_func", type=str, default="mse",
                        choices=["bce", "mse", "l1", "smoothl1", "focalmse"])

    # LoRA / k-bit
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit (requires bitsandbytes).")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated module names to apply LoRA. Leave empty to let PEFT infer.")

    # Dataset-specific
    parser.add_argument("--prompt_style", type=int, default=0,
                        help="Passed to NAID.dataset.TextDataset.")

    # Eval/save
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Limit total saved checkpoints to save disk.")
    parser.add_argument("--load_best_model_at_end", action="store_true",
                        help="Load the best checkpoint (by lowest MSE) at end of training.")
    parser.add_argument("--ndcg_k", type=int, default=20,
                        help="NDCG@K cutoff. (Metric computation still fixed at 20 above; adjust if needed.)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Effective batch for LR scaling
    eff_gpus = max(1, torch.cuda.device_count())
    eff_batch = eff_gpus * args.batch_size * max(1, args.gradient_accumulation_steps)
    if args.learning_rate is None:
        args.learning_rate = args.base_lr * (eff_batch / 256.0)

    # Output directory
    if args.runs_dir is None:
        default_dir = datetime.now().strftime("%m-%d-%H-%M-%S")
        args.runs_dir = os.path.join("official_runs", default_dir)
    os.makedirs(args.runs_dir, exist_ok=True)
    save_args_to_json(args, os.path.join(args.runs_dir, "args.json"))

    # Seed for reproducibility
    set_seed(args.seed)

    # Load data
    df_trainval = pd.read_csv(args.data_path)
    df_test = pd.read_csv(args.test_data_path)  # DO NOT USE FOR PARAM SEARCHING

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        # Use eos as pad to keep SEQ_CLS heads happy
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    # NOTE: DO NOT pass device_map here for DDP (let Trainer/Accelerate handle device placement).
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=args.num_labels,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None),
        load_in_8bit=args.load_in_8bit,
    )
    # Explicitly set problem type for regression if num_labels == 1
    if args.num_labels == 1 and args.loss_func in {"mse", "l1", "smoothl1", "focalmse"}:
        base_model.config.problem_type = "regression"

    # Align pad id
    if getattr(base_model.config, "pad_token_id", None) is None and getattr(base_model.config, "eos_token_id", None) is not None:
        base_model.config.pad_token_id = base_model.config.eos_token_id

    # Wrap with LoRA (and k-bit preparation if enabled)
    model = build_lora_model(
        base_model=base_model,
        load_in_8bit=args.load_in_8bit,
        target_modules=args.target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Datasets
    total_dataset = TextDataset(df_trainval, tokenizer, args.max_length, args.prompt_style)
    total_size = len(total_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

    # (Optional) Test dataset — not used during training to avoid leakage
    _ = TextDataset(df_test, tokenizer, args.max_length, args.prompt_style)

    # Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.runs_dir,
        num_train_epochs=args.total_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.runs_dir, "logs"),
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="mse",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        ddp_find_unused_parameters=False,  # common for PEFT to avoid unused param warnings
        report_to=["tensorboard"],
    )

    # Trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_func=args.loss_func,
    )

    # Train
    trainer.train()

    # Save final (and tokenizer). If load_best_model_at_end=True, this will be the best.
    model_last_dir = os.path.join(args.runs_dir, "last")
    trainer.save_model(model_last_dir)  # includes LoRA adapters
    tokenizer.save_pretrained(model_last_dir)

    # If you need only LoRA adapter weights (without base):
    # from peft import PeftModel
    # if isinstance(model, PeftModel):
    #     model.save_pretrained(os.path.join(args.runs_dir, "lora_adapters_only"))

    print(f"Training done. Model saved to: {model_last_dir}")


if __name__ == "__main__":
    main()
