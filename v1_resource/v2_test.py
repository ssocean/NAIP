# eval_inference.py
# -*- coding: utf-8 -*-
"""
Evaluation script for a LoRA fine-tuned Transformer model (PEFT).
- Loads a model trained with LoRA (AutoPeftModelForSequenceClassification).
- Runs inference/evaluation with Accelerate for multi-GPU support.
- Computes MSE, MAE, and NDCG@20.
- Uses TensorBoard SummaryWriter for logging.

Usage (multi-GPU):
  torchrun --nproc_per_node=8 eval_inference.py --weight_dir <trained_model_dir> --data_path <dataset.csv>

Author: YourName
"""

import os
import json
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from sklearn.metrics import ndcg_score
from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification

# Project-specific dataset loader (must return dict with input_ids, attention_mask, labels)
from offcial_train import TextDataset

# ------------------------
# Reproducibility
# ------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------
# Metrics
# ------------------------
def ndcg_at_k(predictions, labels, k: int = 20) -> float:
    """
    Compute NDCG@k.
    If the number of samples < k, return -1 as sentinel.
    """
    if len(predictions) < k:
        return -1.0
    return ndcg_score([labels], [predictions], k=k)


# ------------------------
# Save Args
# ------------------------
def save_args_to_json(args, file_path: str):
    """
    Save parsed arguments to a JSON file for reproducibility.
    """
    args_dict = vars(args)
    with open(file_path, "w") as f:
        json.dump(args_dict, f, indent=4)


# ------------------------
# Argument Parser
# ------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a transformer model with LoRA adaptation on text classification tasks."
    )

    # Dataset and model config
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset CSV file")
    parser.add_argument("--weight_dir", type=str, required=True,
                        help="Path to trained LoRA model directory")
    parser.add_argument("--checkpoint", type=str, default="llama3_weight",
                        help="(Optional) base model checkpoint path")

    # Tokenization / batching
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")

    # Loss & labels
    parser.add_argument("--loss_func", type=str, default="bce",
                        choices=["bce", "mse", "l1"],
                        help="Loss function used during training (for logging purposes)")
    parser.add_argument("--num_labels", type=int, default=1,
                        help="Number of labels for classification/regression")

    # LoRA config (not directly used here, but logged)
    parser.add_argument("--load_in_8bit", type=bool, default=True,
                        help="Whether model was trained in 8-bit mode")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    # Prompt style (dataset-specific)
    parser.add_argument("--prompt_style", type=int, default=0)

    # Logging
    default_tb_dir = datetime.now().strftime("%m-%d-%H-%M")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=".",  # 当前目录
        help="TensorBoard log directory"
    )

    return parser.parse_args()


# ------------------------
# Main Evaluation
# ------------------------
def main():
    args = get_args()

    # Accelerator (multi-GPU, mixed precision if configured)
    accelerator = Accelerator()

    # TensorBoard writer
    writer = SummaryWriter(args.runs_dir)

    # Save args for reproducibility
    save_args_to_json(args, os.path.join(args.runs_dir, "args.json"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.weight_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA model (AutoPeft handles base + adapter merge)
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        args.weight_dir,
        num_labels=args.num_labels,
        load_in_8bit=args.load_in_8bit,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    print("Model head (score layer) weights shape:", model.score.weight.shape)

    # Load dataset
    full_data = pd.read_csv(args.data_path)
    dataset = TextDataset(full_data, tokenizer,
                          max_length=args.max_length,
                          prompt_style=args.prompt_style)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test DataLoader has {len(test_loader)} batches.")

    # Prepare with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)

    # ------------------------
    # Evaluation Loop
    # ------------------------
    total_val_mse, total_val_mae = 0.0, 0.0
    all_pred, all_gt = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            outputs = model(**batch)
            predictions = outputs["logits"]
            labels = batch["labels"]

            all_gt.append(labels)
            all_pred.append(predictions.squeeze(1))

            mse = nn.MSELoss()(predictions.squeeze(1), labels).item()
            mae = nn.L1Loss()(predictions.squeeze(1), labels).item()

            total_val_mse += mse
            total_val_mae += mae

    # Average losses
    avg_mse = total_val_mse / len(test_loader)
    avg_mae = total_val_mae / len(test_loader)

    # Concatenate all predictions and labels
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    # Gather metrics across processes
    all_pred = accelerator.gather_for_metrics(all_pred)
    all_gt = accelerator.gather_for_metrics(all_gt)

    # Convert to numpy
    all_pred_np = all_pred.cpu().numpy()
    all_gt_np = all_gt.cpu().numpy()

    # Compute NDCG
    ndcg20 = ndcg_at_k(all_pred_np, all_gt_np, k=20)

    # ------------------------
    # Logging
    # ------------------------
    if accelerator.is_main_process:
        print("=" * 40)
        print(f"Evaluated model from: {args.weight_dir}")
        print(f"MSE:   {avg_mse:.4f}")
        print(f"MAE:   {avg_mae:.4f}")
        print(f"NDCG@20: {ndcg20:.4f}")
        print("=" * 40)

        writer.add_scalar("eval/mse", avg_mse)
        writer.add_scalar("eval/mae", avg_mae)
        writer.add_scalar("eval/ndcg20", ndcg20)
        writer.close()


if __name__ == "__main__":
    main()
