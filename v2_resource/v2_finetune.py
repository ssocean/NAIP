import os
import csv
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed, tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

from v2_resource.NAIDv2.dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
class PairwiseBCELoss(nn.Module):
    '''
    Aka RankNetLoss
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred1, pred2, score1, score2):
        label = (score1 > score2).float()  # 1 if pred1 should be higher
        pred_diff = pred1 - pred2
        return F.binary_cross_entropy_with_logits(pred_diff, label)
# ---------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------

def write_val_preds_csv(
    runs_dir: str,
    epoch: int,
    ids: Sequence[Any],
    preds: Sequence[float],
    labels: Optional[Sequence[float]] = None,
    accepts: Optional[Sequence[Any]] = None,
) -> None:
    """
    Write per-sample pointwise validation predictions:
    columns: id, pred, label, accept, abs_error
    """
    os.makedirs(runs_dir, exist_ok=True)
    rows = []
    n = len(ids)
    for i in range(n):
        y_true = None if labels is None or i >= len(labels) else labels[i]
        y_pred = float(preds[i])
        abs_err = "" if y_true is None or y_true == "" else abs(float(y_true) - y_pred)
        rows.append(
            {
                "id": ids[i],
                "pred": y_pred,
                "label": "" if y_true is None else y_true,
                "accept": "" if accepts is None or i >= len(accepts) else accepts[i],
                "abs_error": abs_err,
            }
        )
    epoch_path = os.path.join(runs_dir, f"val_pointwise_preds_epoch{epoch}.csv")
    latest_path = os.path.join(runs_dir, "val_pointwise_preds_latest.csv")
    for p in (epoch_path, latest_path):
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "pred", "label", "accept", "abs_error"]
            )
            writer.writeheader()
            writer.writerows(rows)


def _upgrade_csv_header(csv_path: str, new_fields: List[str]) -> None:
    """If new metric columns appear, upgrade the CSV header while preserving history."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        rows = list(reader)
    if old_fields == new_fields:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for r in rows:
            for k in new_fields:
                r.setdefault(k, "")
            writer.writerow(r)


def log_metrics_single_csv(runs_dir: str, epoch: int, group: str, metrics: Dict[str, Any]) -> None:
    """
    Append metrics into runs_dir/val_metrics.csv.
    Automatically expands columns for new metric keys.
    """
    os.makedirs(runs_dir, exist_ok=True)
    csv_path = os.path.join(runs_dir, "val_metrics.csv")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    row: Dict[str, Any] = {"timestamp": ts, "epoch": epoch, "group": group}
    for k, v in (metrics or {}).items():
        row[k] = "" if v is None else (float(v) if hasattr(v, "__float__") else v)

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            current_fields = reader.fieldnames or []
        new_fields = list(current_fields)
        for k in row.keys():
            if k not in new_fields:
                new_fields.append(k)
        if new_fields != current_fields:
            _upgrade_csv_header(csv_path, new_fields)
    else:
        new_fields = list(row.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        for k in new_fields:
            row.setdefault(k, "")
        writer.writerow(row)


# ---------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------

def _parse_list(s: Optional[str]) -> Optional[List[Any]]:
    """Parse a list: prefer JSON; fallback to comma-separated."""
    if s is None or str(s).strip().lower() in ("", "none", "null"):
        return None
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return [e.strip() for e in str(s).split(",") if e.strip()]


def _parse_bucket_edges(s: Optional[str]) -> Optional[List[float]]:
    lst = _parse_list(s)
    if lst is None:
        return None
    out: List[float] = []
    for x in lst:
        if isinstance(x, str) and x.lower() in ("inf", "+inf", "infinity"):
            out.append(float("inf"))
        else:
            out.append(float(x))
    return out


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _parse_tuple_str(s: Optional[str], default: Tuple[str, ...] = ("pub_year", "cluster_cat")) -> Tuple[str, ...]:
    if not s:
        return tuple(default)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(parts) if parts else tuple(default)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise + Pointwise training with LoRA")

    # Data
    parser.add_argument("--data_path", type=str, default="./data/train.csv")
    parser.add_argument("--gt_field", type=str, default="gt")

    # Pairwise sampling (tunable)
    parser.add_argument("--pw_group_by_cluster_year", type=_str2bool, default=True)
    parser.add_argument("--pw_group_keys", type=str, default="pub_year,cluster_cat")
    parser.add_argument("--pw_min_diff", type=float, default=0.05)
    parser.add_argument(
        "--pw_bucket_edges",
        type=str,
        default='[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"inf"]',
        help='JSON or comma list; use "inf" for +infinity',
    )
    parser.add_argument(
        "--pw_target_ratio",
        type=str,
        default="[0.03,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.27,0.00]",
        help="JSON or comma list; should sum to ~1.0",
    )
    parser.add_argument("--pw_curriculum", type=_str2bool, default=True)
    parser.add_argument("--pw_balance", type=_str2bool, default=True)
    parser.add_argument("--pw_cap_per_paper", type=int, default=32, help="-1 means no cap")
    parser.add_argument("--pw_id_fields", type=str, default="id")
    parser.add_argument("--pw_use_weight", type=_str2bool, default=True)
    parser.add_argument(
        "--pw_weight_mode",
        type=str,
        default="linear_clip",
        choices=["none", "linear", "linear_clip", "sqrt"],
    )
    parser.add_argument("--pw_weight_clip_min", type=float, default=0.2)
    parser.add_argument("--pw_weight_clip_max", type=float, default=1.0)
    parser.add_argument("--pw_verbose", type=_str2bool, default=True)
    parser.add_argument("--pair_val_max_pairs", type=int, default=1_000_000)
    parser.add_argument("--additional_info", type=str, default=None, help="JSON list or comma list")

    # Model / Tokenization
    parser.add_argument("--checkpoint", type=str, default="your-base-checkpoint")
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Given a scientific paper, Title: {title}\nAbstract: {abstract}\nPlease evaluate the overall scientific quality:",
    )

    # Quantization / LoRA
    parser.add_argument("--load_in_8bit", type=_str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    # Train
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_func", type=str, default="default")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shuffle_train", type=_str2bool, default=True)
    parser.add_argument(
        "--need_pairwise_eval",
        action="store_true",
        help="Enable pairwise evaluation",
    )
    parser.add_argument("--loss_soft_tau", type=float, default=0.25)

    args = parser.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)
    save_path = os.path.join(args.runs_dir, "args.json")
    with open(save_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Args saved to {save_path}")
    return args


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------

class SiameseWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_ids = torch.cat([input_ids_a, input_ids_b], dim=0)
        all_mask = torch.cat([attention_mask_a, attention_mask_b], dim=0)
        logits = self.base_model(input_ids=all_ids, attention_mask=all_mask).logits.view(-1)
        p1, p2 = logits.chunk(2)
        return p1, p2


def get_model_and_tokenizer(args: argparse.Namespace) -> Tuple[AutoTokenizer, SiameseWrapper]:
    """Build tokenizer + model with optional 8-bit and LoRA."""
    use_cuda = torch.cuda.is_available()
    load_in_8bit = bool(args.load_in_8bit) and use_cuda

    quant_config = None
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=args.num_labels,
        quantization_config=quant_config,
        device_map="auto" if use_cuda else None,  # rely on HF accelerate mapping when CUDA
    )

    # Padding/eos setup
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = base_model.config.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()] if args.target_modules else None
    if target_modules and target_modules != [""]:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )

    # Prepare for k-bit training if quantized
    if load_in_8bit:
        base_model = prepare_model_for_kbit_training(base_model)

    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    siamese = SiameseWrapper(base_model)
    return tokenizer, siamese


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_regression_metrics(
    preds: Sequence[float],
    labels: Sequence[float],
    accept: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    scores = np.array(preds, dtype=float).squeeze()
    labels = np.array(labels, dtype=float).squeeze()

    # MAE
    metrics["mae"] = float(np.mean(np.abs(scores - labels)))

    # Optional classification metrics if accept labels exist
    if accept is not None and len(accept) == len(scores):
        try:
            accept_arr = np.array(accept).astype(float)
            metrics["roc_auc"] = float(roc_auc_score(accept_arr, scores))
            metrics["pr_auc"] = float(average_precision_score(accept_arr, scores))
        except Exception:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

    # Ranking metrics
    def safe_minmax(x: np.ndarray) -> np.ndarray:
        r = x.max() - x.min()
        return np.zeros_like(x) if r < 1e-8 else (x - x.min()) / (r + 1e-8)

    try:
        metrics["ndcg"] = float(ndcg_score([safe_minmax(labels)], [safe_minmax(scores)], k=20))
    except Exception:
        metrics["ndcg"] = float("nan")

    sr, _ = spearmanr(labels, scores)
    metrics["spearmanr"] = float(sr) if not np.isnan(sr) else 0.0
    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = get_args()

    # Reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    writer = SummaryWriter(log_dir=args.runs_dir) if is_main else None

    # Build model and tokenizer
    tokenizer, model = get_model_and_tokenizer(args)
    model = model.to(device)

    # Datasets
    df_all = pd.read_csv(args.data_path)

    # Shuffle before splitting for randomness
    df_all = df_all.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 9:1 split
    split_idx = int(len(df_all) * 0.9)
    df_train = df_all.iloc[:split_idx].reset_index(drop=True)
    df_val = df_all.iloc[split_idx:].reset_index(drop=True)

    bucket_edges = _parse_bucket_edges(args.pw_bucket_edges)
    target_ratio = _parse_list(args.pw_target_ratio)
    group_keys = _parse_tuple_str(args.pw_group_keys, default=("pub_year", "cluster_cat"))
    id_fields = _parse_tuple_str(args.pw_id_fields, default=("id", "paper_id", "guid"))

    if args.additional_info is None:
        additional_info: List[str] = []
    else:
        try:
            additional_info = json.loads(args.additional_info)
            if not isinstance(additional_info, list):
                additional_info = [str(additional_info)]
        except Exception:
            additional_info = [s for s in str(args.additional_info).split(",") if s.strip()]

    train_ds = PairwisePaperDataset(
        data=df_train,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt_template,
        gt_field=args.gt_field,
        max_pairs=args.max_pairs,
        seed=args.seed,
        group_by_cluster_year=args.pw_group_by_cluster_year,
        group_keys=group_keys,
        min_diff=args.pw_min_diff,
        bucket_edges=bucket_edges,
        target_ratio=target_ratio,
        curriculum=args.pw_curriculum,
        balance=args.pw_balance,
        cap_per_paper=None if args.pw_cap_per_paper is not None and args.pw_cap_per_paper < 0 else args.pw_cap_per_paper,
        id_fields_priority=id_fields,
        use_weight=args.pw_use_weight,
        weight_mode=args.pw_weight_mode,
        weight_clip_min=args.pw_weight_clip_min,
        weight_clip_max=args.pw_weight_clip_max,
        additional_info=additional_info,
        verbose=args.pw_verbose,
        max_samples=args.max_samples,
    )

    val_ds = SingleScoreDataset(
        data=df_val,
        tokenizer=tokenizer,
        max_length=args.max_length,
        gt_field=args.gt_field,
    )

    val_pair = PairwisePaperDataset(
        data=df_val,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt_template,
        gt_field=args.gt_field,
        max_pairs=args.pair_val_max_pairs,
        seed=args.seed,
        group_by_cluster_year=True,
        group_keys=group_keys,
        min_diff=0.0,
        bucket_edges=bucket_edges,
        target_ratio=target_ratio,
        curriculum=False,
        balance=True,
        cap_per_paper=1024,
        id_fields_priority=id_fields,
        use_weight=False,
        weight_mode=args.pw_weight_mode,
        weight_clip_min=args.pw_weight_clip_min,
        weight_clip_max=args.pw_weight_clip_max,
        additional_info=additional_info,
        verbose=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=bool(args.shuffle_train), collate_fn=lambda x: x
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)
    pair_val_loader = DataLoader(val_pair, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    # Optimizer & (later) scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, pair_val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, pair_val_loader
    )

    # Scheduler (calculate after prepare so lengths are per-process)
    num_update_steps = max(1, args.total_epochs * len(train_loader))
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_update_steps),
        num_training_steps=num_update_steps,
    )

    # Loss builder from your project
    loss_fn = PairwiseBCELoss(args)  # noqa: F405
    logging.info(f"Using loss: {loss_fn}")

    # Training loop
    for epoch in range(args.total_epochs):
        model.train()
        if is_main:
            logging.info(f"Epoch {epoch + 1}/{args.total_epochs}")

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", dynamic_ncols=True):
            a = torch.stack([f["input_ids_a"] for f in batch]).to(device)
            m1 = torch.stack([f["attention_mask_a"] for f in batch]).to(device)
            b = torch.stack([f["input_ids_b"] for f in batch]).to(device)
            m2 = torch.stack([f["attention_mask_b"] for f in batch]).to(device)

            score_a = torch.stack([f["gt_a"] for f in batch]).to(device)
            score_b = torch.stack([f["gt_b"] for f in batch]).to(device)

            p1, p2 = model(a, m1, b, m2)
            loss = loss_fn(p1, p2, score_a, score_b)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main and step % 10 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/loss", loss.item(), global_step)

        # -------------------- Validation --------------------
        model.eval()

        # Pointwise validation
        all_preds_chunks, all_labels_chunks, all_accept_chunks, all_id_chunks = [], [], [], []
        id_key: Optional[str] = None

        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(model)
            for batch in val_loader:
                tok = torch.stack([f["input_ids"] for f in batch]).to(device)
                att = torch.stack([f["attention_mask"] for f in batch]).to(device)
                y = torch.stack([f["label"] for f in batch]).to(device).view(-1)

                logits = unwrapped.base_model(input_ids=tok, attention_mask=att).logits.view(-1)

                preds_g = accelerator.gather_for_metrics(logits)
                labels_g = accelerator.gather_for_metrics(y)

                all_preds_chunks.append(preds_g.cpu())
                all_labels_chunks.append(labels_g.cpu())

                if "accept" in batch[0]:
                    acc = torch.stack([f["accept"] for f in batch]).to(device).view(-1)
                    acc_g = accelerator.gather_for_metrics(acc)
                    all_accept_chunks.append(acc_g.cpu())

                if id_key is None:
                    for k in ("id", "paper_id", "guid"):
                        if k in batch[0]:
                            id_key = k
                            break
                if id_key is not None:
                    try:
                        ids_local = torch.tensor(
                            [int(f[id_key]) for f in batch], device=device, dtype=torch.long
                        ).view(-1)
                        ids_g = accelerator.gather_for_metrics(ids_local)
                        all_id_chunks.append(ids_g.cpu())
                    except Exception:
                        id_key = None
                        all_id_chunks.clear()

        all_preds = torch.cat(all_preds_chunks, dim=0).numpy()
        all_labels = torch.cat(all_labels_chunks, dim=0).numpy()
        all_accept = torch.cat(all_accept_chunks, dim=0).numpy().tolist() if len(all_accept_chunks) > 0 else []
        all_ids = (
            torch.cat(all_id_chunks, dim=0).tolist()
            if len(all_id_chunks) > 0
            else list(range(len(all_preds)))
        )

        if is_main:
            pw_metrics = compute_regression_metrics(
                all_preds, all_labels, np.array(all_accept) if all_accept else None
            )
            for k, v in pw_metrics.items():
                writer.add_scalar(f"val/pointwise_{k}", v, epoch)
            logging.info(f"Pointwise Validation Metrics: {pw_metrics}")
            log_metrics_single_csv(args.runs_dir, epoch, "pointwise", pw_metrics)
            write_val_preds_csv(
                runs_dir=args.runs_dir,
                epoch=epoch,
                ids=all_ids,
                preds=all_preds,
                labels=all_labels,
                accepts=(all_accept if all_accept else None),
            )

        # Pairwise validation (optional)
        if args.need_pairwise_eval:
            all_pred_diff_chunks, all_gt_diff_chunks = [], []
            with torch.no_grad():
                unwrapped = accelerator.unwrap_model(model)
                for batch in pair_val_loader:
                    a = torch.stack([f["input_ids_a"] for f in batch]).to(device)
                    m1 = torch.stack([f["attention_mask_a"] for f in batch]).to(device)
                    b = torch.stack([f["input_ids_b"] for f in batch]).to(device)
                    m2 = torch.stack([f["attention_mask_b"] for f in batch]).to(device)
                    ga = torch.stack([f["gt_a"] for f in batch]).to(device).view(-1)
                    gb = torch.stack([f["gt_b"] for f in batch]).to(device).view(-1)

                    pa, pb = unwrapped(a, m1, b, m2)
                    pred_diff_g = accelerator.gather_for_metrics(pa - pb)
                    gt_diff_g = accelerator.gather_for_metrics(ga - gb)

                    all_pred_diff_chunks.append(pred_diff_g.cpu())
                    all_gt_diff_chunks.append(gt_diff_g.cpu())

            if is_main:
                pred_diff = torch.cat(all_pred_diff_chunks, dim=0).numpy()
                gt_diff = torch.cat(all_gt_diff_chunks, dim=0).numpy()

                pairwise_accuracy = float((np.sign(pred_diff) == np.sign(gt_diff)).mean())
                violation_mask = np.sign(pred_diff) != np.sign(gt_diff)
                violation_magnitude = float(
                    np.abs(pred_diff[violation_mask] - gt_diff[violation_mask]).mean()
                ) if violation_mask.any() else 0.0
                pairwise_mae = float(np.mean(np.abs(pred_diff - gt_diff)))

                writer.add_scalar("val/pairwise_accuracy", pairwise_accuracy, epoch)
                writer.add_scalar("val/pairwise_violation_magnitude", violation_magnitude, epoch)
                writer.add_scalar("val/pairwise_mae", pairwise_mae, epoch)

                pair_metrics = {
                    "pairwise_accuracy": pairwise_accuracy,
                    "violation_magnitude": violation_magnitude,
                    "pairwise_mae": pairwise_mae,
                }
                logging.info(f"Pairwise Validation Metrics: {pair_metrics}")
                log_metrics_single_csv(args.runs_dir, epoch, "pairwise", pair_metrics)

    accelerator.wait_for_everyone()

    if is_main:
        unwrapped = accelerator.unwrap_model(model).base_model
        output_dir = args.runs_dir or f"./runs/{datetime.now().strftime('%m%d%H%M')}_acc"
        unwrapped.save_pretrained(output_dir, safe_serialization=False)
        # Reuse tokenizer from outer scope
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")

    if is_main and writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
