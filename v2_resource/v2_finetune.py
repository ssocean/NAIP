import os
import csv
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


from accelerate import Accelerator
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
    get_scheduler,
)

from v2_resource.NAIDv2.dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2_resource.NAIDv2.dataset import _ensure_list

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class PairwiseBCELoss(nn.Module):
    """
    Aka RankNetLoss (BT-style on diff)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred1, pred2, score1, score2):
        label = (score1 > score2).float()
        pred_diff = pred1 - pred2
        return F.binary_cross_entropy_with_logits(pred_diff, label)


# ---------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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
            writer = csv.DictWriter(f, fieldnames=["id", "pred", "label", "accept", "abs_error"])
            writer.writeheader()
            writer.writerows(rows)


def _upgrade_csv_header(csv_path: str, new_fields: List[str]) -> None:
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
def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTS Pairwise + Pointwise training with LoRA")

    # Data path / cols
    parser.add_argument("--data_path", type=str, default=r"C:\Users\Ocean\Documents\GitHub\reduct_academic\NAID_builder\datasets\dataset_heir_clusters_ward_2.2.csv")
    parser.add_argument("--title_col", type=str, default="title")
    parser.add_argument("--abstract_col", type=str, default="abstract")
    parser.add_argument("--year_col", type=str, default="pub_year")
    parser.add_argument("--cluster_col", type=str, default="cluster_id")
    parser.add_argument("--score_col", type=str, default="scores")
    parser.add_argument("--conf_col", type=str, default="confs")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--accept_col", type=str, default="accept")

    # Model / Tokenization
    parser.add_argument("--checkpoint", type=str, default=r"H:\weights\Qwen3-4B-Instruct-2507")
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)

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
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--shuffle_train", type=_str2bool, default=True)
    parser.add_argument("--need_pairwise_eval", action="store_true")

    # Pairwise dataset knobs (match your dataset)
    parser.add_argument("--max_pairs", type=int, default=10_000)
    parser.add_argument("--enable_flip", type=_str2bool, default=True)
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--enable_rts_jitter", type=_str2bool, default=False)
    parser.add_argument("--jitter_std_scale", type=float, default=1.0)
    parser.add_argument("--jitter_prob", type=float, default=0.3)

    # RTS kwargs knobs
    parser.add_argument("--rts_support_lo", type=float, default=0.0)
    parser.add_argument("--rts_support_hi", type=float, default=1.0)
    parser.add_argument("--rts_conf_lower_bound", type=float, default=0.2)
    parser.add_argument("--rts_sigma_base", type=float, default=0.2)
    parser.add_argument("--rts_sigma_bias", type=float, default=0.05)

    # Pointwise dataset knobs
    parser.add_argument("--score_reduce", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--use_accept", type=_str2bool, default=False)

    # Pairwise val size
    parser.add_argument("--pair_val_max_pairs", type=int, default=200_000)

    args = parser.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)
    save_path = os.path.join(args.runs_dir, "args.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    logging.info(f"Args saved to {save_path}")
    return args


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------
class SiameseWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
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
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=args.num_labels,
        device_map={"": torch.cuda.current_device()},
        load_in_8bit=args.load_in_8bit,
    )

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules if len(target_modules) > 0 else None,
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
    )

    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, SiameseWrapper(model)


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

    metrics["mae"] = float(np.mean(np.abs(scores - labels)))

    if accept is not None and len(accept) == len(scores):
        try:
            accept_arr = np.array(accept).astype(float)
            metrics["roc_auc"] = float(roc_auc_score(accept_arr, scores))
            metrics["pr_auc"] = float(average_precision_score(accept_arr, scores))
        except Exception:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

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
    set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    writer = SummaryWriter(log_dir=args.runs_dir) if is_main else None

    # Build model/tokenizer
    tokenizer, model = get_model_and_tokenizer(args)
    model = model.to(device)

    # Load data
    df_all = pd.read_csv(args.data_path)

    # ---- normalize required columns to your dataset expectation ----
    # scores/confs: list or string(list) -> list[float]
    df_all[args.score_col] = df_all[args.score_col].apply(_ensure_list)
    df_all[args.conf_col] = df_all[args.conf_col].apply(_ensure_list)

    # ensure year/cluster usable for groupby
    df_all[args.year_col] = df_all[args.year_col].astype(int)
    df_all[args.cluster_col] = df_all[args.cluster_col].astype(int)

    # Shuffle + split 9:1
    df_all = df_all.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    split_idx = int(len(df_all) * 0.9)
    df_train = df_all.iloc[:split_idx].reset_index(drop=True)
    df_val = df_all.iloc[split_idx:].reset_index(drop=True)

    # Year scales (建议用 train 统计，避免 val 泄漏；你想全量也行)
    year_scales = build_year_scales_from_df(
        df_train,
        score_col=args.score_col,
        conf_col=args.conf_col,
        year_col=args.year_col,
    )

    rts_kwargs = dict(
        support=(args.rts_support_lo, args.rts_support_hi),
        conf_lower_bound=args.rts_conf_lower_bound,
        sigma_base=args.rts_sigma_base,
        sigma_bias=args.rts_sigma_bias,
    )

    # ---- Datasets (use YOUR RTS datasets) ----
    train_ds = PairwisePaperDataset(
        df_train,
        tokenizer,
        year_scales=year_scales,
        max_length=args.max_length,
        max_pairs=args.max_pairs,
        enable_flip=args.enable_flip,
        flip_prob=args.flip_prob,
        enable_rts_jitter=args.enable_rts_jitter,
        jitter_std_scale=args.jitter_std_scale,
        jitter_prob=args.jitter_prob,
        rts_kwargs=rts_kwargs,
        seed=args.seed,
    )

    val_ds = SingleScoreDataset(
        df_val,
        tokenizer,
        max_length=args.max_length,
        score_reduce=args.score_reduce,
        use_accept=(args.use_accept and args.accept_col in df_val.columns),
    )

    # Pairwise val: 用同一种 PairwisePaperDataset，但关闭 jitter/curriculum 的概念在你这份里没有
    val_pair_ds = PairwisePaperDataset(
        df_val,
        tokenizer,
        year_scales=year_scales,   # 仍用 train 的 year_scales
        max_length=args.max_length,
        max_pairs=args.pair_val_max_pairs,
        enable_flip=False,
        enable_rts_jitter=False,
        rts_kwargs=rts_kwargs,
        seed=args.seed,
    )

    def _collate_identity(x):  # keep list-of-dicts
        return x

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=bool(args.shuffle_train), collate_fn=_collate_identity)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_identity)
    pair_val_loader = DataLoader(val_pair_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_identity)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model, optimizer, train_loader, val_loader, pair_val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, pair_val_loader
    )

    num_update_steps = max(1, args.total_epochs * len(train_loader))
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_update_steps),
        num_training_steps=num_update_steps,
    )

    loss_fn = PairwiseBCELoss()
    logging.info(f"Using loss: {loss_fn.__class__.__name__}")

    # -------------------- Training loop --------------------
    for epoch in range(args.total_epochs):
        model.train()
        if is_main:
            logging.info(f"Epoch {epoch + 1}/{args.total_epochs}")

        for step, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}",
            dynamic_ncols=True,
        ):
            a = torch.stack([f["input_ids_a"] for f in batch]).to(device)
            m1 = torch.stack([f["attention_mask_a"] for f in batch]).to(device)
            b = torch.stack([f["input_ids_b"] for f in batch]).to(device)
            m2 = torch.stack([f["attention_mask_b"] for f in batch]).to(device)

            # IMPORTANT: your dataset uses score_a/score_b (NOT gt_a/gt_b)
            score_a = torch.stack([f["score_a"] for f in batch]).to(device).view(-1)
            score_b = torch.stack([f["score_b"] for f in batch]).to(device).view(-1)

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
        id_key = args.id_col if args.id_col in df_val.columns else None

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

                if id_key is not None and id_key in batch[0]:
                    try:
                        ids_local = torch.tensor([int(f[id_key]) for f in batch], device=device, dtype=torch.long).view(-1)
                        ids_g = accelerator.gather_for_metrics(ids_local)
                        all_id_chunks.append(ids_g.cpu())
                    except Exception:
                        id_key = None
                        all_id_chunks.clear()

        all_preds = torch.cat(all_preds_chunks, dim=0).numpy()
        all_labels = torch.cat(all_labels_chunks, dim=0).numpy()
        all_accept = torch.cat(all_accept_chunks, dim=0).numpy().tolist() if len(all_accept_chunks) > 0 else []
        all_ids = torch.cat(all_id_chunks, dim=0).tolist() if len(all_id_chunks) > 0 else list(range(len(all_preds)))

        if is_main:
            pw_metrics = compute_regression_metrics(all_preds, all_labels, np.array(all_accept) if all_accept else None)
            for k, v in pw_metrics.items():
                writer.add_scalar(f"val/pointwise_{k}", v, epoch)
            logging.info(f"Pointwise Validation Metrics: {pw_metrics}")
            log_metrics_single_csv(args.runs_dir, epoch, "pointwise", pw_metrics)
            write_val_preds_csv(args.runs_dir, epoch, all_ids, all_preds, all_labels, (all_accept if all_accept else None))

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

                    ga = torch.stack([f["score_a"] for f in batch]).to(device).view(-1)
                    gb = torch.stack([f["score_b"] for f in batch]).to(device).view(-1)

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
                violation_magnitude = float(np.abs(pred_diff[violation_mask] - gt_diff[violation_mask]).mean()) if violation_mask.any() else 0.0
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
        unwrapped_base = accelerator.unwrap_model(model).base_model
        output_dir = args.runs_dir or f"./runs/{datetime.now().strftime('%m%d%H%M')}_acc"
        unwrapped_base.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")

    if is_main and writer is not None:
        writer.close()


if __name__ == "__main__":
    main()