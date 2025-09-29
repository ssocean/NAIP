#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAIPv2 eval script (pointwise) aligned with the latest training code.

✅ Matches the training pipeline semantics:
- Loads the **saved base model** (training wrapped it with `SiameseWrapper`,
  but at save-time it wrote `accelerator.unwrap_model(model).base_model`).
- Uses the same `SingleScoreDataset` for pointwise evaluation.
- Computes metrics consistent with training (`NDCG@20` via min-max scaling,
  Spearman, ROC AUC) + best-threshold classification summaries:
  - Best-F1 row: P / R / F1 / Acc
  - Best-Acc row: P / R / F1 / Acc
- Writes two CSV files into the checkpoint directory (`--ckpt_dir`):
  1) `eval_metrics.csv`: rows for best-F1 and best-Acc containing
     P, R, F1, Acc, NDCG, AUC, Spearman (and the threshold used).
  2) `eval_pointwise_preds_with_meta.csv`: id, pred, label, accept,
     pub_year, cluster_cat (and abs_error when label is present).

Note: pairwise evaluation is not included, as requested — this focuses on
pointwise metrics and per-sample outputs with meta columns.
"""

import os
import csv
import json
import time
import math
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import ndcg_score

# tokenizer / model
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from v2_resource.NAIDv2.dataset import SingleScoreDataset


# datasets (same modules as training)



def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def safe_minmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    r = x.max() - x.min()
    if not np.isfinite(r) or r < 1e-8:
        return np.zeros_like(x)
    return (x - x.min()) / (r + 1e-8)


def compute_ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)

    ndcg = float(ndcg_score([safe_minmax(labels)], [safe_minmax(scores)], k=20))
    sr, _ = spearmanr(labels, scores)
    sr = 0.0 if (sr is None or np.isnan(sr)) else float(sr)
    return {"NDCG": ndcg, "Spearman": sr}


def compute_auc(scores: np.ndarray, accepts: np.ndarray) -> float:
    """ROC AUC if possible; returns NaN if not computable."""
    try:
        return float(roc_auc_score(accepts.astype(int), scores.astype(float)))
    except Exception:
        return float("nan")


def best_threshold_fast(scores: np.ndarray, accepts: np.ndarray) -> Tuple[dict, dict]:
    """
    Efficient sweep of all decision thresholds (on sorted unique scores) to find
    the threshold that maximizes F1 and the one that maximizes Accuracy.

    Returns (best_f1_row, best_acc_row), where each row is a dict containing:
    {"threshold", "precision", "recall", "f1", "accuracy"}
    """
    s = scores.astype(float)
    y = accepts.astype(int)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos

    # Sort by score (desc). At cut k => predict positive for top-k.
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp_cum = np.cumsum(y_sorted)
    ks = np.arange(1, n + 1)
    tp = tp_cum
    fp = ks - tp
    fn = pos - tp
    tn = neg - (ks - tp)

    precision = np.divide(tp, ks, out=np.zeros_like(tp, dtype=float), where=ks > 0)
    recall = np.divide(tp, pos, out=np.zeros_like(tp, dtype=float), where=pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    accuracy = np.divide(tp + tn, n, out=np.zeros_like(tp, dtype=float), where=n > 0)

    # Include k=0 case (predict all negative)
    precision0 = 0.0
    recall0 = 0.0
    f10 = 0.0
    acc0 = (tn := neg) / n if n > 0 else 0.0

    # Find best F1 & best Acc across k=0..N
    # Threshold corresponding to k means t = s_sorted[k-1]. For k=0, use +inf.
    best_f1_idx = int(np.argmax(np.concatenate([[f10], f1])))
    best_acc_idx = int(np.argmax(np.concatenate([[acc0], accuracy])))

    def pack(idx: int) -> dict:
        if idx == 0:
            return {
                "threshold": float("inf"),
                "precision": precision0,
                "recall": recall0,
                "f1": f10,
                "accuracy": float(acc0),
            }
        k = idx  # 1..N maps to numpy index k-1
        return {
            "threshold": float(s_sorted[k - 1]),
            "precision": float(precision[k - 1]),
            "recall": float(recall[k - 1]),
            "f1": float(f1[k - 1]),
            "accuracy": float(accuracy[k - 1]),
        }

    return pack(best_f1_idx), pack(best_acc_idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NAIPv2 Eval (pointwise), aligned with latest training code")

    # Data / model
    p.add_argument("--ckpt_dir", type=str, required=True, help="Path to the saved checkpoint directory (weights folder)")
    p.add_argument("--test_data_path", type=str, required=True, help="CSV with the evaluation set")
    p.add_argument("--gt_field", type=str, default="gt_10", help="Ground-truth field name for regression / ranking metrics")

    # Tokenization
    p.add_argument("--max_length", type=int, default=512)

    # Eval runtime
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load_in_8bit", type=lambda s: str(s).lower() in ("1","true","t","yes","y"), default=True)

    return p.parse_args()


from peft import AutoPeftModelForSequenceClassification

def load_tok_and_model(ckpt_dir: str, load_in_8bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)


    model = AutoPeftModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        torch_dtype=torch.float16,
        num_labels=1
    )


    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model



def main():
    args = parse_args()

    # Repro & env parity with training
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    # Load tokenizer & saved base model
    tokenizer, model = load_tok_and_model(args.ckpt_dir, args.load_in_8bit)
    if not getattr(model, "is_loaded_in_8bit", False) and not getattr(model, "is_loaded_in_4bit", False):
        model = model.to(device)

    # Load evaluation CSV (we will later merge meta from here using `id`)
    df_val = pd.read_csv(args.test_data_path)

    # Build dataset / loader (identical SingleScoreDataset as training pointwise val)
    val_ds = SingleScoreDataset(data=df_val, tokenizer=tokenizer, max_length=args.max_length, gt_field=args.gt_field)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    model, val_loader = accelerator.prepare(model, val_loader)

    # Collect predictions / labels / accepts / ids
    all_preds_t, all_labels_t, all_accept_t, all_ids_t = [], [], [], []
    id_key = None

    model.eval()
    with torch.no_grad():
        unwrapped = accelerator.unwrap_model(model)  # NOTE: saved base model, *not* SiameseWrapper
        for batch in val_loader:
            tok = torch.stack([f['input_ids'] for f in batch]).to(device)
            att = torch.stack([f['attention_mask'] for f in batch]).to(device)
            y = torch.stack([f['label'] for f in batch]).to(device).view(-1)

            logits = unwrapped(input_ids=tok, attention_mask=att).logits.view(-1)

            preds_g = accelerator.gather_for_metrics(logits)
            labels_g = accelerator.gather_for_metrics(y)
            all_preds_t.append(preds_g.cpu())
            all_labels_t.append(labels_g.cpu())

            if 'accept' in batch[0]:
                acc = torch.stack([f['accept'] for f in batch]).to(device).view(-1)
                acc_g = accelerator.gather_for_metrics(acc)
                all_accept_t.append(acc_g.cpu())

            # Try to gather numeric id so we can merge back meta (pub_year, cluster_cat)
            if id_key is None:
                for k in ("id", "paper_id", "guid"):
                    if k in batch[0]:
                        id_key = k
                        break
            if id_key is not None:
                try:
                    ids_local = torch.tensor([int(f[id_key]) for f in batch], device=device, dtype=torch.long).view(-1)
                    ids_g = accelerator.gather_for_metrics(ids_local)
                    all_ids_t.append(ids_g.cpu())
                except Exception:
                    id_key = None
                    all_ids_t.clear()

    # Flatten
    preds = torch.cat(all_preds_t, dim=0).numpy()
    labels = torch.cat(all_labels_t, dim=0).numpy()
    accepts = (torch.cat(all_accept_t, dim=0).numpy().astype(int) if len(all_accept_t) > 0 else None)
    all_ids = (torch.cat(all_ids_t, dim=0).numpy().tolist() if len(all_ids_t) > 0 else list(range(len(preds))))

    # ----- Metrics -----
    ranking = compute_ranking_metrics(preds, labels)

    # ROC AUC (if accept available)
    auc = float("nan")
    best_f1_row = {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "accuracy": float("nan")}
    best_acc_row = {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "accuracy": float("nan")}

    if accepts is not None and accepts.shape[0] == preds.shape[0]:
        auc = compute_auc(preds, accepts)
        bf1, bacc = best_threshold_fast(preds, accepts)
        best_f1_row = bf1
        best_acc_row = bacc

    # ----- Write CSVs to ckpt_dir -----
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 1) Metrics CSV
    metrics_rows = []
    row_f1 = {
        "scenario": "best_f1",
        "threshold": best_f1_row["threshold"],
        "P": best_f1_row["precision"],
        "R": best_f1_row["recall"],
        "F1": best_f1_row["f1"],
        "Acc": best_f1_row["accuracy"],
        "NDCG": ranking["NDCG"],
        "AUC": auc,
        "Spearman": ranking["Spearman"],
    }
    row_acc = {
        "scenario": "best_acc",
        "threshold": best_acc_row["threshold"],
        "P": best_acc_row["precision"],
        "R": best_acc_row["recall"],
        "F1": best_acc_row["f1"],
        "Acc": best_acc_row["accuracy"],
        "NDCG": ranking["NDCG"],
        "AUC": auc,
        "Spearman": ranking["Spearman"],
    }
    metrics_rows.extend([row_f1, row_acc])

    metrics_path = os.path.join(args.ckpt_dir, "eval_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_f1.keys()))
        writer.writeheader()
        for r in metrics_rows:
            writer.writerow(r)

    # 2) Per-sample predictions with meta (pub_year, cluster_cat)
    # Merge by id if we managed to gather it; else write id as running index.
    # Convert id to string for robust merge
    # 2) Per-sample predictions with meta (pub_year, cluster_cat)
    df_out = pd.DataFrame({
        "id": [str(i) for i in all_ids],
        "pred": preds.astype(float),
        "label": labels.astype(float),
    })

    if accepts is not None:
        df_out["accept"] = accepts.astype(int)


    if "pub_year" in df_val.columns and len(df_val) == len(df_out):
        df_out["pub_year"] = df_val["pub_year"].values
    if "cluster_cat" in df_val.columns and len(df_val) == len(df_out):
        df_out["cluster_cat"] = df_val["cluster_cat"].values


    if "id" in df_val.columns:
        meta = df_val[["id", "pub_year", "cluster_cat"]].copy()
        meta["id"] = meta["id"].astype(str)
        merged = df_out.merge(meta, on="id", how="left", suffixes=("", "_m"))

        need_fallback = False
        for col in ["pub_year_m", "cluster_cat_m"]:
            if col in merged.columns:
                nan_ratio = merged[col].isna().mean()
                if nan_ratio > 0.3:
                    need_fallback = True
                    break
        if not need_fallback:

            if "pub_year_m" in merged: merged["pub_year"] = merged["pub_year_m"]
            if "cluster_cat_m" in merged: merged["cluster_cat"] = merged["cluster_cat_m"]
            merged.drop(columns=[c for c in ["pub_year_m", "cluster_cat_m"] if c in merged], inplace=True)
            df_out = merged

    # abs error
    try:
        df_out["abs_error"] = (df_out["label"].astype(float) - df_out["pred"].astype(float)).abs()
    except Exception:
        pass

    preds_path = os.path.join(args.ckpt_dir, "eval_pointwise_preds_with_meta.csv")
    df_out.to_csv(preds_path, index=False, encoding="utf-8")


    if accelerator.is_main_process:
        print("\nSaved:")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Preds+meta: {preds_path}")
        print("\nBest-F1:", row_f1)
        print("Best-Acc:", row_acc)


if __name__ == "__main__":
    main()
