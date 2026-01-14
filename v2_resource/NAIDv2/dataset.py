# =========================
# Imports
# =========================
import ast
import math
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =========================
# Utilities
# =========================
def _ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    raise ValueError(f"Unsupported type: {type(x)}")


# =========================
# Year-level normalization
# =========================
def build_year_scales_from_df(
    df: pd.DataFrame,
    *,
    score_col: str = "scores",
    conf_col: str = "confs",
    year_col: str = "pub_year",
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Build per-year normalization ranges from the entire dataset.
    Dataset-level statistics (NOT per-paper).
    """
    year_scales = {}

    for year, g in df.groupby(year_col):
        all_scores = []
        all_confs = []

        for _, r in g.iterrows():
            all_scores.extend(_ensure_list(r[score_col]))
            all_confs.extend(_ensure_list(r[conf_col]))

        if not all_scores or not all_confs:
            raise ValueError(f"No reviews found for year {year}")

        year_scales[int(year)] = {
            "score_range": (float(min(all_scores)), float(max(all_scores))),
            "conf_range":  (float(min(all_confs)),  float(max(all_confs))),
        }

    return year_scales


# =========================
# RTS with uncertainty
# =========================
def compute_rts_with_uncertainty(
    scores,
    confs,
    *,
    score_range,
    conf_range,
    support=(0.0, 1.0),
    conf_lower_bound=0.2,
    sigma_base=0.2,
    sigma_bias=0.05,
    eps=1e-12,
):
    """
    Returns
    -------
    rts_mean : float
    rts_std  : float   (posterior std BEFORE truncation)
    """
    scores = np.asarray(scores, dtype=float)
    confs  = np.asarray(confs, dtype=float)

    x_lo, x_hi = support
    s_min, s_max = score_range
    c_min, c_max = conf_range

    # score normalization
    s_norm = (scores - s_min) / (s_max - s_min)
    s_norm = x_lo + s_norm * (x_hi - x_lo)
    s_norm = np.clip(s_norm, x_lo, x_hi)

    # confidence normalization
    c_norm = (confs - c_min) / (c_max - c_min)
    c_norm = np.clip(c_norm, conf_lower_bound, 1.0)

    sigma = sigma_base * (1.0 - c_norm) + sigma_bias
    precision = 1.0 / (sigma ** 2 + eps)

    mu = float(np.sum(s_norm * precision) / np.sum(precision))
    var = 1.0 / np.sum(precision)
    std = math.sqrt(max(var, eps))

    # truncated normal mean
    alpha = (x_lo - mu) / std
    beta  = (x_hi - mu) / std

    def phi(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def Phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    Z = Phi(beta) - Phi(alpha)
    if Z < eps:
        rts = float(np.clip(mu, x_lo, x_hi))
    else:
        rts = mu + std * (phi(alpha) - phi(beta)) / Z
        rts = float(np.clip(rts, x_lo, x_hi))

    return rts, std


# =========================
# Pair construction (offline, deterministic)
# =========================
def build_pairs(
    df: pd.DataFrame,
    *,
    year_scales: dict,
    score_col="scores",
    conf_col="confs",
    year_col="pub_year",
    cluster_col="cluster_id",
    max_pairs=10_000,
    rts_kwargs: dict | None = None,
):
    """
    Build hard-mined unordered pairs with clean RTS supervision.
    """
    if rts_kwargs is None:
        rts_kwargs = {}

    all_pairs = []
    groups = df.groupby([year_col, cluster_col])

    for (year, _), g in groups:
        if len(g) < 2:
            continue

        cfg = year_scales[int(year)]
        rows = g.to_dict("records")

        # compute clean RTS (mean only, no jitter)
        rts_list = []
        for r in rows:
            rts, _ = compute_rts_with_uncertainty(
                _ensure_list(r[score_col]),
                _ensure_list(r[conf_col]),
                score_range=cfg["score_range"],
                conf_range=cfg["conf_range"],
                **rts_kwargs,
            )
            rts_list.append(float(rts))

        # enumerate unordered pairs
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                if rts_list[i] != rts_list[j]:
                    all_pairs.append(
                        (rows[i], rows[j], rts_list[i], rts_list[j])
                    )

    # hard mining by |ΔRTS|
    all_pairs.sort(key=lambda x: abs(x[2] - x[3]), reverse=True)
    return all_pairs[:max_pairs]


# =========================
# Pointwise baseline dataset
# =========================
class SingleScoreDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        score_reduce: str = "mean",  # mean | median
        use_accept: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_accept = use_accept

        if score_reduce == "mean":
            self.df["quality"] = self.df["scores"].apply(
                lambda x: float(sum(x)) / len(x)
            )
        elif score_reduce == "median":
            self.df["quality"] = self.df["scores"].apply(
                lambda x: float(pd.Series(x).median())
            )
        else:
            raise ValueError(score_reduce)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        text = (
            f"Title: {r.title}\n"
            f"Abstract: {r.abstract}\n"
            f"Evaluate the quality of this paper."
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(r.quality, dtype=torch.float),
        }

        if self.use_accept:
            out["accept"] = torch.tensor(int(r.accept), dtype=torch.long)

        return out


# =========================
# Pairwise RTS dataset (final)
# =========================
class PairwisePaperDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        year_scales: dict,
        max_length: int = 512,
        max_pairs: int = 10_000,
        enable_flip: bool = True,
        flip_prob: float = 0.5,
        enable_rts_jitter: bool = False,
        jitter_std_scale: float = 1.0,
        jitter_prob: float = 0.3,
        rts_kwargs: dict | None = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.enable_flip = enable_flip
        self.flip_prob = flip_prob

        self.enable_rts_jitter = enable_rts_jitter
        self.jitter_std_scale = jitter_std_scale
        self.jitter_prob = jitter_prob

        self.support = (0.0, 1.0)
        self.jitter_eps = 1e-6

        self.rng = random.Random(seed)

        self.pairs = build_pairs(
            df,
            year_scales=year_scales,
            max_pairs=max_pairs,
            rts_kwargs=rts_kwargs,
        )

    def __len__(self):
        return len(self.pairs)

    def _encode(self, r):
        text = (
            f"Title: {r['title']}\n"
            f"Abstract: {r['abstract']}\n"
            f"Evaluate the quality of this paper."
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        # unpack clean pair
        a, b, rts_a, rts_b = self.pairs[idx]

        # flip augmentation
        if self.enable_flip and self.rng.random() < self.flip_prob:
            a, b = b, a
            rts_a, rts_b = rts_b, rts_a

        # RTS jitter (anti-symmetric, runtime only)
        if self.enable_rts_jitter and self.rng.random() < self.jitter_prob:
            delta = abs(rts_a - rts_b)
            scale = self.jitter_std_scale * max(delta, self.jitter_eps)
            eps = self.rng.gauss(0.0, scale)

            rts_a = float(np.clip(rts_a + eps, *self.support))
            rts_b = float(np.clip(rts_b - eps, *self.support))

        diff = rts_a - rts_b
        label = 1.0 if diff > 0 else 0.0

        ida, ama = self._encode(a)
        idb, amb = self._encode(b)

        return {
            "input_ids_a": ida,
            "attention_mask_a": ama,
            "input_ids_b": idb,
            "attention_mask_b": amb,

            # RTS supervision
            "score_a": torch.tensor(rts_a, dtype=torch.float),
            "score_b": torch.tensor(rts_b, dtype=torch.float),

            # auxiliary
            "label": torch.tensor(label, dtype=torch.float),
            "diff_abs": torch.tensor(abs(diff), dtype=torch.float),
        }
