import itertools
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import re
import random
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


def safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    if not isinstance(x, str):
        return str(x)
    return x


class SingleScoreDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 512, gt_field: str = "score_gauss_mix"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.gt_field = gt_field

        if self.gt_field not in data.columns:
            raise ValueError(f"gt_field '{self.gt_field}' not found in DataFrame columns.")
        if "title" not in data.columns or "abstract" not in data.columns:
            raise ValueError("DataFrame must contain 'title' and 'abstract' columns.")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _to_text(x) -> str:
        if pd.isna(x):
            return ""
        s = str(x)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = float(row[self.gt_field])

        title = self._to_text(row.get("title", ""))
        abstract = self._to_text(row.get("abstract", ""))

        text = f"Given a certain paper, Title: {title}\nAbstract: {abstract}\nEvaluate the quality of this paper:"

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=True,
        )

        length = int(encoded["length"].item())
        if length > self.max_length:
            print(f"[Truncated] sample #{idx}: tokenized length {length} > max_length={self.max_length}")

        output = {
            "input_ids": encoded["input_ids"].squeeze(0).to(torch.long),
            "attention_mask": encoded["attention_mask"].squeeze(0).to(torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }

        if "accept" in row and not pd.isna(row["accept"]):
            output["accept"] = torch.tensor(int(row["accept"]), dtype=torch.long)

        return output


def _to_list_str(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if (s[0] == "[" or s[0] == '"'):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
                if isinstance(parsed, str):
                    return [parsed]
            except Exception:
                pass
        return [s]
    return [str(v)]


def _safe_tensor(x, dtype=None):
    t = torch.as_tensor(x)
    return t.to(dtype) if dtype is not None else t


def drop_invalid_group_rows(df: pd.DataFrame,
                            group_keys=("pub_year", "cluster_cat")) -> pd.DataFrame:
    df = df.copy()
    df[group_keys] = df[list(group_keys)].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=group_keys)
    return df


class PairwisePaperDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        prompt_template: str = "Given a certain paper, Title: {title}\nAbstract: {abstract}\nEvaluate the quality of this paper:",
        pad_to_max_length: bool = True,
        gt_field: str = "score_gauss_mix",
        max_pairs: Optional[int] = None,
        seed: int = 42,
        group_by_cluster_year: bool = True,
        group_keys: Union[str, Sequence[str]] = ("pub_year", "cluster_cat"),
        min_diff: float = 0.05,
        bucket_edges: Optional[List[float]] = None,
        target_ratio: Optional[List[float]] = None,
        curriculum: bool = True,
        balance: bool = True,
        cap_per_paper: Optional[int] = 8,
        id_fields_priority: Tuple[str, ...] = ("id",),
        use_weight: bool = True,
        weight_mode: str = "linear_clip",
        weight_clip_min: float = 0.2,
        weight_clip_max: float = 1.0,
        additional_info: Optional[List[str]] = None,
        verbose: bool = True,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.prompt_template = prompt_template
        self.pad_to_max_length = pad_to_max_length

        self.gt_field = gt_field
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.max_pairs = int(max_pairs) if max_pairs is not None else None
        self.seed = int(seed)

        self.group_by_cluster_year = bool(group_by_cluster_year)
        self.group_keys = self._normalize_group_keys(group_keys)

        self.min_diff = float(min_diff)
        self.curriculum = bool(curriculum)
        self.balance = bool(balance)

        self.cap_per_paper = None if cap_per_paper is None else int(cap_per_paper)
        self.id_fields_priority = tuple(id_fields_priority)

        self.use_weight = bool(use_weight)
        self.weight_mode = weight_mode
        self.weight_clip_min = float(weight_clip_min)
        self.weight_clip_max = float(weight_clip_max)
        self.additional_info = additional_info or []
        self.verbose = bool(verbose)

        self._rng = random.Random(self.seed)

        if bucket_edges is None:
            bucket_edges = [i / 10.0 for i in range(10)] + [float("inf")]
        self.bucket_edges = list(bucket_edges)
        assert len(self.bucket_edges) >= 2 and self.bucket_edges[0] == 0.0

        if target_ratio is None:
            k = len(self.bucket_edges) - 1
            target_ratio = [1.0 / k] * k
        self.target_ratio = list(target_ratio)
        assert len(self.target_ratio) == (len(self.bucket_edges) - 1)

        df = data.copy()
        if self.max_samples is not None and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=self.seed).reset_index(drop=True)
        before = len(df)

        if self.group_by_cluster_year and len(self.group_keys) > 0:
            missing = [c for c in self.group_keys if c not in df.columns]
            if missing:
                raise ValueError(f"Missing group_keys columns: {missing}")

            df[list(self.group_keys)] = df[list(self.group_keys)].replace(r'^\s*$', pd.NA, regex=True)
            df = df.dropna(subset=list(self.group_keys))

        after = len(df)
        removed = before - after
        print(f"[Cleaning Dataset] keys={self.group_keys}, kept={after}, removed={removed}, total={before}")

        raw_pairs = self._build_raw_pairs(df)
        if not raw_pairs:
            raise ValueError("No valid raw pairs (non-zero diff) could be generated.")

        pairs = self._sample_pairs_with_buckets(raw_pairs)

        if self.curriculum:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        else:
            self._rng.shuffle(pairs)

        self.pairs = pairs

        self.weights = None
        if self.use_weight:
            self.weights = [self._calc_weight(abs(d)) for _, _, d in self.pairs]

        if self.verbose:
            self._print_stats()

    @staticmethod
    def _normalize_group_keys(group_keys: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
        if group_keys is None:
            return tuple()
        if isinstance(group_keys, str):
            return (group_keys,)
        keys: List[str] = []
        for k in group_keys:
            if k is None:
                continue
            ks = str(k).strip()
            if ks:
                keys.append(ks)
        return tuple(keys)

    def _build_raw_pairs(self, data: pd.DataFrame) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        rows = data.to_dict(orient="records")
        raw_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []

        if self.group_by_cluster_year and len(self.group_keys) > 0:
            groups = defaultdict(list)
            for r in rows:
                key = tuple(r.get(k, None) for k in self.group_keys)
                if any(v is None for v in key):
                    continue
                groups[key].append(r)

            for group in groups.values():
                if len(group) < 2:
                    continue
                for a, b in itertools.combinations(group, 2):
                    sa = float(a[self.gt_field]); sb = float(b[self.gt_field])
                    diff = sa - sb
                    if diff != 0.0:
                        raw_pairs.append((a, b, diff))
        else:
            for a, b in itertools.combinations(rows, 2):
                sa = float(a[self.gt_field]); sb = float(b[self.gt_field])
                diff = sa - sb
                if diff != 0.0:
                    raw_pairs.append((a, b, diff))

        return raw_pairs

    def _paper_key(self, row: Dict[str, Any]) -> Optional[str]:
        for k in self.id_fields_priority:
            if k in row and row[k] is not None:
                return str(row[k])
        return None

    def _sample_pairs_with_buckets(self, raw_pairs):
        buckets = [[] for _ in range(len(self.bucket_edges) - 1)]
        for a, b, diff in raw_pairs:
            d = abs(diff)
            if d < self.min_diff:
                continue
            idx = self._bucket_index(d)
            buckets[idx].append((a, b, diff))

        total_target = self.max_pairs or sum(len(b) for b in buckets)
        want = [int(total_target * r) for r in self.target_ratio]

        sampled = []
        have = [0] * len(buckets)

        for i, bucket in enumerate(buckets):
            n_take = want[i]
            got = self._take_with_cap(bucket, n_take)
            sampled.extend(got)
            have[i] = len(got)

        deficit = total_target - len(sampled)
        if deficit > 0:
            for i in reversed(range(len(buckets))):
                if deficit <= 0:
                    break
                pool = buckets[i]
                remaining = pool[have[i]:] if have[i] < len(pool) else []
                extra = min(deficit, len(remaining))
                if extra > 0:
                    tmp = list(remaining)
                    self._rng.shuffle(tmp)
                    sampled.extend(tmp[:extra])
                    have[i] += extra
                    deficit -= extra

        if len(sampled) > total_target:
            sampled = sampled[:total_target]

        return sampled

    def _bucket_index(self, d: float) -> int:
        for i in range(len(self.bucket_edges) - 1):
            low, high = self.bucket_edges[i], self.bucket_edges[i + 1]
            if (low <= d < high) or (high == float("inf") and d >= low):
                return i
        return len(self.bucket_edges) - 2

    def _take_with_cap(self, candidates, n):
        if n <= 0:
            return []

        if self.cap_per_paper is None:
            if len(candidates) <= n:
                return list(candidates)
            tmp = list(candidates)
            self._rng.shuffle(tmp)
            return tmp[:n]

        counts = Counter()
        selected = []
        tmp = list(candidates)
        self._rng.shuffle(tmp)
        for a, b, diff in tmp:
            ka = self._paper_key(a)
            kb = self._paper_key(b)
            if ka is not None and counts[ka] >= self.cap_per_paper:
                continue
            if kb is not None and counts[kb] >= self.cap_per_paper:
                continue
            selected.append((a, b, diff))
            if ka is not None:
                counts[ka] += 1
            if kb is not None:
                counts[kb] += 1
            if len(selected) >= n:
                break
        return selected

    def _calc_weight(self, d_abs: float) -> float:
        if self.weight_mode == "none":
            return 1.0
        elif self.weight_mode == "linear":
            return max(1e-6, float(d_abs))
        elif self.weight_mode == "linear_clip":
            w = float(d_abs)
            return float(min(max(w, self.weight_clip_min), self.weight_clip_max))
        elif self.weight_mode == "sqrt":
            import math
            return math.sqrt(max(1e-6, float(d_abs)))
        else:
            w = float(d_abs)
            return float(min(max(w, self.weight_clip_min), self.weight_clip_max))

    def _print_stats(self):
        k = len(self.bucket_edges) - 1
        counts = [0] * k
        for _, _, diff in self.pairs:
            idx = self._bucket_index(abs(diff))
            counts[idx] += 1
        labels = []
        for i in range(k):
            low, high = self.bucket_edges[i], self.bucket_edges[i + 1]
            if high == float("inf"):
                labels.append(f"[{low:.1f}, +inf)")
            else:
                labels.append(f"[{low:.1f}, {high:.1f})")

        df = pd.DataFrame({"diff_range": labels, "count": counts})
        total = sum(counts)
        if total > 0:
            df["ratio"] = [c / total for c in counts]
        print("📊 Pair diff bucket distribution:")
        print(df.to_string(index=False))
        print(f"✅ Loaded {len(self.pairs)} pairs "
              f"{'(sorted by difficulty)' if self.curriculum else '(shuffled)'}; "
              f"min_diff={self.min_diff}, cap_per_paper={self.cap_per_paper}, use_weight={self.use_weight}")

    def __len__(self):
        return len(self.pairs)

    def _encode_text(self, item: dict):
        def safe_str(x):
            return str(x) if x is not None else ""

        title = safe_str(item.get("title")).strip()
        if not title:
            title = "[Title is empty]"

        abstract = safe_str(item.get("abstract")).strip()
        if not abstract:
            abstract = "[Abstract is empty]"

        extra_parts, extra_content = [], ""

        def _process_intro():
            intro_list = _to_list_str(item.get("intro_sentences"))
            if intro_list:
                return ["Entire Introduction: "] + intro_list, " ".join(intro_list)
            return ["Entire Introduction: [Empty]"], ""

        def _process_reviews():
            review_list = _to_list_str(item.get("review_comments"))
            if review_list:
                parts = ["Official Reviews: "] + [
                    f"Reviewer {i}: {review}" for i, review in enumerate(review_list, 1)
                ]
                return parts, " ".join(review_list)
            return ["Official Reviews: [Empty]"], ""

        def _process_conclusion():
            conclusion = safe_str(item.get("extracted_conclusion")).strip()
            if conclusion:
                return ["Conclusion: " + conclusion], conclusion
            return ["Conclusion: [Empty]"], ""

        def _process_image_score():
            score = safe_str(item.get("key_fig_score")).strip()
            if score:
                return [f"Key Fig (Aesthetics): {score}"], score
            return ["Key Fig (Aesthetics): [Empty]"], ""

        def _process_table_eval():
            eval_str = safe_str(item.get("table_desc")).strip()
            if eval_str:
                return [f"Key Tab (Objective Evaluation): {eval_str}"], eval_str
            return ["Key Tab (Objective Evaluation): [Empty]"], ""

        field_processors = {
            "intro_sentences": _process_intro,
            "review_comments": _process_reviews,
            "extracted_conclusion": _process_conclusion,
            "key_fig_score": _process_image_score,
            "table_desc": _process_table_eval,
        }

        for field in getattr(self, "additional_info", []):
            processor = field_processors.get(field)
            if processor:
                parts, content = processor()
                extra_parts.extend(parts)
                extra_content += content

        abstract_aug = abstract
        if extra_parts:
            abstract_aug += "\n" + "\n".join(extra_parts)

        text = self.prompt_template.format(title=title, abstract=abstract_aug)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=("max_length" if self.pad_to_max_length else False),
            truncation=True,
            return_tensors="pt",
            return_length=True,
        )

        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
            int(encoded["length"].item()),
        )

    def __getitem__(self, idx):
        a, b, diff = self.pairs[idx]

        if self.balance and self._rng.random() < 0.5:
            a, b = b, a
            diff = -diff

        label = 1.0 if diff > 0 else 0.0
        diff_abs = float(abs(diff))

        input_ids_a, attn_a, _ = self._encode_text(a)
        input_ids_b, attn_b, _ = self._encode_text(b)

        out = {
            "input_ids_a": _safe_tensor(input_ids_a, torch.long),
            "attention_mask_a": _safe_tensor(attn_a, torch.long),
            "input_ids_b": _safe_tensor(input_ids_b, torch.long),
            "attention_mask_b": _safe_tensor(attn_b, torch.long),
            "label": _safe_tensor(label, torch.float),
            "diff_abs": _safe_tensor(diff_abs, torch.float),
        }

        gt_a = label * diff_abs
        gt_b = (1.0 - label) * diff_abs
        out["gt_a"] = _safe_tensor(gt_a, torch.float)
        out["gt_b"] = _safe_tensor(gt_b, torch.float)

        if self.use_weight and self.weights is not None:
            w = float(self.weights[idx])
            out["weight"] = _safe_tensor(w, torch.float)

        return out
