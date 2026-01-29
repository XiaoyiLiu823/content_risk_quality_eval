from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import safe_div


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int


def confusion_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Confusion:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def metrics_from_confusion(c: Confusion) -> Dict[str, float]:
    precision = safe_div(c.tp, c.tp + c.fp)
    recall = safe_div(c.tp, c.tp + c.fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # As required
    mis_kill_rate = safe_div(c.fp, c.tp + c.fp)  # FP/(TP+FP)
    miss_rate = safe_div(c.fn, c.tp + c.fn)      # FN/(TP+FN)

    return {
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "tn": c.tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mis_kill_rate": mis_kill_rate,
        "miss_rate": miss_rate,
        "support_pos": c.tp + c.fn,
        "support_neg": c.tn + c.fp,
        "support_total": c.tp + c.fp + c.fn + c.tn,
    }


def add_segments(df: pd.DataFrame, segments: Tuple[Tuple[str, int, int | None], ...]) -> pd.DataFrame:
    """
    Segment by text length in chars (after strip).
    Buckets:
      0-50, 51-120, 121-300, 301+
    """
    out = df.copy()
    lengths = out["text"].astype(str).str.strip().str.len()
    out["text_len"] = lengths

    def bucket(n: int) -> str:
        for name, lo, hi in segments:
            if hi is None:
                if n >= lo:
                    return name
            else:
                if lo <= n <= hi:
                    return name
        # fallback (shouldn't happen)
        return segments[-1][0]

    out["segment"] = out["text_len"].apply(bucket)
    return out


def segment_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """
    Return per-segment confusion + derived metrics
    """
    rows = []
    for seg, g in df.groupby("segment", dropna=False):
        c = confusion_from_labels(g[y_true_col].to_numpy(), g[y_pred_col].to_numpy())
        m = metrics_from_confusion(c)
        m["segment"] = seg
        rows.append(m)

    seg_df = pd.DataFrame(rows).sort_values(by="support_total", ascending=False).reset_index(drop=True)
    return seg_df
