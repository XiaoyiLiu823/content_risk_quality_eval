from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import Config
from .metrics import add_segments, confusion_from_labels, metrics_from_confusion, segment_metrics
from .utils import clamp_text, ensure_dir, safe_div


def _version_name(th: float) -> str:
    # Make stable names for thresholds
    return f"v_th_{th:.2f}"


def _make_pred(df: pd.DataFrame, threshold: float) -> pd.Series:
    return (df["score"] >= float(threshold)).astype(int)


def _flatten_segment_metrics(seg_df: pd.DataFrame) -> Dict[str, float]:
    """
    Flatten per-segment metrics into a single row dict with prefixed columns.
    Example: seg_0-50_precision, seg_0-50_tp ...
    """
    out: Dict[str, float] = {}
    for _, r in seg_df.iterrows():
        seg = str(r["segment"]).replace("+", "plus").replace("-", "_")
        prefix = f"seg_{seg}_"
        for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "mis_kill_rate", "miss_rate", "support_total"]:
            out[prefix + k] = float(r[k])
    return out


def build_reports_and_save(cfg: Config, df_scored: pd.DataFrame) -> None:
    """
    Create 3 CSVs:
      - daily_quality_metrics.csv
      - top_fp_fn_breakdown.csv
      - action_list.csv
    """
    out_dir = ensure_dir(cfg.outputs_dir)

    # Add segment column
    df = add_segments(df_scored, cfg.segments)

    daily_rows: List[Dict[str, float]] = []
    breakdown_rows: List[Dict[str, float]] = []
    action_rows: List[Dict[str, object]] = []

    for th in cfg.thresholds:
        version = _version_name(th)
        pred_col = f"pred_label_{version}"

        df[pred_col] = _make_pred(df, th)

        # Overall metrics
        c = confusion_from_labels(df["label"].to_numpy(), df[pred_col].to_numpy())
        overall = metrics_from_confusion(c)

        # Per-segment metrics
        seg_df = segment_metrics(df, "label", pred_col)
        seg_flat = _flatten_segment_metrics(seg_df)

        daily_row = {
            "version": version,
            "threshold": float(th),
            **overall,
            **seg_flat,
        }
        daily_rows.append(daily_row)

        # FP/FN breakdown by segment
        for error_type in ["FP", "FN"]:
            if error_type == "FP":
                err_mask = (df["label"] == 0) & (df[pred_col] == 1)
            else:
                err_mask = (df["label"] == 1) & (df[pred_col] == 0)

            err_df = df.loc[err_mask, ["segment"]].copy()
            total_err = int(err_df.shape[0])

            seg_counts = (
                err_df.groupby("segment")
                .size()
                .reset_index(name="count")
                .sort_values(by="count", ascending=False)
                .reset_index(drop=True)
            )

            if seg_counts.empty:
                # still output a row indicating no errors
                breakdown_rows.append(
                    {
                        "version": version,
                        "threshold": float(th),
                        "error_type": error_type,
                        "segment": "ALL",
                        "count": 0,
                        "share": 0.0,
                        "rank": 1,
                        "total_error": 0,
                    }
                )
            else:
                seg_counts["share"] = seg_counts["count"].apply(lambda x: safe_div(int(x), total_err))
                seg_counts["rank"] = range(1, len(seg_counts) + 1)

                for _, r in seg_counts.iterrows():
                    breakdown_rows.append(
                        {
                            "version": version,
                            "threshold": float(th),
                            "error_type": error_type,
                            "segment": r["segment"],
                            "count": int(r["count"]),
                            "share": float(r["share"]),
                            "rank": int(r["rank"]),
                            "total_error": total_err,
                        }
                    )

            # Action list samples (max 50 each)
            sample_n = int(cfg.sample_per_error_type)
            sample_df = df.loc[err_mask, ["id", "score", "segment", "text"]].copy()
            if not sample_df.empty:
                # More actionable: sort by confidence (FP high score, FN low score)
                sample_df = sample_df.sort_values(
                    by="score",
                    ascending=(error_type == "FN"),
                ).head(sample_n)

                for _, r in sample_df.iterrows():
                    action_rows.append(
                        {
                            "id": int(r["id"]),
                            "version": version,
                            "error_type": error_type,
                            "score": float(r["score"]),
                            "segment": r["segment"],
                            "text_snippet": clamp_text(r["text"], cfg.text_snippet_len),
                        }
                    )

    # Save outputs
    daily_df = pd.DataFrame(daily_rows)
    daily_path = Path(out_dir) / "daily_quality_metrics.csv"
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")

    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_path = Path(out_dir) / "top_fp_fn_breakdown.csv"
    breakdown_df.to_csv(breakdown_path, index=False, encoding="utf-8-sig")

    action_df = pd.DataFrame(action_rows)
    action_path = Path(out_dir) / "action_list.csv"
    action_df.to_csv(action_path, index=False, encoding="utf-8-sig")
