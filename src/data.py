from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from .config import Config


def _fetch_subset(subset: str, cfg: Config, categories: list[str]) -> Tuple[list[str], list[int]]:
    """
    Offline-first: if dataset not cached locally and download_if_missing=False,
    sklearn will raise an error; we convert to FileNotFoundError with guidance.
    """
    try:
        data = fetch_20newsgroups(
            subset=subset,
            categories=categories,
            remove=("headers", "footers", "quotes"),
            download_if_missing=cfg.download_if_missing,
        )
        return data.data, data.target
    except Exception as e:
        # sklearn historically raises IOError/OSError, depending on version/environment.
        raise FileNotFoundError(
            "20newsgroups dataset is not found in local sklearn cache and "
            "download_if_missing=False (offline mode).\n"
            f"Original error: {repr(e)}"
        ) from e


def _build_df(texts: list[str], y: list[int], cfg: Config) -> pd.DataFrame:
    """
    Map sklearn targets (0/1) to our label convention:
      risk_category -> label 1
      safe_category -> label 0
    We control categories order: [safe, risk] OR [risk, safe]?
    We'll explicitly map by category name order in load function.
    """
    df = pd.DataFrame({"text": texts, "raw_target": y})
    df.reset_index(inplace=True)
    df.rename(columns={"index": "id"}, inplace=True)
    return df


def load_binary_20newsgroups(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns train/test DataFrames with columns: id, text, label
    """
    # Ensure deterministic mapping:
    # fetch_20newsgroups target indices follow categories list order.
    # We'll set categories = [cfg.safe_category, cfg.risk_category]
    categories = [cfg.safe_category, cfg.risk_category]

    # Load full available (train+test subsets) then do our own split for a stable pipeline.
    # This keeps a single evaluation split regardless of sklearn's internal split definition.
    texts_train, y_train = _fetch_subset("train", cfg, categories)
    texts_test, y_test = _fetch_subset("test", cfg, categories)

    df_all = pd.concat(
        [
            _build_df(texts_train, y_train, cfg),
            _build_df(texts_test, y_test, cfg),
        ],
        ignore_index=True,
    )

    # Map raw_target: 0 => safe_category => label=0, 1 => risk_category => label=1
    df_all["label"] = df_all["raw_target"].astype(int)
    df_all = df_all[["id", "text", "label"]]

    # Stratified split
    df_train, df_test = train_test_split(
        df_all,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=df_all["label"],
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_test
