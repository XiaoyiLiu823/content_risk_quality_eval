from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import Config


def train_and_score(cfg: Config, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Train TF-IDF + LogisticRegression, output per-sample predicted probability score on test set.
    Returns df_scored with columns: id, text, label, score
    """
    if df_train.empty or df_test.empty:
        raise ValueError("Train/Test data is empty. Check dataset loading and split.")

    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg.tfidf_max_features,
                    ngram_range=cfg.tfidf_ngram_range,
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=2,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    C=cfg.lr_C,
                    max_iter=cfg.lr_max_iter,
                    solver="liblinear",  # stable for binary classification
                    n_jobs=1,
                ),
            ),
        ]
    )

    pipe.fit(df_train["text"].astype(str), df_train["label"].astype(int))

    # Probability of positive class (label=1)
    proba = pipe.predict_proba(df_test["text"].astype(str))[:, 1]

    df_scored = df_test.copy()
    df_scored["score"] = proba.astype(float)

    return df_scored
