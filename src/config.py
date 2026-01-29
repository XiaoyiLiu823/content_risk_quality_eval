from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Config:
    # Offline-first: do NOT download if missing.
    download_if_missing: bool = False

    # Pick any 2 classes from 20newsgroups as binary labels
    # label: 1=risk, 0=safe
    risk_category: str = "talk.politics.mideast"
    safe_category: str = "sci.space"

    # Data split
    test_size: float = 0.25
    random_seed: int = 42

    # Model params
    tfidf_max_features: int = 50000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    lr_C: float = 2.0
    lr_max_iter: int = 2000

    # Threshold versions
    thresholds: List[float] = (0.50, 0.60)

    # Segment buckets by text length (character count after strip)
    # (low, high) inclusive ranges; last bucket uses high=None
    segments = (
        ("0-50", 0, 50),
        ("51-120", 51, 120),
        ("121-300", 121, 300),
        ("301+", 301, None),
    )

    # Sampling for action list
    sample_per_error_type: int = 50
    text_snippet_len: int = 200

    # Paths
    outputs_dir: str = "outputs"
