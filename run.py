

## run.py（入口）

import sys
from pathlib import Path

from src.config import Config
from src.data import load_binary_20newsgroups
from src.model import train_and_score
from src.reporting import build_reports_and_save
from src.utils import ensure_dir, pretty_print


def main() -> int:
    cfg = Config()

    # Ensure outputs directory exists
    ensure_dir(cfg.outputs_dir)

    try:
        df_train, df_test = load_binary_20newsgroups(cfg)
    except FileNotFoundError as e:
        pretty_print("DATASET NOT FOUND (offline mode)", str(e))
        pretty_print(
            "HOW TO PRE-CACHE ONCE",
            "Run (with network) to cache, then you can run offline:\n"
            "python -c \"from sklearn.datasets import fetch_20newsgroups; "
            "fetch_20newsgroups(subset='train', download_if_missing=True); "
            "fetch_20newsgroups(subset='test', download_if_missing=True)\""
        )
        return 2
    except Exception as e:
        pretty_print("UNEXPECTED ERROR WHILE LOADING DATA", repr(e))
        return 3

    try:
        scored = train_and_score(cfg, df_train, df_test)
        build_reports_and_save(cfg, scored)
    except Exception as e:
        pretty_print("UNEXPECTED ERROR WHILE RUNNING PIPELINE", repr(e))
        return 4

    pretty_print(
        "DONE",
        f"Outputs saved under: {Path(cfg.outputs_dir).resolve()}\n"
        f"- daily_quality_metrics.csv\n"
        f"- top_fp_fn_breakdown.csv\n"
        f"- action_list.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
