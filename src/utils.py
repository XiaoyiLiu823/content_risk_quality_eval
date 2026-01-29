from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clamp_text(text: str, max_len: int) -> str:
    if text is None:
        return ""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return t[:max_len]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def pretty_print(title: str, msg: Any) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}\n{msg}\n")
