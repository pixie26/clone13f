"""Runtime metadata, progress, and serialization helpers.

This module is deliberately independent of research and backtest logic so CLI
entry points can use it without creating domain-layer dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def process_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return None


def frame_shallow_mb(frame: pd.DataFrame | None) -> float:
    if frame is None:
        return 0.0
    return float(frame.memory_usage(index=True, deep=False).sum() / (1024**2))


def progress_printer(stage_started: float, log_path: Path | None = None):
    def emit(message: str) -> None:
        rss = process_rss_mb()
        rss_text = f", rss={rss:,.0f}MB" if rss is not None else ""
        line = f"{message}; stage_elapsed={time.perf_counter() - stage_started:.1f}s{rss_text}"
        print(f"      {line}", flush=True)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    return emit


def config_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def write_manifest(path: Path, payload: dict) -> None:
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config_hash": config_hash(payload),
        "library_versions": versions,
        **payload,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
