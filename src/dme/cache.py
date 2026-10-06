from __future__ import annotations

import contextlib
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock, Timeout


@contextlib.contextmanager
def claimed_run_lock(lock_path: Path):
    """Yields the lock if it was free, None if another worker holds it."""
    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        yield None
        return
    try:
        yield lock
    finally:
        lock.release()


def save_json(path: Path, data: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    tmp.replace(path)


def save_jsonl_gz(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)


def load_jsonl_gz(path: Path) -> list[Any]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_or_init_cache_config(cache_dir: Path, config: dict[str, dict[str, Any]]) -> None:
    """
    Write config.json ({section: {key: value}}) on first use; raise ValueError if it differs from an existing one,
    so workers with different settings never mix results in one cache dir.
    """
    config_path = cache_dir / "config.json"
    if not config_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Per-process tmp name so concurrent workers don't race on the same file. os.replace is atomic on POSIX;
        # last writer wins (all write identical content).
        tmp = config_path.parent / f"config.json.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        tmp.replace(config_path)
        print(f"Cache config written to {config_path}")
        return

    existing = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = [
        f"  {sec}.{key}: cached={val!r}, current={config.get(sec, {}).get(key)!r}"
        for sec, entries in existing.items()
        for key, val in entries.items()
        if val is not None and config.get(sec, {}).get(key) != val
    ]
    if mismatches:
        raise ValueError(
            f"Cache directory '{cache_dir}' was created with a different config.\n"
            "Mismatched fields:\n" + "\n".join(mismatches) +
            "\nUse a different cache dir or delete the existing cache to start fresh."
        )
