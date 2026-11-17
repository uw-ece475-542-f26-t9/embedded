"""load_session(): the one entry point. Picks the reader from the file type."""

from __future__ import annotations

from pathlib import Path

from .capture import read_capture
from .config import ROOT, load_config
from .legacy import read_legacy
from .session import Session


def load_session(source: str | Path, cfg: dict | None = None) -> Session:
    """A capture .bin, a sensor-testing .csv.gz, or a bare sensor-testing name like "A_squeeze"."""
    cfg = cfg or load_config()
    path = Path(source)
    if path.suffix == ".bin":
        return read_capture(path, cfg)
    if path.name.endswith((".csv.gz", ".csv")):
        return read_legacy(path, cfg)
    data_dir = (ROOT / cfg["input"]["legacy_data_dir"]).resolve()
    for candidate in (data_dir / f"{source}.csv.gz", data_dir / f"{source}.csv"):
        if candidate.exists():
            return read_legacy(candidate, cfg)
    raise FileNotFoundError(f"no recording called {source!r} (looked for a .bin, a .csv.gz, and in {data_dir})")
