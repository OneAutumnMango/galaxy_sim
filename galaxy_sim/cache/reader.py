"""Read GADGET-4 HDF5 snapshots.

Snapshots can be split across multiple files (snapshot_000.0.hdf5, .1, …).
This reader transparently handles both single-file and multi-file snapshots.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


class Snapshot:
    """One simulation snapshot loaded into memory."""

    def __init__(self, time: float, pos: np.ndarray, vel: np.ndarray):
        self.time = time
        self.pos = pos    # (N, 3) float32, kpc
        self.vel = vel    # (N, 3) float32, km/s

    def __repr__(self) -> str:
        return f"Snapshot(t={self.time:.3f}, N={len(self.pos)})"


def _load_hdf5_snap(path: Path) -> Snapshot:
    """Load positions and velocities from a single HDF5 snapshot file."""
    pos_parts, vel_parts = [], []

    with h5py.File(path, "r") as f:
        time = float(f["Header"].attrs["Time"])
        for ptype in range(6):
            key = f"PartType{ptype}"
            if key not in f:
                continue
            grp = f[key]
            if "Coordinates" not in grp:
                continue
            pos_parts.append(grp["Coordinates"][:])
            vel_parts.append(grp["Velocities"][:])

    pos = np.concatenate(pos_parts, axis=0) if pos_parts else np.empty((0, 3), np.float32)
    vel = np.concatenate(vel_parts, axis=0) if vel_parts else np.empty((0, 3), np.float32)
    return Snapshot(time, pos, vel)


class SnapshotCache:
    """Lazy index of all snapshots in an output directory.

    Usage::

        cache = SnapshotCache("output/")
        for snap in cache:
            print(snap.time, snap.pos.shape)

        snap = cache[5]   # index by snapshot number
    """

    _SNAP_RE = re.compile(r"snapshot_(\d+)(?:\.\d+)?\.hdf5$")

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._files: list[Path] = self._discover()

    def _discover(self) -> list[Path]:
        files: dict[int, Path] = {}
        for p in sorted(self.output_dir.rglob("*.hdf5")):
            m = self._SNAP_RE.search(p.name)
            if m:
                idx = int(m.group(1))
                # prefer the .0 sub-file; otherwise take any
                if idx not in files or ".0." in p.name:
                    files[idx] = p
        return [files[k] for k in sorted(files)]

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Snapshot:
        return _load_hdf5_snap(self._files[idx])

    def __iter__(self) -> Iterator[Snapshot]:
        for path in self._files:
            yield _load_hdf5_snap(path)

    def times(self) -> list[float]:
        result = []
        for path in self._files:
            with h5py.File(path, "r") as f:
                result.append(float(f["Header"].attrs["Time"]))
        return result
