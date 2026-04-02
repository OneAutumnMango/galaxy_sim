"""
Resumable, fault-tolerant Bonsai → HDF5 converter.

Usage (CLI):
    python -m galaxy_sim.viz.convert_snapshots [--output-dir output_2_1M]

Usage (API):
    from galaxy_sim.viz.convert_snapshots import convert_directory
    n_errors = convert_directory(Path("output_2_1M_100"))

For every raw snapshot found in the output directory:
  • If the matching HDF5 already exists and is valid  → delete the raw (free space)
  • If the matching HDF5 exists but is corrupt        → reconvert, then delete the raw
  • If the matching HDF5 does not exist yet           → convert, then delete the raw

After the main loop a final scan_and_fix pass catches any stragglers that may
have been missed.

Memory strategy: each file is processed in a subprocess via
multiprocessing.Pool(processes=1, maxtasksperchild=1).  The worker exits
after every single file so the OS reclaims ALL memory (numpy arrays, hdf5
buffers, Python heap fragmentation) between files.  The main process stays
under ~50 MB for the entire run regardless of how many files there are.
"""

from __future__ import annotations

import argparse
import multiprocessing
import struct
import sys
import time
from pathlib import Path

import h5py

from . import fix_corrupt


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[str, int, str]:
    """Delegate to fix_corrupt.fix_and_clean in a subprocess.

    Returns (status, n_particles, snap_name).
    • status is one of: "ok" | "fixed" | "converted"
    Raises on unrecoverable error (pool.apply re-raises in the main process).
    """
    snap_str, out_str = args
    snap_path = Path(snap_str)
    out_path  = Path(out_str)
    status, n = fix_corrupt.fix_and_clean(snap_path, out_path)
    return status, n, snap_path.name


# ---------------------------------------------------------------------------
# Time-based index helpers
# ---------------------------------------------------------------------------

def _raw_sim_time(path: Path) -> float:
    """Read the simulation time stored in the first 8 bytes of a Bonsai Tipsy file."""
    with open(path, "rb") as f:
        (t,) = struct.unpack("d", f.read(8))
    return t


def _build_hdf5_time_index(output_dir: Path) -> dict[float, tuple[int, Path]]:
    """Scan all snapshot_NNN.hdf5 files and return {sim_time: (index, path)}.

    Used to match surviving raw files to their already-converted HDF5 partners
    even when some raws have been deleted (so positional indexing is wrong).
    Corrupt/unreadable HDF5 files are silently skipped (they'll be reconverted).
    """
    index: dict[float, tuple[int, Path]] = {}
    for hdf5 in output_dir.glob("snapshot_*.hdf5"):
        try:
            idx = int(hdf5.stem.split("_")[-1])
        except ValueError:
            continue
        try:
            with h5py.File(hdf5, "r") as f:
                t = float(f["Header"].attrs["Time"])
            index[t] = (idx, hdf5)
        except Exception:
            continue
    return index


# ---------------------------------------------------------------------------
# Core logic (callable from other modules)
# ---------------------------------------------------------------------------

def convert_directory(output_dir: Path) -> int:
    """Convert / verify all raw Bonsai snapshots in *output_dir* to HDF5.

    Returns the number of errors (0 = all good).
    """
    output_dir = Path(output_dir).resolve()

    snaps = sorted(
        s for s in output_dir.iterdir()
        if s.suffix != ".hdf5" and not s.name.endswith(".tmp")
        and "snapshot" in s.name
    )

    total = len(snaps)
    if total == 0:
        return 0

    # Build a time → (index, hdf5_path) lookup from already-converted files.
    print("Building time index from existing HDF5 files ...", flush=True)
    hdf5_time_index = _build_hdf5_time_index(output_dir)
    next_new_index  = max((idx for idx, _ in hdf5_time_index.values()), default=-1) + 1

    snap_to_hdf5: list[tuple[Path, Path]] = []
    for snap in snaps:
        try:
            t = _raw_sim_time(snap)
        except Exception as exc:
            print(f"  WARNING: cannot read time from {snap.name}: {exc}", file=sys.stderr)
            t = None

        if t is not None and t in hdf5_time_index:
            _, hdf5_path = hdf5_time_index[t]
        else:
            hdf5_path = output_dir / f"snapshot_{next_new_index:03d}.hdf5"
            if t is not None:
                hdf5_time_index[t] = (next_new_index, hdf5_path)
            next_new_index += 1

        snap_to_hdf5.append((snap, hdf5_path))

    already_have_hdf5 = sum(1 for _, h in snap_to_hdf5 if h.exists())
    print(f"Found {total} raw Bonsai snapshots in {output_dir}")
    print(f"  {already_have_hdf5} have an existing HDF5 (will verify before deleting raw)")
    print(f"  {total - already_have_hdf5} need fresh conversion\n")

    errors:    list[tuple[int, Path, str]] = []
    converted  = 0
    verified   = 0
    fixed      = 0

    WORKER_TIMEOUT = 20  # seconds

    pool = multiprocessing.Pool(processes=1, maxtasksperchild=1)
    try:
        for i, (snap, out_path) in enumerate(snap_to_hdf5):
            t0 = time.monotonic()
            try:
                print(f"[{i+1:4d}/{total}] {snap.name}", flush=True)
                status, n, name = pool.apply_async(
                    _worker, ((str(snap), str(out_path)),)
                ).get(timeout=WORKER_TIMEOUT)
                elapsed = time.monotonic() - t0

                if status == "ok":
                    verified += 1
                    tag = "verified"
                elif status == "fixed":
                    fixed += 1
                    tag = "FIXED (was corrupt)"
                else:
                    converted += 1
                    tag = "converted"

                print(
                    f"[{i+1:4d}/{total}] done: {name} → {out_path.name}"
                    f"  {tag}  ({n:,} particles, {elapsed:.1f}s)"
                )
                sys.stdout.flush()
            except multiprocessing.TimeoutError:
                elapsed = time.monotonic() - t0
                pool.terminate()
                pool.join()
                pool = multiprocessing.Pool(processes=1, maxtasksperchild=1)
                msg = f"TimeoutError: worker did not finish within {WORKER_TIMEOUT}s"
                errors.append((i, snap, msg))
                print(
                    f"[{i+1:4d}/{total}] ERROR {snap.name}: {msg} ({elapsed:.1f}s)",
                    file=sys.stderr,
                )
                sys.stderr.flush()
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = f"{type(exc).__name__}: {exc}"
                errors.append((i, snap, msg))
                print(
                    f"[{i+1:4d}/{total}] ERROR {snap.name}: {msg} ({elapsed:.1f}s)",
                    file=sys.stderr,
                )
                sys.stderr.flush()
    finally:
        pool.terminate()
        pool.join()

    print(f"\n--- Conversion pass complete ---")
    print(f"Converted this run  : {converted}")
    print(f"Verified & cleaned  : {verified}")
    print(f"Corrupt & re-fixed  : {fixed}")
    print(f"Errors this run     : {len(errors)}")

    if errors:
        print("\nFailed snapshots:")
        for idx, path, msg in errors:
            print(f"  [{idx:4d}] {path.name}: {msg}")

    # Final safety pass: catch any raws still present (e.g. from previous failed runs)
    print("\n--- Running final cleanup pass ---")
    n_remaining_errors = fix_corrupt.scan_and_fix(output_dir)

    return len(errors) + n_remaining_errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Bonsai snapshots to HDF5")
    parser.add_argument(
        "--output-dir", default="output_2_1M",
        help="Directory containing Bonsai snapshots (default: output_2_1M)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    n_errors = convert_directory(output_dir)
    sys.exit(0 if n_errors == 0 else 1)


if __name__ == "__main__":
    main()
