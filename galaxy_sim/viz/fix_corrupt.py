"""Shared corruption-detection/repair library for Bonsai → HDF5 conversion.

As a module   : from fix_corrupt import verify_hdf5, tipsy_to_hdf5, fix_and_clean, scan_and_fix
As a script   : python fix_corrupt.py [--output-dir output_2_1M]

The script scans the output directory.  For every raw Bonsai snapshot found,
it uses its sorted position to determine the matching snapshot_NNN.hdf5:
  • HDF5 exists and is valid   → delete raw
  • HDF5 exists but corrupt    → reconvert from raw, delete raw
  • HDF5 does not exist yet    → convert raw, delete raw
A final summary reports any failures.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

DARKMATTERID = 3_000_000_000_000_000_000


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def verify_hdf5(path: Path) -> tuple[bool, str]:
    """Quick sanity-check a GADGET-4 HDF5 snapshot.

    Returns (True, info_string) if the file looks valid,
            (False, error_string) otherwise.
    """
    try:
        with h5py.File(path, "r") as f:
            t = float(f["Header"].attrs["Time"])
            n = int(f["PartType1/Coordinates"].shape[0])
        return True, f"time={t:.4f}, N={n:,}"
    except Exception as exc:
        return False, str(exc)


def tipsy_to_hdf5(src: Path, dst: Path) -> int:
    """Convert a single Bonsai Tipsy V2 snapshot to GADGET-4 HDF5.

    Writes to a .tmp sidecar first so a crash never leaves a corrupt HDF5.
    Returns the number of dark-matter particles written.
    """
    with open(src, "rb") as f:
        raw_header = f.read(32)
        time_val, nbodies, ndim, nsph, ndark, nstar, version = struct.unpack(
            "d6i", raw_header
        )
        dark_dtype = np.dtype([
            ("mass",  np.float32),
            ("pos",   np.float32, (3,)),
            ("vel",   np.float32, (3,)),
            ("id_lo", np.int32),
            ("id_hi", np.int32),
        ])
        assert dark_dtype.itemsize == 36, "Unexpected dark particle size"
        raw = f.read(ndark * 36)
        if len(raw) != ndark * 36:
            raise ValueError(
                f"Truncated file: expected {ndark * 36} bytes, got {len(raw)}"
            )

    p    = np.frombuffer(raw, dtype=dark_dtype)
    pos  = p["pos"].astype(np.float64)
    vel  = p["vel"].astype(np.float64)
    mass = p["mass"].astype(np.float64)
    ids  = (
        (p["id_lo"].astype(np.uint64) & np.uint64(0xFFFFFFFF))
        | (p["id_hi"].astype(np.uint64) << np.uint64(32))
    )
    ids = ids - np.uint64(DARKMATTERID) + np.uint64(1)

    npart = np.zeros(2, dtype=np.uint32)
    npart[1] = ndark

    tmp = dst.with_suffix(".tmp")
    try:
        with h5py.File(tmp, "w") as out:
            hdr = out.create_group("Header")
            hdr.attrs["NumPart_ThisFile"]       = npart
            hdr.attrs["NumPart_Total"]          = npart
            hdr.attrs["NumPart_Total_HighWord"] = np.zeros(2, dtype=np.uint32)
            hdr.attrs["MassTable"]              = np.zeros(2, dtype=np.float64)
            hdr.attrs["Time"]                   = float(time_val)
            hdr.attrs["Redshift"]               = 0.0
            hdr.attrs["BoxSize"]                = 0.0
            hdr.attrs["NumFilesPerSnapshot"]    = 1
            hdr.attrs["Omega0"]                 = 0.0
            hdr.attrs["OmegaLambda"]            = 0.0
            hdr.attrs["HubbleParam"]            = 1.0
            for flag in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge",
                         "Flag_Metals", "Flag_Feedback", "Flag_DoublePrecision"):
                hdr.attrs[flag] = 0
            grp = out.create_group("PartType1")
            grp.create_dataset("Coordinates", data=pos)
            grp.create_dataset("Velocities",  data=vel)
            grp.create_dataset("Masses",      data=mass)
            grp.create_dataset("ParticleIDs", data=ids)
        tmp.rename(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return ndark


def fix_and_clean(raw_path: Path, hdf5_path: Path) -> tuple[str, int]:
    """Ensure *hdf5_path* is a valid conversion of *raw_path*, then delete *raw_path*.

    Decision tree
    -------------
    1. HDF5 exists and is valid  → delete raw, return ("ok", N)
    2. HDF5 exists but corrupt   → reconvert from raw, delete raw, return ("fixed", N)
    3. HDF5 does not exist       → convert raw, delete raw, return ("converted", N)

    Raises FileNotFoundError if the raw is missing and the HDF5 is also
    absent or corrupt (nothing we can do).
    """
    raw_name  = raw_path.name
    hdf5_name = hdf5_path.name
    was_corrupt = False

    if hdf5_path.exists():
        print(f"    found pair   : {raw_name} + {hdf5_name}", flush=True)
        print(f"    checking     : verifying {hdf5_name} ...", flush=True)
        ok, info = verify_hdf5(hdf5_path)
        if ok:
            print(f"    valid        : {info}", flush=True)
            print(f"    deleting raw : {raw_name}", flush=True)
            raw_path.unlink()
            with h5py.File(hdf5_path, "r") as f:
                n = int(f["PartType1/Coordinates"].shape[0])
            return "ok", n
        # Corrupt — must reconvert
        print(f"    CORRUPT      : {info}", flush=True)
        print(f"    removing bad : {hdf5_name}", flush=True)
        was_corrupt = True
        hdf5_path.unlink(missing_ok=True)
    else:
        print(f"    new file     : {raw_name} (no HDF5 yet)", flush=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file missing and HDF5 {'corrupt' if was_corrupt else 'absent'}: {raw_path}"
        )

    print(f"    converting   : {raw_name} → {hdf5_name} ...", flush=True)
    n = tipsy_to_hdf5(raw_path, hdf5_path)
    print(f"    deleting raw : {raw_name}", flush=True)
    raw_path.unlink()
    return ("fixed" if was_corrupt else "converted"), n


# ---------------------------------------------------------------------------
# Directory-level scan
# ---------------------------------------------------------------------------

def scan_and_fix(output_dir: Path) -> int:
    """Scan *output_dir* for raw Bonsai snapshots.

    Uses the same sorted-order index convention as convert_snapshots.py:
    the i-th raw (alphabetically) corresponds to snapshot_{i:03d}.hdf5.

    For each raw found:
      • verifies the matching HDF5 (reconverts if corrupt or absent)
      • deletes the raw file once the HDF5 is confirmed good

    Returns the number of failures (0 means all clean).
    """
    raws = sorted(
        s for s in output_dir.iterdir()
        if s.suffix != ".hdf5" and not s.name.endswith(".tmp")
        and "snapshot" in s.name
    )

    if not raws:
        print("No raw snapshots found — nothing to do.")
        return 0

    print(f"Found {len(raws)} raw snapshot(s) to process.")
    errors: list[tuple[int, str, str]] = []

    for i, raw in enumerate(raws):
        hdf5 = output_dir / f"snapshot_{i:03d}.hdf5"
        try:
            status, n = fix_and_clean(raw, hdf5)
            tag = {"ok": "OK (verified)", "fixed": "FIXED (reconverted)",
                   "converted": "CONVERTED"}.get(status, status.upper())
            print(f"  [{i:4d}] {raw.name} → {hdf5.name}  {tag}  ({n:,} particles)")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  [{i:4d}] {raw.name}  ERROR: {msg}", file=sys.stderr)
            errors.append((i, raw.name, msg))

    total_hdf5    = len(list(output_dir.glob("snapshot_*.hdf5")))
    remaining_raw = len([s for s in output_dir.iterdir()
                         if s.suffix != ".hdf5" and "snapshot" in s.name])
    print(f"\nTotal HDF5 files : {total_hdf5}")
    print(f"Remaining raw    : {remaining_raw}")
    print(f"Errors           : {len(errors)}")

    if errors:
        print("\nFailed snapshots:")
        for idx, name, msg in errors:
            print(f"  [{idx:4d}] {name}: {msg}")

    return len(errors)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify, repair, and clean up raw Bonsai snapshots"
    )
    parser.add_argument(
        "--output-dir", default="output_2_1M",
        help="Directory containing Bonsai snapshots (default: output_2_1M)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    n_errors = scan_and_fix(output_dir)
    sys.exit(0 if n_errors == 0 else 1)


if __name__ == "__main__":
    main()
