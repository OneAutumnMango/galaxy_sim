"""Run the Bonsai GPU N-body code instead of GADGET-4."""

from __future__ import annotations

import multiprocessing
import os
import resource
import subprocess
import time
from multiprocessing import cpu_count
from pathlib import Path

from galaxy_sim.sim.params import SimParams


def _set_unlimited_stack() -> None:
    """Set the process stack size to unlimited (needed by Bonsai)."""
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def run_bonsai(params: SimParams) -> Path:
    """Generate ICs as a Tipsy file and launch Bonsai2.

    Returns the output directory path.
    """
    from galaxy_sim.ic.ic_gen import generate_galaxy_ic
    from galaxy_sim.ic.tipsy_writer import write_tipsy_ic

    bonsai_bin = Path(params.bonsai_bin)
    if not bonsai_bin.exists():
        raise FileNotFoundError(
            f"Bonsai binary not found at {bonsai_bin}. "
            "Expected at bonsai/runtime/build/bonsai2_slowdust"
        )

    params.output_dir.mkdir(parents=True, exist_ok=True)
    params.ic_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate ICs
    print(f"Generating ICs: {params.n_particles} particles...")
    pos, vel, mass = generate_galaxy_ic(params)

    # Bonsai uses G=1 internally (N-body convention).  Our ICs use physical
    # units: positions in kpc, velocities in km/s, masses in 1e10 Msun.
    # Scaling masses by G_code makes Bonsai's G=1 equivalent to the physical G,
    # so that circular velocities and orbital periods come out correctly.
    G_CODE = 4.302e4   # kpc (km/s)^2 (1e10 Msun)^-1
    tipsy_ic = params.ic_file.with_suffix(".tipsy")
    write_tipsy_ic(tipsy_ic, pos, vel, mass * G_CODE)
    print(f"Tipsy IC written to {tipsy_ic}")

    # Bonsai fixed timestep: fine enough for ~10 substeps per snapshot
    dt = min(params.snap_interval / 10.0, 0.005)

    snap_base = str((params.output_dir / "snapshot_").resolve())

    cmd = [
        str(bonsai_bin.resolve()),
        "-i", str(tipsy_ic.resolve()),
        "-t", str(dt),
        "-T", str(params.time_end),
        "-e", str(params.softening),
        "--snapname", snap_base,
        "--snapiter", str(params.snap_interval),
        "--dev", "0",
    ]

    env = os.environ.copy()
    omp_threads = params.n_omp if params.n_omp > 0 else cpu_count()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    print(f"OMP_NUM_THREADS={omp_threads}")
    print(f"Running: {' '.join(cmd)}")

    subprocess.run(cmd, env=env, check=True, preexec_fn=_set_unlimited_stack)

    # Convert Tipsy snapshots → HDF5 for the downstream viz pipeline
    _convert_snapshots(params.output_dir, params)

    return params.output_dir


def _convert_worker(args: tuple) -> str:
    """Worker: convert one Tipsy file to HDF5 and delete the raw file.

    Runs in an isolated subprocess (maxtasksperchild=1) so all numpy/hdf5
    memory is returned to the OS by the kernel after each file exits.
    """
    from galaxy_sim.sim.snap_convert import tipsy_to_hdf5
    snap_str, out_str, params = args
    snap_path = Path(snap_str)
    out_path  = Path(out_str)
    tipsy_to_hdf5(snap_path, out_path, params)
    snap_path.unlink()
    return snap_path.name


def _convert_snapshots(output_dir: Path, params: SimParams) -> None:
    """Convert all Bonsai Tipsy output snapshots to GADGET-4-compatible HDF5.

    Resumable: skips any snapshot whose .hdf5 already exists.
    Fault-tolerant: logs errors per file and continues rather than aborting.
    Memory-safe: each file is converted in a fresh subprocess
    (maxtasksperchild=1) so the OS reclaims all memory between files.
    """
    snaps = sorted(
        s for s in output_dir.iterdir()
        if s.suffix not in (".hdf5", ".tmp") and "snapshot" in s.name
    )

    total = len(snaps)
    print(f"Converting {total} Bonsai snapshots to HDF5...")
    errors = 0

    with multiprocessing.Pool(processes=1, maxtasksperchild=1) as pool:
        for i, snap in enumerate(snaps):
            out = output_dir / f"snapshot_{i:03d}.hdf5"
            if out.exists():
                continue
            t0 = time.monotonic()
            try:
                name = pool.apply(_convert_worker, ((str(snap), str(out), params),))
                elapsed = time.monotonic() - t0
                print(f"  [{i:03d}] {name} → {out.name}  ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.monotonic() - t0
                errors += 1
                print(f"  [{i:03d}] ERROR converting {snap.name}: {exc}  ({elapsed:.1f}s)")

    if errors:
        print(f"Done ({errors} error(s) — re-run to retry failed files).")
    else:
        print("Done.")

