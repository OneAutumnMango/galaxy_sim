"""Run the Bonsai GPU N-body code instead of GADGET-4."""

from __future__ import annotations

import os
import resource
import subprocess
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

    tipsy_ic = params.ic_file.with_suffix(".tipsy")
    write_tipsy_ic(tipsy_ic, pos, vel, mass)
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
        "--log",
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


def _convert_snapshots(output_dir: Path, params: SimParams) -> None:
    """Convert all Bonsai Tipsy output snapshots to GADGET-4-compatible HDF5."""
    from galaxy_sim.sim.snap_convert import tipsy_to_hdf5

    snaps = sorted(output_dir.glob("snapshot_*"))
    # Filter out any already-converted HDF5 files
    snaps = [s for s in snaps if s.suffix != ".hdf5"]

    print(f"Converting {len(snaps)} Bonsai snapshots to HDF5...")
    for i, snap in enumerate(snaps):
        out = output_dir / f"snapshot_{i:03d}.hdf5"
        tipsy_to_hdf5(snap, out, params)
    print("Done.")
