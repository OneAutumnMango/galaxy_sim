"""
python -m galaxy_sim.run  --n 1000000 --steps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from galaxy_sim.sim.params import SimParams
from galaxy_sim.sim.runner import run_simulation


def main() -> None:
    p = argparse.ArgumentParser(description="Run a galaxy N-body simulation via GADGET-4")
    p.add_argument("--n", type=int, default=1_000_000,
                   help="Total number of particles across both galaxies (split equally between the two)")
    p.add_argument("--mass", type=float, default=1.0, help="Galaxy mass (1e10 Msun)")
    p.add_argument("--time", type=float, default=5.0, help="End time (code units ≈ Gyr)")
    p.add_argument("--snap-interval", type=float, default=0.05)
    p.add_argument("--softening", type=float, default=0.1, help="Gravitational softening (kpc)")
    p.add_argument("--output", default="output/", help="Output directory")
    p.add_argument("--gadget4-bin", default="gadget4/Gadget4")
    p.add_argument("--bonsai-bin", default="bonsai/runtime/build/bonsai2_slowdust")
    p.add_argument("--backend", choices=["bonsai", "gadget4"], default="bonsai",
                   help="Simulation backend (default: bonsai for GPU)")
    p.add_argument("--omp", type=int, default=0, help="OpenMP threads (0=auto-detect all cores)")
    args = p.parse_args()

    params = SimParams(
        n_particles=args.n // 2,  # --n is total; each galaxy gets half
        galaxy_mass=args.mass,
        time_end=args.time,
        snap_interval=args.snap_interval,
        softening=args.softening,
        output_dir=Path(args.output),
        gadget4_bin=Path(args.gadget4_bin),
        bonsai_bin=Path(args.bonsai_bin),
        backend=args.backend,
        n_omp=args.omp,
    )

    run_simulation(params)


if __name__ == "__main__":
    main()
