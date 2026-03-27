from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimParams:
    # Particle count
    n_particles: int = 1_000_000

    # Galaxy geometry — Hernquist profile (bulge) + exponential disk
    galaxy_mass: float = 1.0        # 1e10 Msun
    bulge_fraction: float = 0.1     # fraction of mass in bulge
    disk_scale_length: float = 3.0  # kpc
    disk_scale_height: float = 0.3  # kpc
    bulge_scale_radius: float = 0.5 # kpc, Hernquist a

    # Simulation duration & output
    time_end: float = 5.0           # internal time units (Gyr at kpc/km/s units)
    snap_interval: float = 0.05     # time between snapshots
    softening: float = 0.1          # gravitational softening (kpc)

    # GADGET-4 binary & paths
    gadget4_bin: Path = Path("gadget4/Gadget4")
    output_dir: Path = Path("output")
    ic_file: Path = Path("ic/galaxy.hdf5")
    param_template: Path = Path("configs/galaxy.param")
    n_mpi: int = 1                  # MPI ranks (set >1 for cluster runs)
    n_omp: int = 0                  # OpenMP threads (0 = auto)

    def __post_init__(self):
        self.gadget4_bin = Path(self.gadget4_bin)
        self.output_dir = Path(self.output_dir)
        self.ic_file = Path(self.ic_file)
        self.param_template = Path(self.param_template)
