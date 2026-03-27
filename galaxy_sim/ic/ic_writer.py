"""Write GADGET-4 HDF5 initial condition files."""

from __future__ import annotations

import numpy as np
import h5py
from pathlib import Path

from galaxy_sim.sim.params import SimParams


def write_hdf5_ic(
    path: Path,
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    params: SimParams,
    ntypes: int = 2,
) -> None:
    """Write positions/velocities/masses to a GADGET-4 HDF5 IC file.

    All particles are type 1 (dark matter / collisionless).
    ntypes must match the NTYPES compiled into GADGET-4 (default 2).
    """
    N = len(pos)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    npart = np.zeros(ntypes, dtype=np.uint32)
    npart[1] = N  # type 1 = collisionless

    with h5py.File(path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["NumPart_ThisFile"] = npart
        hdr.attrs["NumPart_Total"] = npart
        hdr.attrs["NumPart_Total_HighWord"] = np.zeros(ntypes, dtype=np.uint32)
        hdr.attrs["MassTable"] = np.zeros(ntypes, dtype=np.float64)
        hdr.attrs["Time"] = 0.0
        hdr.attrs["Redshift"] = 0.0
        hdr.attrs["BoxSize"] = 0.0
        hdr.attrs["NumFilesPerSnapshot"] = 1
        hdr.attrs["Omega0"] = 0.0
        hdr.attrs["OmegaLambda"] = 0.0
        hdr.attrs["HubbleParam"] = 1.0
        hdr.attrs["Flag_Sfr"] = 0
        hdr.attrs["Flag_Cooling"] = 0
        hdr.attrs["Flag_StellarAge"] = 0
        hdr.attrs["Flag_Metals"] = 0
        hdr.attrs["Flag_Feedback"] = 0
        hdr.attrs["Flag_DoublePrecision"] = 1

        grp = f.create_group("PartType1")
        grp.create_dataset("Coordinates", data=pos.astype(np.float64))
        grp.create_dataset("Velocities", data=vel.astype(np.float64))
        grp.create_dataset("Masses", data=mass.astype(np.float64))
        grp.create_dataset(
            "ParticleIDs",
            data=np.arange(1, N + 1, dtype=np.uint64),
        )
