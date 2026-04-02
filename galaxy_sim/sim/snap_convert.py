"""Convert a Bonsai Tipsy V2 snapshot to a GADGET-4-compatible HDF5 file."""

from __future__ import annotations

import gc
import struct
import numpy as np
import h5py
from pathlib import Path

from galaxy_sim.sim.params import SimParams

DARKMATTERID = 3_000_000_000_000_000_000


def tipsy_to_hdf5(tipsy_path: Path, hdf5_path: Path, params: SimParams) -> None:
    """Read a Bonsai Tipsy V2 binary snapshot and write it as GADGET-4 HDF5.

    Only dark matter particles are handled (Bonsai collisionless mode).
    """
    tipsy_path = Path(tipsy_path)
    hdf5_path  = Path(hdf5_path)

    with open(tipsy_path, "rb") as f:
        raw_header = f.read(32)
        time, nbodies, ndim, nsph, ndark, nstar, version = struct.unpack("d6i", raw_header)

        # dark_particleV2: mass(f32) pos[3](f32) vel[3](f32) _ID[2](i32x2)
        dark_dtype = np.dtype([
            ("mass",  np.float32),
            ("pos",   np.float32, (3,)),
            ("vel",   np.float32, (3,)),
            ("id_lo", np.int32),
            ("id_hi", np.int32),
        ])
        assert dark_dtype.itemsize == 36

        raw = f.read(ndark * 36)
        if len(raw) != ndark * 36:
            raise ValueError(
                f"Truncated file: expected {ndark * 36} bytes, got {len(raw)}"
            )
        particles = np.frombuffer(raw, dtype=dark_dtype)

    pos  = particles["pos"].astype(np.float64)
    vel  = particles["vel"].astype(np.float64)
    mass = particles["mass"].astype(np.float64)
    ids  = (particles["id_lo"].astype(np.uint64) & np.uint64(0xFFFFFFFF)) | \
           (particles["id_hi"].astype(np.uint64) << np.uint64(32))
    # Strip the DARKMATTERID offset to recover 0-based sequential IDs
    ids = ids - np.uint64(DARKMATTERID) + np.uint64(1)

    # Free the raw read buffer — no longer needed now that typed arrays are built.
    # This drops ~36 MB before the HDF5 write, keeping peak RSS much lower.
    del raw, particles
    gc.collect()

    N     = ndark
    ntypes = 2
    npart  = np.zeros(ntypes, dtype=np.uint32)
    npart[1] = N

    # Write to a temp file first so a crash mid-write never leaves a corrupt HDF5
    tmp_path = hdf5_path.with_suffix(".tmp")
    try:
        with h5py.File(tmp_path, "w") as out:
            hdr = out.create_group("Header")
            hdr.attrs["NumPart_ThisFile"]        = npart
            hdr.attrs["NumPart_Total"]           = npart
            hdr.attrs["NumPart_Total_HighWord"]  = np.zeros(ntypes, dtype=np.uint32)
            hdr.attrs["MassTable"]               = np.zeros(ntypes, dtype=np.float64)
            hdr.attrs["Time"]                    = float(time)
            hdr.attrs["Redshift"]                = 0.0
            hdr.attrs["BoxSize"]                 = 0.0
            hdr.attrs["NumFilesPerSnapshot"]     = 1
            hdr.attrs["Omega0"]                  = 0.0
            hdr.attrs["OmegaLambda"]             = 0.0
            hdr.attrs["HubbleParam"]             = 1.0
            hdr.attrs["Flag_Sfr"]                = 0
            hdr.attrs["Flag_Cooling"]            = 0
            hdr.attrs["Flag_StellarAge"]         = 0
            hdr.attrs["Flag_Metals"]             = 0
            hdr.attrs["Flag_Feedback"]           = 0
            hdr.attrs["Flag_DoublePrecision"]    = 0

            grp = out.create_group("PartType1")
            grp.create_dataset("Coordinates",  data=pos)
            grp.create_dataset("Velocities",   data=vel)
            grp.create_dataset("Masses",       data=mass)
            grp.create_dataset("ParticleIDs",  data=ids)
        # Atomic rename: only visible after fully written
        tmp_path.rename(hdf5_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
