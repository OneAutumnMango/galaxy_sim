"""Write a Bonsai-compatible Tipsy V2 binary IC file from pos/vel/mass arrays."""

from __future__ import annotations

import struct
import numpy as np
from pathlib import Path

# Bonsai treats IDs >= DARKMATTERID as dark matter particles
DARKMATTERID = 3_000_000_000_000_000_000  # 3e18


def write_tipsy_ic(path: Path, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> None:
    """Write positions/velocities/masses to a Bonsai Tipsy V2 binary file.

    All particles are written as dark matter (type 1).

    Tipsy V2 layout
    ---------------
    Header (32 bytes):
        time     f64
        nbodies  i32
        ndim     i32
        nsph     i32
        ndark    i32
        nstar    i32
        version  i32   (2 for V2)

    dark_particleV2 (36 bytes each):
        mass     f32
        pos[3]   f32 x3
        vel[3]   f32 x3
        id       u64       (stored as two i32; must be >= DARKMATTERID)
    """
    N = len(pos)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = pos.astype(np.float32)
    vel = vel.astype(np.float32)
    mass = mass.astype(np.float32)

    # dark_particleV2 C layout (36 bytes, no padding):
    #   mass f32 | pos[3] f32 | vel[3] f32 | _ID[2] i32  (two i32s = one u64)
    # Using two i32s for ID avoids numpy inserting 4-byte alignment padding
    # before a u64 that would be at offset 28 (non-8-byte-aligned).
    dark_dtype = np.dtype([
        ("mass",  np.float32),
        ("pos",   np.float32, (3,)),
        ("vel",   np.float32, (3,)),
        ("id_lo", np.int32),
        ("id_hi", np.int32),
    ])
    assert dark_dtype.itemsize == 36, f"unexpected itemsize {dark_dtype.itemsize}"

    ids = np.arange(N, dtype=np.uint64) + np.uint64(DARKMATTERID)
    particles = np.empty(N, dtype=dark_dtype)
    particles["mass"]   = mass
    particles["pos"]    = pos
    particles["vel"]    = vel
    particles["id_lo"]  = (ids & np.uint64(0xFFFFFFFF)).astype(np.int32)
    particles["id_hi"]  = (ids >> np.uint64(32)).astype(np.int32)

    header = struct.pack("d6i", 0.0, N, 3, 0, N, 0, 2)

    with open(path, "wb") as f:
        f.write(header)
        f.write(particles.tobytes())
