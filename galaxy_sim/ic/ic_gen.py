"""
Initial condition generators for galaxy simulations.

Two-galaxy orbit mode (default):
  - Galaxy A at (0, +sep/2, 0) moving east  (+X)
  - Galaxy B at (0, −sep/2, 0) moving west  (−X)
  Velocities are orthogonal to the separation so the pair orbits the
  shared barycentre rather than colliding head-on.
  Separation = 5 visual diameters ≈ 10 × 5 × disk_scale_length.
  Orbital speed is configurable (default 40 km/s, "fairly low").

Single galaxy: use generate_single_galaxy_ic().

Profiles:
  - Bulge : Hernquist profile  (Hernquist 1990)
  - Disk  : exponential surface density + sech² vertical profile

Velocities are set via the Jeans / epicycle approximation — sufficient to
produce a quasi-stable disc for visualisation purposes.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from galaxy_sim.sim.params import SimParams


def _hernquist_sample(n: int, a: float, rng: np.random.Generator,
                       r_max_factor: float = 20.0) -> np.ndarray:
    """Sample 3-D positions from a Hernquist density profile, truncated at
    r_max = r_max_factor * a to avoid unbound tail particles."""
    r_max = r_max_factor * a
    # Fraction of mass within r_max: F(r_max) = (r_max/(r_max+a))^2
    u_max = (r_max / (r_max + a)) ** 2
    u = rng.uniform(0.0, u_max, n)
    r = a * np.sqrt(u) / (1.0 - np.sqrt(u))   # inverse CDF

    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2 * np.pi, n)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return np.stack([x, y, z], axis=1)


def _disk_sample(n: int, Rd: float, hz: float, rng: np.random.Generator) -> np.ndarray:
    """Sample 3-D positions from an exponential disk profile."""
    # Radial: inverse CDF of exponential surface density Σ ∝ exp(-R/Rd)
    # CDF: F(R) = 1 - (1 + R/Rd) exp(-R/Rd); solved numerically via rejection
    u = rng.uniform(0.0, 1.0, n)
    # approximate inverse using Newton iterations on CDF
    R = Rd * (-np.log(1.0 - u * (1.0 - np.exp(-10.0))))
    R = np.clip(R, 0.0, 30 * Rd)

    phi = rng.uniform(0.0, 2 * np.pi, n)
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    # Vertical: sech² distribution — sample via log-uniform + arctan trick
    w = rng.uniform(0.0, 1.0, n)
    z = hz * np.log(np.tan(np.pi * w / 2.0))  # inverse CDF of sech²

    return np.stack([x, y, z], axis=1)


def _circular_velocity(R: np.ndarray, M_total: float, a: float) -> np.ndarray:
    """Circular velocity for a Hernquist sphere at cylindrical radius R (kpc)."""
    G = 4.302e-3   # pc Msun⁻¹ (km/s)² → convert to kpc, 1e10 Msun
    G_code = G * 1e-3 * 1e10   # kpc (1e10 Msun)⁻¹ (km/s)²

    r = np.sqrt(R**2)
    M_enc = M_total * (r / (r + a))**2
    vc2 = G_code * M_enc / np.maximum(R, 1e-4)
    return np.sqrt(np.maximum(vc2, 0.0))


def _generate_one_galaxy(
    params: SimParams,
    rng: np.random.Generator,
    center: np.ndarray,
    bulk_vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate positions, velocities and masses for a single galaxy.

    *center* (kpc) and *bulk_vel* (km/s) are 3-vectors added on top of the
    internal structure so the galaxy can be placed arbitrarily in space.
    """
    N = params.n_particles
    N_bulge = int(N * params.bulge_fraction)
    N_disk = N - N_bulge

    M_total = params.galaxy_mass
    M_bulge = M_total * params.bulge_fraction
    M_disk = M_total - M_bulge

    # --- positions ---
    pos_bulge = _hernquist_sample(N_bulge, params.bulge_scale_radius, rng)
    pos_disk = _disk_sample(N_disk, params.disk_scale_length, params.disk_scale_height, rng)
    pos = np.vstack([pos_bulge, pos_disk]) + center  # shift to galaxy centre

    # --- velocities ---
    vel = np.zeros((N, 3), dtype=np.float64)

    # bulge: isotropic Jeans — approximate σ ≈ vc / √2
    R_bulge = np.linalg.norm(pos_bulge, axis=1)
    vc_bulge = _circular_velocity(R_bulge, M_total, params.bulge_scale_radius)
    sigma_bulge = vc_bulge / np.sqrt(2.0)
    vel[:N_bulge] = rng.normal(0.0, sigma_bulge[:, None], (N_bulge, 3))

    # disk: circular rotation + small radial/vertical dispersion
    R_disk = np.sqrt(pos_disk[:, 0]**2 + pos_disk[:, 1]**2)
    vc_disk = _circular_velocity(R_disk, M_total, params.bulge_scale_radius)
    phi = np.arctan2(pos_disk[:, 1], pos_disk[:, 0])
    vx = -vc_disk * np.sin(phi)
    vy = vc_disk * np.cos(phi)
    sigma_disk = 0.1 * vc_disk
    vx += rng.normal(0.0, sigma_disk)
    vy += rng.normal(0.0, sigma_disk)
    vz = rng.normal(0.0, sigma_disk * 0.3)
    vel[N_bulge:] = np.stack([vx, vy, vz], axis=1)

    vel += bulk_vel  # add centre-of-mass velocity

    # --- masses ---
    mass_bulge = np.full(N_bulge, M_bulge / max(N_bulge, 1))
    mass_disk = np.full(N_disk, M_disk / max(N_disk, 1))
    mass = np.concatenate([mass_bulge, mass_disk])

    return pos.astype(np.float32), vel.astype(np.float32), mass.astype(np.float32)


def generate_galaxy_ic(
    params: SimParams, seed: int = 42,
    approach_speed: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions (N*2,3), velocities (N*2,3), masses (N*2,) for a
    two-galaxy mutual orbit in code units.

    Galaxy A starts at (0, +sep/2, 0) moving east (+X).
    Galaxy B starts at (0, −sep/2, 0) moving west (−X).
    Velocities are tangential so the pair spirals around the barycentre.
    Separation = 10 × 5 × disk_scale_length  (≈ 5 visual diameters).
    *approach_speed* sets the tangential orbital speed in km/s.
    """
    rng = default_rng(seed)

    # Visual diameter ≈ 2 × 5 × Rd; separation = 5 diameters
    diameter = 2.0 * 5.0 * params.disk_scale_length   # kpc
    sep = 5.0 * diameter                               # kpc between centres

    half = sep / 2.0

    # Galaxy A: north (+Y), tangential velocity east (+X)
    center_a = np.array([0.0,  half, 0.0])
    vel_a    = np.array([+approach_speed, 0.0, 0.0])

    # Galaxy B: south (−Y), tangential velocity west (−X)
    # → both galaxies orbit the barycentre counter-clockwise (viewed from +Z)
    center_b = np.array([0.0, -half, 0.0])
    vel_b    = np.array([-approach_speed, 0.0, 0.0])

    pos_a, vel_a_arr, mass_a = _generate_one_galaxy(params, rng, center_a, vel_a)
    pos_b, vel_b_arr, mass_b = _generate_one_galaxy(params, rng, center_b, vel_b)

    pos  = np.vstack([pos_a,  pos_b])
    vel  = np.vstack([vel_a_arr, vel_b_arr])
    mass = np.concatenate([mass_a, mass_b])

    return pos.astype(np.float32), vel.astype(np.float32), mass.astype(np.float32)


def generate_single_galaxy_ic(
    params: SimParams, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ICs for a single isolated galaxy (original behaviour)."""
    rng = default_rng(seed)
    return _generate_one_galaxy(
        params, rng,
        center=np.zeros(3),
        bulk_vel=np.zeros(3),
    )
