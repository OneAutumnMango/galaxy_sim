# galaxy_sim

Galaxy N-body simulator wrapping **GADGET-4** (the de-facto standard cosmological N-body/SPH code), with a Python interface for generating initial conditions, running simulations, and visualising results.

## Why GADGET-4?

- **TreePM** algorithm — O(N log N), scales to billions of particles
- **MPI + OpenMP** — parallel on any CPU cluster
- **GPU-brand-agnostic** — pure CPU by default
- **HDF5 snapshots** — easy to read in Python
- Works on **Linux** and **Windows (WSL)**

## Structure

```
galaxy_sim/
├── gadget4/            ← GADGET-4 source (git clone here)
├── configs/            ← GADGET-4 parameter files
├── ic/                 ← initial condition generators (Python + C extension)
│   ├── ic_gen.py       ← pure-python Plummer / disk IC generator
│   └── ic_writer.py    ← writes GADGET HDF5 IC files
├── sim/
│   ├── runner.py       ← launches GADGET-4, manages output dirs
│   └── params.py       ← dataclass for sim parameters
├── cache/              ← HDF5 snapshot reader / replay cache
│   └── reader.py
├── viz/
│   └── visualise.py    ← vispy real-time 3D viewer + matplotlib fallback
├── setup.py            ← builds optional C extension (fast IC gen)
├── requirements.txt
└── install_gadget4.sh  ← automated GADGET-4 build script
```

## Quick start

```bash
# 1. Install deps & build GADGET-4
./install_gadget4.sh

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run a 1M-particle galaxy sim
python -m galaxy_sim.run --n 1000000 --steps 200 --softening 0.01

# 4. Visualise cached snapshots
python -m galaxy_sim.viz.visualise --snapdir output/

python -m galaxy_sim.run --n 1000 --time 2.0 --snap-interval 0.05
python -m galaxy_sim.viz.visualise --snapdir output/ --backend matplotlib
```

## Requirements

- gcc / g++ (or clang)
- MPI (OpenMPI or MPICH)
- HDF5 (libhdf5-dev)
- Python >= 3.9
- See `requirements.txt` for Python packages

## Platform notes

- **Linux**: native build
- **Windows**: use WSL2 with Ubuntu — all dependencies available via apt
