# galaxy_sim

Galaxy N-body simulator wrapping **GADGET-4** (the de-facto standard cosmological N-body/SPH code), with a Python interface for generating initial conditions, running simulations, and visualising results.

## Backends

### Bonsai (GPU — default)

[Bonsai2](https://github.com/treecode/Bonsai) is a Barnes-Hut tree code that runs entirely on the GPU via CUDA.  
Build once for your GPU architecture (see *Setup* below), then use `--backend bonsai` (the default).

### GADGET-4 (CPU)

- **TreePM** algorithm — O(N log N), scales to billions of particles
- **OpenMP** — parallel across all CPU cores
- **HDF5 snapshots** — easy to read in Python
- Works on **Linux** and **Windows (WSL)**

### Benchmark (100k particles, t=0.3, i7-8700K + GTX 1060 6GB)

| Backend | Wall time |
|---|---|
| Bonsai GPU (GTX 1060) | **~3.5 s** |
| GADGET-4 OMP (12 threads) | **~26 min** (estimated) |

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
# 1. Install Python deps
pip install -r requirements.txt

# 2. Run a small GPU test (1k particles, t=0.1)
python -m galaxy_sim.run --n 1000 --time 0.1 --snap-interval 0.05 --output output_test

# 3. Medium run (100k particles, t=0.3) — completes in ~3.5s on GTX 1060
python -m galaxy_sim.run --n 100000 --time 0.3 --snap-interval 0.1 --output output_100k

# 4. Large run (1M particles, t=3.0)
python -m galaxy_sim.run --n 1000000 --time 3.0 --snap-interval 0.3 --output output_1M

# 5. CPU backend (GADGET-4, 12 threads)
python -m galaxy_sim.run --backend gadget4 --omp 12 --n 10000 --time 0.1 --snap-interval 0.05 --output output_gadget4

# 6. Visualise snapshots
python -m galaxy_sim.viz.visualise --snapdir output_100k/
```

## CLI reference

```
python -m galaxy_sim.run [OPTIONS]

  --n N                    Number of particles (default: 10000)
  --mass MASS              Galaxy mass in 1e10 Msun (default: 1.0)
  --time TIME              End time in code units ~Gyr (default: 1.0)
  --snap-interval FLOAT    Snapshot cadence in code units (default: 0.1)
  --softening FLOAT        Gravitational softening in kpc (default: 0.01)
  --output DIR             Output directory (default: output/)
  --backend {bonsai,gadget4}  Simulation backend (default: bonsai)
  --bonsai-bin PATH        Path to bonsai2_slowdust binary
  --gadget4-bin PATH       Path to Gadget4 binary
  --omp N                  OpenMP threads for GADGET-4 (0 = all cores)
```

## Requirements

- Python >= 3.9 (see `requirements.txt`)
- **GPU backend**: NVIDIA GPU (CUDA), Bonsai2 built for your `sm_XX`
- **CPU backend**: gcc/g++, HDF5 (`libhdf5-dev`)

## Platform notes

- **Linux**: native build
- **Windows**: WSL2 with Ubuntu — all dependencies available via apt
