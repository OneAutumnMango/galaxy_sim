import os
import shutil
import subprocess
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

from galaxy_sim.sim.params import SimParams


def _write_param_file(params: SimParams, dest: Path) -> None:
    template = params.param_template.read_text()

    # GADGET-4 appends .hdf5 automatically — strip the extension
    ic_path = params.ic_file.resolve()
    if ic_path.suffix == ".hdf5":
        ic_path = ic_path.with_suffix("")

    soft = str(params.softening)
    replacements = {
        "InitCondFile": str(ic_path),
        "OutputDir": str(params.output_dir.resolve()) + "/",
        "TimeMax": str(params.time_end),
        "TimeBetSnapshot": str(params.snap_interval),
        "SofteningComovingClass1": soft,
        "SofteningMaxPhysClass1": soft,
    }

    lines = []
    for line in template.splitlines():
        stripped = line.lstrip()
        key = stripped.split()[0] if stripped and not stripped.startswith("%") else None
        if key and key in replacements:
            lines.append(f"{key:<36}{replacements[key]}")
        else:
            lines.append(line)

    dest.write_text("\n".join(lines))


def run_simulation(params: SimParams) -> Path:
    """Generate ICs and launch the configured simulation backend.

    Returns the output directory path.
    """
    if params.backend == "bonsai":
        from galaxy_sim.sim.bonsai_runner import run_bonsai
        return run_bonsai(params)

    from galaxy_sim.ic.ic_gen import generate_galaxy_ic
    from galaxy_sim.ic.ic_writer import write_hdf5_ic

    if not params.gadget4_bin.exists():
        raise FileNotFoundError(
            f"GADGET-4 binary not found at {params.gadget4_bin}. "
            "Run ./install_gadget4.sh first."
        )

    params.output_dir.mkdir(parents=True, exist_ok=True)
    params.ic_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating ICs: {params.n_particles} particles...")
    pos, vel, mass = generate_galaxy_ic(params)
    write_hdf5_ic(params.ic_file, pos, vel, mass, params)
    print(f"ICs written to {params.ic_file}")

    param_file = params.output_dir / "run.param"
    _write_param_file(params, param_file)
    print(f"Param file written to {param_file}")

    env = os.environ.copy()
    omp_threads = params.n_omp if params.n_omp > 0 else cpu_count()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    print(f"OMP_NUM_THREADS={omp_threads}")
    cmd = [str(params.gadget4_bin.resolve()), str(param_file.resolve())]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

    return params.output_dir
