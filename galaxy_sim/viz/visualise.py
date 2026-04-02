"""
Galaxy sim visualiser.

Two backends:
  - vispy (default): GPU-accelerated OpenGL scatter, handles millions of points
  - matplotlib: fallback for headless/scripting use

Usage
-----
Interactive replay::

    from galaxy_sim.viz.visualise import replay
    replay("output/", backend="vispy")

Static frame PNG::

    from galaxy_sim.viz.visualise import plot_frame
    plot_frame(snapshot, out="frame.png")
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np


def _configure_vispy_backend() -> None:
    """Select GLFW as the vispy app backend, with WSL2/WSLg auto-setup.

    On WSL2 the only working windowed backend is GLFW.  When running under
    WSLg we also inject the display environment variables that are normally
    only present in VS Code-spawned terminals, so the visualiser works from
    any terminal without extra manual setup.
    """
    import os

    # ── WSL2 / WSLg environment setup ──────────────────────────────────────
    # WSLg needs several env vars to know which Wayland socket and D-Bus
    # session to use.  VS Code sets them automatically; external terminals
    # often don't.  We fill in safe defaults when they're missing.
    try:
        is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        is_wsl = False

    if is_wsl:
        uid = os.getuid()
        os.environ.setdefault("DISPLAY", ":0")
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
        os.environ.setdefault("WSL2_GUI_APPS_ENABLED", "1")
        os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS",
                              f"unix:path=/run/user/{uid}/bus")

    # ── vispy backend ───────────────────────────────────────────────────────
    import vispy
    try:
        vispy.use("glfw")
    except Exception:
        pass  # fall back to whatever vispy auto-detects

from galaxy_sim.cache.reader import Snapshot, SnapshotCache, _load_hdf5_snap
from galaxy_sim.viz.convert_snapshots import convert_directory


# ---------------------------------------------------------------------------
# Lean bounds sampler — reads ONLY coordinates, strided, no velocities
# ---------------------------------------------------------------------------

def _sample_coords(path: Path, max_samples: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays of at most *max_samples* points from *path*.

    Opens the HDF5 file once, reads only the Coordinates dataset with a
    stride so that only a small fraction of data is pulled off disk and
    no velocity data is loaded at all.
    """
    xs, ys = [], []
    with __import__('h5py').File(path, 'r') as f:
        for ptype in range(6):
            key = f"PartType{ptype}"
            if key not in f:
                continue
            grp = f[key]
            if 'Coordinates' not in grp:
                continue
            ds = grp['Coordinates']
            n = ds.shape[0]
            step = max(1, n // max_samples)
            # h5py supports basic slicing — only the strided rows cross the bus
            coords = ds[::step, :2]   # shape (n//step, 2), dtype float32/64
            xs.append(coords[:, 0])
            ys.append(coords[:, 1])
    if not xs:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Parallel frame-render worker  (module-level so it is picklable)
# ---------------------------------------------------------------------------

def _render_frame_task(
    snap_path: Path,
    out_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_points: int,
    point_size: float,
    resolution: int,
) -> Path:
    """Load one snapshot file, render it, write PNG.  Runs in a worker process."""
    snap = _load_hdf5_snap(snap_path)
    plot_frame(snap, out=out_path, max_points=max_points, xlim=xlim, ylim=ylim,
               point_size=point_size, resolution=resolution)
    return out_path


# ---------------------------------------------------------------------------
# MP4 export
# ---------------------------------------------------------------------------

def render_mp4(
    frame_dir: str | Path,
    output: str | Path = "galaxy.mp4",
    fps: float = 24.0,
    pattern: str = "frame_%04d.png",
) -> Path:
    """Combine PNG frames in *frame_dir* into an MP4 using ffmpeg.

    Parameters
    ----------
    frame_dir:  directory containing sequentially numbered PNG frames
    output:     path for the output MP4 file
    fps:        frames per second (inverse of time between frames)
    pattern:    printf-style filename pattern matching the saved frame names

    Returns
    -------
    Path to the written MP4 file.

    Raises
    ------
    RuntimeError if ffmpeg is not found or the encode fails.
    """
    frame_dir = Path(frame_dir)
    output = Path(output)

    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",          # suppress per-frame progress spam
        "-framerate", str(fps),
        "-i", str(frame_dir / pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # libx264 requires even dimensions
        "-c:v", "libx264",
        "-preset", "medium",           # 'slow' is very CPU-heavy; medium is a good balance
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    print(f"Encoding {len(frames)} frames → {output}  ({fps} fps)", flush=True)
    with subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        _, stderr_data = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}):\n{stderr_data}"
        )
    print(f"MP4 written: {output}")
    return output


# ---------------------------------------------------------------------------
# matplotlib backend (always available)
# ---------------------------------------------------------------------------

def plot_frame(
    snap: Snapshot,
    out: str | Path | None = None,
    max_points: int = 200_000,  # kept for API compatibility, no longer used
    cmap: str = "inferno",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_size: float = 0.1,   # kept for API compatibility, no longer used
    resolution: int = 1024,
) -> None:
    """Render a snapshot as a 2D density map via histogram2d + imshow.

    Much faster than scatter for large N: bins all particles into a grid in
    pure numpy, then renders a single image.  Works with millions of particles
    without any subsampling.
    """
    import matplotlib.pyplot as plt

    pos = snap.pos          # use ALL particles
    x, y = pos[:, 0], pos[:, 1]

    if xlim is None:
        xlim = (float(np.percentile(x, 1.0)), float(np.percentile(x, 99.0)))
    if ylim is None:
        ylim = (float(np.percentile(y, 1.0)), float(np.percentile(y, 99.0)))

    # 2D density grid — O(N) numpy, single imshow draw call
    H, _, _ = np.histogram2d(x, y, bins=resolution, range=[xlim, ylim])
    img = np.log1p(H.T)     # log scale reveals both dense core and outer halo

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(
        img,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(f"t = {snap.time:.3f}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# vispy backend — interactive 3D
# ---------------------------------------------------------------------------

def _replay_vispy(cache: SnapshotCache, max_points: int, point_size: float) -> None:
    _configure_vispy_backend()
    import vispy.scene as scene
    from vispy import app, color

    snaps = list(cache)
    if not snaps:
        print("No snapshots found.")
        return

    canvas = scene.SceneCanvas(
        title="Galaxy Sim", bgcolor="black", keys="interactive", show=True
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, elevation=30, azimuth=0)

    def _downsample(snap: Snapshot):
        pos = snap.pos
        vel = snap.vel
        if len(pos) > max_points:
            idx = np.random.choice(len(pos), max_points, replace=False)
            pos = pos[idx]
            vel = vel[idx]
        speed = np.linalg.norm(vel, axis=1)
        speed = (speed - speed.min()) / (np.ptp(speed) + 1e-9)
        clr = color.get_colormap("inferno").map(speed)
        return pos, clr

    pos0, clr0 = _downsample(snaps[0])
    scatter = scene.visuals.Markers()
    scatter.set_data(pos0, face_color=clr0, size=point_size, edge_width=0)
    view.add(scatter)
    view.camera.set_range(x=(pos0[:, 0].min(), pos0[:, 0].max()))

    title_text = scene.visuals.Text(
        f"t = {snaps[0].time:.3f}",
        color="white", font_size=12, parent=canvas.scene,
        pos=(70, 20),
    )

    state = {"idx": 0, "playing": True}

    def advance(_):
        if state["playing"]:
            state["idx"] = (state["idx"] + 1) % len(snaps)
            snap = snaps[state["idx"]]
            pos, clr = _downsample(snap)
            scatter.set_data(pos, face_color=clr, size=point_size, edge_width=0)
            title_text.text = f"t = {snap.time:.3f}"
            canvas.update()  # WSLg won't redraw unless explicitly requested

    @canvas.events.key_press.connect
    def on_key(event):
        if event.key == " ":
            state["playing"] = not state["playing"]
        elif event.key == "Left":
            state["playing"] = False
            state["idx"] = (state["idx"] - 1) % len(snaps)
            snap = snaps[state["idx"]]
            pos, clr = _downsample(snap)
            scatter.set_data(pos, face_color=clr, size=point_size, edge_width=0)
            title_text.text = f"t = {snap.time:.3f}"
            canvas.update()
        elif event.key == "Right":
            state["playing"] = False
            state["idx"] = (state["idx"] + 1) % len(snaps)
            snap = snaps[state["idx"]]
            pos, clr = _downsample(snap)
            scatter.set_data(pos, face_color=clr, size=point_size, edge_width=0)
            title_text.text = f"t = {snap.time:.3f}"
            canvas.update()

    timer = app.Timer(interval=0.05, connect=advance, start=True)
    print("Controls: SPACE = play/pause, LEFT/RIGHT = step, drag = rotate, scroll = zoom")
    app.run()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def replay(
    output_dir: str | Path = "output/",
    backend: Literal["vispy", "matplotlib"] = "vispy",
    max_points: int = 500_000,
    point_size: float = 1.5,
    frame_dir: str | Path | None = None,
    axis_percentile: float = 98.0,
    mp4_out: str | Path | None = None,
    fps: float = 24.0,
    n_workers: int = 0,
) -> None:
    """Replay all snapshots interactively or export as PNG frames / MP4.

    Parameters
    ----------
    output_dir:       directory containing GADGET-4 HDF5 snapshots
    backend:          'vispy' for interactive 3D, 'matplotlib' for PNG export
    max_points:       downsample to this many points for rendering
    point_size:       rendered point size (vispy)
    frame_dir:        if set, export each frame as a PNG instead of displaying
    axis_percentile:  central percentile used for axis limits (default 98)
    mp4_out:          if set, encode exported frames into an MP4 at this path
    fps:              frames per second for the MP4 (default 24)
    n_workers:        worker processes for parallel frame export (0 = auto:
                      min(4, cpu_count−1)).  Keep low to limit RAM usage.
    """
    output_dir = Path(output_dir)

    # Auto-convert any raw Bonsai snapshots that haven't been converted yet.
    raw_snaps = [
        s for s in output_dir.iterdir()
        if s.suffix != ".hdf5" and not s.name.endswith(".tmp")
        and "snapshot" in s.name
    ] if output_dir.exists() else []
    if raw_snaps:
        print(f"Found {len(raw_snaps)} unconverted raw snapshot(s) — converting before visualising…")
        convert_directory(output_dir)

    cache = SnapshotCache(output_dir)
    print(f"Found {len(cache)} snapshots in {output_dir}")

    if frame_dir:
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        # Compute global axis limits by sampling every snapshot lightly.
        # _sample_coords reads ONLY strided Coordinates from HDF5 — no
        # velocities, no full arrays — so the main process stays lean even
        # with 1000 snapshots × 1 M particles.
        LIMIT_SAMPLE = 5_000   # per snapshot; plenty for stable percentiles
        all_x, all_y = [], []
        for snap_path in cache.paths:
            x, y = _sample_coords(snap_path, LIMIT_SAMPLE)
            all_x.append(x)
            all_y.append(y)
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        lower = (100 - axis_percentile) / 2
        upper = 100 - lower
        xlim = (float(np.percentile(all_x, lower)), float(np.percentile(all_x, upper)))
        ylim = (float(np.percentile(all_y, lower)), float(np.percentile(all_y, upper)))

        # Parallel render — each worker loads, renders and writes one frame
        # independently so only n_workers snapshots are in RAM at once.
        workers = n_workers if n_workers > 0 else max(1, min(4, cpu_count() - 1))
        paths = cache.paths
        out_paths = [frame_dir / f"frame_{i:04d}.png" for i in range(len(paths))]
        futures_map = {}
        completed = 0
        print(f"Rendering {len(paths)} frames with {workers} worker(s)…")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (snap_path, out) in enumerate(zip(paths, out_paths)):
                fut = pool.submit(
                    _render_frame_task,
                    snap_path, out, xlim, ylim, max_points, point_size, 1024,
                )
                futures_map[fut] = out
            for fut in as_completed(futures_map):
                out = futures_map[fut]
                exc = fut.exception()
                if exc:
                    print(f"  ERROR rendering {out}: {exc}")
                else:
                    completed += 1
                    print(f"  wrote {out}  ({completed}/{len(paths)})")
        if mp4_out:
            render_mp4(frame_dir, output=mp4_out, fps=fps)
        return

    if backend == "vispy":
        try:
            _replay_vispy(cache, max_points, point_size)
        except ImportError:
            print("vispy not installed, falling back to matplotlib.")
            backend = "matplotlib"

    if backend == "matplotlib":
        for snap in cache:
            plot_frame(snap, max_points=max_points, point_size=point_size)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="Visualise galaxy sim snapshots")
    parser.add_argument("--snapdir", default="output/", help="Snapshot directory")
    parser.add_argument("--backend", choices=["vispy", "matplotlib"], default="vispy")
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--frames", default="frames/", help="Export frames to this directory (matplotlib only)")
    parser.add_argument("--axis-percentile", type=float, default=98.0, help="Central percentile for axis limits (default: 98.0)")
    parser.add_argument("--mp4", default=None, metavar="FILE",
                        help="After exporting frames, encode them into an MP4 at this path (requires ffmpeg)")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Frames per second for the MP4 output (default: 24)")
    parser.add_argument("--workers", type=int, default=0, metavar="N",
                        help="Worker processes for parallel PNG export (0 = auto: min(4, cpu-1))")
    args = parser.parse_args()

    # Only use frames if backend is matplotlib (or mp4 output is requested)
    frame_dir = args.frames if (args.backend == "matplotlib" or args.mp4) else None
    replay(
        output_dir=args.snapdir,
        backend=args.backend,
        max_points=args.max_points,
        point_size=args.point_size,
        frame_dir=frame_dir,
        axis_percentile=args.axis_percentile,
        mp4_out=args.mp4,
        fps=args.fps,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    _main()
