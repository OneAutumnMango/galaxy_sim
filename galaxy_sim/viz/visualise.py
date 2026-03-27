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
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from galaxy_sim.cache.reader import Snapshot, SnapshotCache


# ---------------------------------------------------------------------------
# matplotlib backend (always available)
# ---------------------------------------------------------------------------

def plot_frame(
    snap: Snapshot,
    out: str | Path | None = None,
    max_points: int = 200_000,
    cmap: str = "inferno",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_size: float = 0.1,
) -> None:
    import matplotlib.pyplot as plt

    pos = snap.pos
    if len(pos) > max_points:
        idx = np.random.choice(len(pos), max_points, replace=False)
        pos = pos[idx]

    speed = np.linalg.norm(snap.vel[idx] if len(snap.pos) > max_points else snap.vel, axis=1)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=speed, cmap=cmap,
        s=point_size, linewidths=0, alpha=0.6,
    )
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
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
        elif event.key == "Right":
            state["playing"] = False
            state["idx"] = (state["idx"] + 1) % len(snaps)
            snap = snaps[state["idx"]]
            pos, clr = _downsample(snap)
            scatter.set_data(pos, face_color=clr, size=point_size, edge_width=0)
            title_text.text = f"t = {snap.time:.3f}"

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
) -> None:
    """Replay all snapshots interactively or export as PNG frames.

    Parameters
    ----------
    output_dir:   directory containing GADGET-4 HDF5 snapshots
    backend:      'vispy' for interactive 3D, 'matplotlib' for PNG export
    max_points:   downsample to this many points for rendering
    point_size:   rendered point size (vispy)
    frame_dir:    if set, export each frame as a PNG instead of displaying
    """
    cache = SnapshotCache(output_dir)
    print(f"Found {len(cache)} snapshots in {output_dir}")

    if frame_dir:
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        # Compute global min/max for all frames for alignment, filtering outliers
        all_x = []
        all_y = []
        for snap in cache:
            pos = snap.pos
            if len(pos) > max_points:
                idx = np.random.choice(len(pos), max_points, replace=False)
                pos = pos[idx]
            all_x.append(pos[:, 0])
            all_y.append(pos[:, 1])
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        # Use central percentile for axis limits
        lower = (100 - axis_percentile) / 2
        upper = 100 - lower
        xlim = (np.percentile(all_x, lower), np.percentile(all_x, upper))
        ylim = (np.percentile(all_y, lower), np.percentile(all_y, upper))

        for i, snap in enumerate(cache):
            out = frame_dir / f"frame_{i:04d}.png"
            plot_frame(snap, out=out, max_points=max_points, xlim=xlim, ylim=ylim, point_size=point_size)
            print(f"  wrote {out}")
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
    args = parser.parse_args()

    # Only use frames if backend is matplotlib
    frame_dir = args.frames if args.backend == "matplotlib" else None
    replay(
        output_dir=args.snapdir,
        backend=args.backend,
        max_points=args.max_points,
        point_size=args.point_size,
        frame_dir=frame_dir,
        axis_percentile=args.axis_percentile,
    )


if __name__ == "__main__":
    _main()
