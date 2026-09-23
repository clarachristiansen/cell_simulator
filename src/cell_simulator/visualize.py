from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from cell_simulator.simulator import SimulationResult

NM_PER_UM = 1000.0

COLORS = {
    "observed": "0.6",
    "truth": "black",
    "best": "C0",
    "worst": "C3",
}


def set_style() -> None:
    """Sans-serif scientific layout: no top/right spines, inward ticks, light grid."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap(phi):
    """Wrap angles to (-pi, pi]. Phase is circular: phi and phi + 2 pi are
    the same model, so they must also be the same point on the plot."""
    return np.angle(np.exp(1j * np.asarray(phi, dtype=float)))


def _pitch_um(rest_um: np.ndarray) -> float:
    """Nearest-neighbour pillar spacing."""
    d = np.linalg.norm(rest_um[:, None, :] - rest_um[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _arrow_scale(forces: np.ndarray, rest_um: np.ndarray) -> float:
    """pN per µm of arrow, chosen so the longest arrow is ~0.9 lattice spacing.

    Computed once per figure/animation and then kept fixed, so arrow length
    is comparable across frames and across panels.
    """
    f_max = float(np.max(np.linalg.norm(forces, axis=-1)))
    return f_max / (0.9 * _pitch_um(rest_um)) if f_max > 0 else 1.0


def _select_forces(result: SimulationResult, which: str) -> np.ndarray:
    if which == "forces":
        return result.forces
    if which == "observed_forces":
        if result.observed_forces is None:
            raise ValueError("result has no observed_forces (no observation model was configured)")
        return result.observed_forces
    raise ValueError(f"which must be 'forces' or 'observed_forces', got {which!r}")


def _target_forces(truth: SimulationResult) -> np.ndarray:
    """What the search fits to: the noisy observation if there is one."""
    return truth.observed_forces if truth.observed_forces is not None else truth.forces


def _param_label(result: SimulationResult) -> str:
    dyn = result.config.dynamics
    if hasattr(dyn, "omega") and hasattr(dyn, "phase"):
        return f"ω={dyn.omega:.3f}, φ={float(_wrap(dyn.phase)):+.2f}"
    return repr(dyn)


def _draw_pillars(ax: plt.Axes, result: SimulationResult) -> np.ndarray:
    """Static background: pillars under the cell black, the rest grey."""
    rest_um = result.rest_positions / NM_PER_UM
    under = result.under_cell
    ax.scatter(*rest_um[~under].T, s=8, color="0.8", zorder=1)
    ax.scatter(*rest_um[under].T, s=10, color="black", zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    return rest_um


def _radial_force(forces: np.ndarray, result: SimulationResult, pillar_idx: int) -> np.ndarray:
    """F · n_j for one pillar: the signed inward force, a single number per frame."""
    return forces[:, pillar_idx, :] @ result.n[pillar_idx]


def _per_frame_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=(1, 2))


def _check_same_times(*results: SimulationResult) -> None:
    t0 = results[0].times
    for r in results[1:]:
        if r.times.shape != t0.shape or not np.allclose(r.times, t0):
            raise ValueError("all results must share the same frame times")


# ---------------------------------------------------------------------------
# 1a. Force field on a static pillar lattice
# ---------------------------------------------------------------------------
def plot_force_field(
    result: SimulationResult,
    frame_idx: int,
    which: str = "forces",
    scale: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Pillars at their rest positions with one force arrow each, at one frame.

    ``which`` is ``"forces"`` (true) or ``"observed_forces"`` (noisy).
    Pass a fixed ``scale`` (pN per µm of arrow) when comparing several
    frames; by default it is set from the whole time series, not this frame.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    forces = _select_forces(result, which)
    rest_um = _draw_pillars(ax, result)
    if scale is None:
        scale = _arrow_scale(forces, rest_um)
    F = forces[frame_idx]
    ax.quiver(*rest_um.T, F[:, 0], F[:, 1], color="C1", angles="xy", scale_units="xy", scale=scale, zorder=2)
    ax.set_title(f"t = {result.times[frame_idx]:.2f} s, a = {result.a[frame_idx]:.2f}")
    return ax


def animate_force_field(result: SimulationResult, which: str = "forces", fps: int = 15) -> FuncAnimation:
    """Video of :func:`plot_force_field`: pillars stay put, arrows change."""
    forces = _select_forces(result, which)
    fig, ax = plt.subplots(figsize=(6, 6))
    rest_um = _draw_pillars(ax, result)
    scale = _arrow_scale(forces, rest_um)
    quiv = ax.quiver(
        *rest_um.T,
        forces[0, :, 0],
        forces[0, :, 1],
        color="C1",
        angles="xy",
        scale_units="xy",
        scale=scale,
        zorder=2,
    )
    title = ax.set_title("")
    fig.tight_layout()

    def update(k):
        quiv.set_UVC(forces[k, :, 0], forces[k, :, 1])
        title.set_text(f"t = {result.times[k]:.2f} s, a = {result.a[k]:.2f}")
        return quiv, title

    return FuncAnimation(fig, update, frames=len(result.times), interval=1000 / fps, blit=False)


# ---------------------------------------------------------------------------
# 1b. One pillar's chain from clock to path
# ---------------------------------------------------------------------------
def _pillar_chain_series(result: SimulationResult, pillar_idx: int):
    F = result.forces[:, pillar_idx, :]
    u = result.displacements[:, pillar_idx, :]
    return result.times, result.a, np.linalg.norm(F, axis=-1), u


def plot_pillar_motion(result: SimulationResult, pillar_idx: int) -> plt.Figure:
    """For one pillar: (a) clock a(t), (b) force magnitude, (c) displacement
    components, (d) path of the pillar top."""
    times, a, F_T, u = _pillar_chain_series(result, pillar_idx)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))

    axes[0].plot(times, a, color="C1")
    axes[0].set_title("(a) signal a(t)")
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].plot(times, F_T, color="C0")
    axes[1].set_title(r"(b) $|F_j|$ [pN]")

    axes[2].plot(times, u[:, 0], label=r"$u_x$")
    axes[2].plot(times, u[:, 1], label=r"$u_y$", color="C2")
    axes[2].set_title(r"(c) $u_j$ [nm]")
    axes[2].legend(frameon=False, fontsize=8)

    axes[3].plot(u[:, 0], u[:, 1], color="C3")
    axes[3].plot(u[0, 0], u[0, 1], "k+")
    axes[3].set_xlabel(r"$u_x$ [nm]")
    axes[3].set_ylabel(r"$u_y$ [nm]")
    axes[3].set_title("(d) path of the pillar top")
    axes[3].set_aspect("equal")

    for ax in axes[:3]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"pillar {pillar_idx}")
    fig.tight_layout()
    return fig


def animate_pillar_motion(result: SimulationResult, pillar_idx: int, fps: int = 20) -> FuncAnimation:
    """Video version of :func:`plot_pillar_motion`: the time series sweep a
    playhead and the path is traced out live."""
    times, a, F_T, u = _pillar_chain_series(result, pillar_idx)

    fig, (ax_a, ax_F, ax_u, ax_path) = plt.subplots(1, 4, figsize=(15, 3.2))
    ax_a.plot(times, a, color="0.8")
    ax_F.plot(times, F_T, color="0.8")
    ax_u.plot(times, u[:, 0], color="C0", alpha=0.25)
    ax_u.plot(times, u[:, 1], color="C2", alpha=0.25)

    (line_a,) = ax_a.plot([], [], color="C1")
    (line_F,) = ax_F.plot([], [], color="C0")
    (line_ux,) = ax_u.plot([], [], color="C0", label=r"$u_x$")
    (line_uy,) = ax_u.plot([], [], color="C2", label=r"$u_y$")
    (line_path,) = ax_path.plot([], [], color="C3")
    (dot_path,) = ax_path.plot([], [], "o", color="C3", ms=6)

    ax_a.set_title("(a) signal a(t)")
    ax_a.set_ylim(-0.02, 1.02)
    ax_F.set_title(r"(b) $|F_j|$ [pN]")
    ax_u.set_title(r"(c) $u_j$ [nm]")
    ax_u.legend(frameon=False, fontsize=8, loc="lower right")
    ax_path.set_title("(d) path of the pillar top")
    ax_path.set_xlabel(r"$u_x$ [nm]")
    ax_path.set_ylabel(r"$u_y$ [nm]")
    ax_path.set_aspect("equal")
    pad = 0.1 * max(np.ptp(u[:, 0]), np.ptp(u[:, 1]), 1.0)
    ax_path.set_xlim(u[:, 0].min() - pad, u[:, 0].max() + pad)
    ax_path.set_ylim(u[:, 1].min() - pad, u[:, 1].max() + pad)
    for ax in (ax_a, ax_F, ax_u):
        ax.set_xlabel("time [s]")
        ax.set_xlim(times[0], times[-1])
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    def update(k):
        line_a.set_data(times[: k + 1], a[: k + 1])
        line_F.set_data(times[: k + 1], F_T[: k + 1])
        line_ux.set_data(times[: k + 1], u[: k + 1, 0])
        line_uy.set_data(times[: k + 1], u[: k + 1, 1])
        line_path.set_data(u[: k + 1, 0], u[: k + 1, 1])
        dot_path.set_data([u[k, 0]], [u[k, 1]])
        fig.suptitle(f"pillar {pillar_idx} — t = {times[k]:.2f} s")
        return line_a, line_F, line_ux, line_uy, line_path, dot_path

    return FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)


# ---------------------------------------------------------------------------
# 2a. Where did the samples land, and how good were they?
# ---------------------------------------------------------------------------
def plot_parameter_search(
    omegas: np.ndarray,
    phis: np.ndarray,
    losses: np.ndarray,
    true_omega: float,
    true_phase: float,
    noise_floor: Optional[float] = None,
) -> plt.Figure:
    """Three views of the same samples.

    (a) every sample in (omega, phi), coloured by log loss; truth = star,
        best sample = ring.
    (b) loss vs. omega, (c) loss vs. phi -- each shows how sharply the loss
        depends on that parameter, which is what makes a parameter easy or
        hard to search for.

    ``noise_floor`` (optional) is the loss of the *true* parameters against
    the noisy target. No fit can honestly go below it; a best sample well
    below it is fitting noise.
    """
    omegas = np.asarray(omegas, dtype=float)
    phis = _wrap(phis)
    losses = np.asarray(losses, dtype=float)
    true_phase = float(_wrap(true_phase))
    best = int(np.argmin(losses))
    positive = losses[losses > 0]
    floor = positive.min() * 0.5 if positive.size else 1e-12
    shown = np.maximum(losses, floor)  # log colour scale cannot show 0
    norm = LogNorm(vmin=shown.min(), vmax=shown.max())

    fig, (ax_p, ax_w, ax_f) = plt.subplots(1, 3, figsize=(15, 4.3))

    sc = ax_p.scatter(omegas, phis, c=shown, norm=norm, cmap="viridis_r", s=18, edgecolor="none")
    ax_p.plot(true_omega, true_phase, "*", color=COLORS["truth"], ms=16, mfc="none", mew=1.5, label="truth")
    ax_p.plot(omegas[best], phis[best], "o", color=COLORS["best"], ms=12, mfc="none", mew=1.8, label="best sample")
    ax_p.set_xlabel("ω [rad/s]")
    ax_p.set_ylabel("φ [rad]")
    ax_p.set_ylim(-np.pi, np.pi)
    ax_p.set_title(f"(a) {len(losses)} samples")
    ax_p.legend(frameon=False, fontsize=8, loc="upper right")
    fig.colorbar(sc, ax=ax_p, label="loss", fraction=0.046, pad=0.04)

    for ax, x, x_true, xlabel, tag in [
        (ax_w, omegas, true_omega, "ω [rad/s]", "(b)"),
        (ax_f, phis, true_phase, "φ [rad]", "(c)"),
    ]:
        ax.scatter(x, shown, s=12, color="0.4", edgecolor="none")
        ax.plot(x[best], shown[best], "o", color=COLORS["best"], ms=9, mfc="none", mew=1.8)
        ax.axvline(x_true, color=COLORS["truth"], ls="--", lw=1, label="truth")
        if noise_floor is not None:
            ax.axhline(noise_floor, color=COLORS["observed"], ls=":", lw=1.2, label="noise floor")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("loss")
        ax.set_title(f"{tag} loss vs. {xlabel.split()[0]}")
        ax.legend(frameon=False, fontsize=8)
    ax_f.set_xlim(-np.pi, np.pi)

    fig.tight_layout()
    return fig


def plot_loss_landscape(
    omega_grid: np.ndarray,
    phi_grid: np.ndarray,
    loss_grid: np.ndarray,
    true_omega: float,
    true_phase: float,
    samples: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Loss evaluated on a dense grid, ``loss_grid.shape == (len(phi_grid), len(omega_grid))``.

    With only two parameters a grid is cheap, and it shows the whole
    landscape the random search is sampling from: how wide the basin
    around the truth is, and whether there are false minima. Overlay the
    random samples with ``samples=(omegas, phis)``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.8))
    loss_grid = np.asarray(loss_grid, dtype=float)
    positive = loss_grid[loss_grid > 0]
    floor = positive.min() * 0.5 if positive.size else 1e-12
    mesh = ax.pcolormesh(
        omega_grid,
        phi_grid,
        np.maximum(loss_grid, floor),
        norm=LogNorm(),
        cmap="viridis_r",
        shading="auto",
    )
    plt.colorbar(mesh, ax=ax, label="loss")
    if samples is not None:
        ax.scatter(samples[0], _wrap(samples[1]), s=6, color="white", alpha=0.7, edgecolor="none", label="samples")
    ax.plot(true_omega, float(_wrap(true_phase)), "*", color="red", ms=14, label="truth")
    ax.set_xlabel("ω [rad/s]")
    ax.set_ylabel("φ [rad]")
    ax.set_title("loss landscape")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor="white")
    ax.grid(False)
    return ax


# ---------------------------------------------------------------------------
# 2b. Truth vs. best vs. worst over time
# ---------------------------------------------------------------------------
def plot_fit_comparison(
    truth: SimulationResult,
    best: SimulationResult,
    worst: SimulationResult,
    pillar_idx: int,
) -> plt.Figure:
    """(a) the clock a(t), (b) one pillar's radial force F·n_j, with the
    noisy target as grey dots, (c) per-frame mean squared error against the
    target. The black "truth" curve in (c) is the noise floor over time.
    """
    _check_same_times(truth, best, worst)
    t = truth.times
    target = _target_forces(truth)
    runs = [
        ("truth", truth, truth.forces),
        ("best", best, best.forces),
        ("worst", worst, worst.forces),
    ]

    fig, (ax_a, ax_F, ax_e) = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")

    for name, res, _ in runs:
        ax_a.plot(
            t, res.a, color=COLORS[name], lw=1.6 if name == "truth" else 1.2, label=f"{name}: {_param_label(res)}"
        )
    ax_a.set_ylabel("a(t)")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_title("(a) contraction signal")
    ax_a.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    if truth.observed_forces is not None:
        ax_F.plot(
            t, _radial_force(target, truth, pillar_idx), ".", color=COLORS["observed"], ms=4, label="observed (target)"
        )
    for name, _, F in runs:
        ax_F.plot(t, _radial_force(F, truth, pillar_idx), color=COLORS[name], lw=1.4, label=name)
    ax_F.set_ylabel(r"$F\cdot n_j$ [pN]")
    ax_F.set_title(f"(b) inward force on pillar {pillar_idx} (ρ = {truth.rho[pillar_idx]:.2f})")
    ax_F.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    for name, _, F in runs:
        ax_e.plot(t, _per_frame_mse(F, target), color=COLORS[name], lw=1.2, label=name)
    ax_e.set_yscale("log")
    ax_e.set_ylabel(r"MSE [pN$^2$]")
    ax_e.set_xlabel("time [s]")
    ax_e.set_title("(c) per-frame error against the target, all pillars")
    ax_e.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


# ---------------------------------------------------------------------------
# 2c. Truth vs. best vs. worst as force fields
# ---------------------------------------------------------------------------
def _field_panels(truth, best, worst):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), layout="constrained")
    rest_um = truth.rest_positions / NM_PER_UM
    scale = _arrow_scale(truth.forces, rest_um)  # one scale for all three panels
    target = _target_forces(truth)
    specs = [("truth", truth), ("best", best), ("worst", worst)]
    quivers = []
    for ax, (name, res) in zip(axes, specs):
        _draw_pillars(ax, truth)
        q_target = ax.quiver(
            *rest_um.T,
            target[0, :, 0],
            target[0, :, 1],
            color=COLORS["observed"],
            alpha=0.6,
            angles="xy",
            scale_units="xy",
            scale=scale,
            zorder=2,
        )
        q_fit = ax.quiver(
            *rest_um.T,
            res.forces[0, :, 0],
            res.forces[0, :, 1],
            color=COLORS[name],
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.004,
            zorder=3,
        )
        ax.set_title(f"{name}\n{_param_label(res)}", fontsize=10)
        quivers.append((q_target, q_fit, res))
    for ax in axes[1:]:
        ax.set_ylabel("")
    return fig, axes, quivers, target


def plot_field_comparison(
    truth: SimulationResult,
    best: SimulationResult,
    worst: SimulationResult,
    frame_idx: int,
) -> plt.Figure:
    """Three force fields on the same pillar lattice and the same arrow
    scale. In every panel the grey arrows are the (noisy) target; the
    coloured arrows are that run's prediction."""
    _check_same_times(truth, best, worst)
    fig, _, quivers, target = _field_panels(truth, best, worst)
    for q_target, q_fit, res in quivers:
        q_target.set_UVC(target[frame_idx, :, 0], target[frame_idx, :, 1])
        q_fit.set_UVC(res.forces[frame_idx, :, 0], res.forces[frame_idx, :, 1])
    fig.suptitle(f"t = {truth.times[frame_idx]:.2f} s   (grey = target)")
    return fig


def animate_fit_comparison(
    truth: SimulationResult,
    best: SimulationResult,
    worst: SimulationResult,
    fps: int = 15,
) -> FuncAnimation:
    """Video of :func:`plot_field_comparison` over all frames."""
    _check_same_times(truth, best, worst)
    fig, _, quivers, target = _field_panels(truth, best, worst)
    suptitle = fig.suptitle("")

    def update(k):
        artists = []
        for q_target, q_fit, res in quivers:
            q_target.set_UVC(target[k, :, 0], target[k, :, 1])
            q_fit.set_UVC(res.forces[k, :, 0], res.forces[k, :, 1])
            artists += [q_target, q_fit]
        suptitle.set_text(f"t = {truth.times[k]:.2f} s   (grey = target)")
        return artists + [suptitle]

    return FuncAnimation(fig, update, frames=len(truth.times), interval=1000 / fps, blit=False)
