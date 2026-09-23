from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cell_simulator.dynamics import ContractionDynamics
from cell_simulator.forces import ForceModel
from cell_simulator.geometry import Cell, PillarGrid
from cell_simulator.integrators import Integrator
from cell_simulator.observations import TrackingObservationModel


@dataclass
class ExperimentConfig:
    """Everything needed to run a forward simulation. Each field is one of
    the swappable components described in the package docstring."""

    pillar_grid: PillarGrid
    cell: Cell
    dynamics: ContractionDynamics
    integrator: Integrator
    force_model: ForceModel
    observation_model: Optional[TrackingObservationModel] = None
    rng: Optional[np.random.Generator] = None  # only used by stochastic integrators


@dataclass
class SimulationResult:
    """Everything produced by one :meth:`Experiment.run` call."""

    times: np.ndarray  # (T,) s
    states: np.ndarray  # (T, state_dim)
    a: np.ndarray  # (T,) clock signal in [0, 1]
    forces: np.ndarray  # (T, P, 2) pN, true (noise-free)
    displacements: np.ndarray  # (T, P, 2) nm, true (noise-free)
    observed_positions: Optional[np.ndarray]  # (T, P, 2) nm, or None
    observed_forces: Optional[np.ndarray]  # (T, P, 2) pN, k * (x_hat - r0), or None
    rest_positions: np.ndarray  # (P, 2) nm
    rho: np.ndarray  # (P,)
    n: np.ndarray  # (P, 2)
    under_cell: np.ndarray  # (P,) bool
    config: ExperimentConfig

    @property
    def pillar_positions(self) -> np.ndarray:
        """True (noise-free) pillar-top positions, r_j^0 + u_j(t): shape (T, P, 2)."""
        return self.rest_positions + self.displacements


class Experiment:
    """Runs :class:`ExperimentConfig` at a sequence of frame times (Algorithm 1)."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        cell, positions = config.cell, config.pillar_grid.positions
        self.rho = cell.rho(positions)  # Eq. 2, static
        self.n = cell.n(positions)  # Eq. 2, static
        self.under_cell = cell.under_cell(positions)  # Eq. 2, static

    def run(self, frame_times: np.ndarray, steps_per_frame: int = 1) -> SimulationResult:
        """Simulate at ``frame_times`` (must start at 0 and be uniformly spaced).

        ``steps_per_frame`` subdivides each inter-frame interval into finer
        integrator sub-steps; only the states at the requested frame times
        are reported. This lets a numerical integrator's accuracy be
        controlled independently of the camera's frame rate Δ.

        If an observation model is configured, the result also contains the
        noisy observed positions and the force estimate an analyst would
        compute from them, F_hat = k (x_hat - r0), using the true rest
        positions r0.
        """
        frame_times = np.asarray(frame_times, dtype=float)
        if len(frame_times) < 2:
            raise ValueError("need at least two frame times")
        if frame_times[0] != 0.0:
            raise ValueError("frame_times must start at 0.0 (ContractionDynamics.initial_state() is defined at t=0)")
        spacing = np.diff(frame_times)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("frame_times must be uniformly spaced")

        n_frames = len(frame_times) - 1
        n_steps_total = n_frames * steps_per_frame
        x0 = self.config.dynamics.initial_state()
        _, states_fine = self.config.integrator.integrate(
            self.config.dynamics,
            frame_times[0],
            frame_times[-1],
            n_steps_total,
            x0,
            rng=self.config.rng,
        )
        states = states_fine[::steps_per_frame]

        a = self.config.dynamics.contraction_signal(states)
        F = self.config.force_model.force(a, self.rho, self.n, self.under_cell)
        u = self.config.force_model.displacement(F)

        rest = self.config.pillar_grid.positions
        observed = None
        observed_F = None
        if self.config.observation_model is not None:
            observed = self.config.observation_model.observe(rest, u)
            observed_F = self.config.observation_model.force_estimate(observed, rest, self.config.force_model.k)

        return SimulationResult(
            times=frame_times,
            states=states,
            a=a,
            forces=F,
            displacements=u,
            observed_positions=observed,
            observed_forces=observed_F,
            rest_positions=rest,
            rho=self.rho,
            n=self.n,
            under_cell=self.under_cell,
            config=self.config,
        )
