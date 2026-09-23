from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from cell_simulator.dynamics import ContractionDynamics


class Integrator(ABC):
    """A one-step method x(t) -> x(t+h), plus a loop that chains it up."""

    #: Theoretical global order of accuracy (for the diagnostics in checks.py).
    order: float

    @abstractmethod
    def step(
        self,
        dynamics: ContractionDynamics,
        t: float,
        state: np.ndarray,
        h: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Advance ``state`` at time ``t`` by one step of size ``h``."""

    def integrate(
        self,
        dynamics: ContractionDynamics,
        t0: float,
        t1: float,
        n_steps: int,
        x0: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from t0 to t1 in n_steps equal steps of size h.

        Returns ``(times, states)`` with ``times.shape == (n_steps + 1,)``
        and ``states.shape == (n_steps + 1, *x0.shape)``, inclusive of both
        endpoints.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        h = (t1 - t0) / n_steps
        times = t0 + h * np.arange(n_steps + 1)
        x0 = np.asarray(x0, dtype=float)
        states = np.empty((n_steps + 1,) + x0.shape)
        states[0] = x0
        state = x0.copy()
        for i in range(n_steps):
            state = self.step(dynamics, times[i], state, h, rng=rng)
            states[i + 1] = state
        return times, states

    def __repr__(self) -> str:
        return f"{type(self).__name__}(order={self.order})"


class RK4(Integrator):
    """Classical fourth-order Runge-Kutta. The error per step is so small
    that, for this system, the drift discussed in Sec. 2.3 is invisible."""

    order = 4.0

    def step(self, dynamics, t, state, h, rng=None):
        f = dynamics.drift
        k1 = f(t, state)
        k2 = f(t + h / 2, state + h / 2 * k1)
        k3 = f(t + h / 2, state + h / 2 * k2)
        k4 = f(t + h, state + h * k3)
        return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class EulerMaruyama(Integrator):
    """Euler-Maruyama for dX = f(t, X) dt + g(t, X) dW (strong order 1/2).

    Not used anywhere in this deterministic model -- but this is the point
    of separating ``drift`` from ``diffusion`` in
    :class:`~cell_simulator.dynamics.ContractionDynamics`: a future stochastic clock
    (state-dependent or additive noise, per the thesis's open diffusion-term
    question) only needs to implement ``diffusion()`` and can then be handed
    to this integrator, or plugged into the same
    :class:`~cell_simulator.simulator.Experiment`, without any other module changing.
    """

    order = 0.5

    def step(self, dynamics, t, state, h, rng=None):
        if rng is None:
            raise ValueError("EulerMaruyama needs a numpy random Generator: integrate(..., rng=rng).")
        g = dynamics.diffusion(t, state)
        if g is None:
            raise TypeError(
                f"{type(dynamics).__name__}.diffusion() returned None (no stochastic "
                "part). Use ForwardEuler, SymplecticEuler or RK4 for a purely "
                "deterministic ContractionDynamics."
            )
        drift_term = h * dynamics.drift(t, state)
        diffusion_term = g * np.sqrt(h) * rng.standard_normal(np.shape(state))
        return state + drift_term + diffusion_term
