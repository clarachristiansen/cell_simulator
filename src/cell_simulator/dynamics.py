from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ContractionDynamics(ABC):
    """Abstract state-space model for the contraction clock.

    A concrete subclass must define the state dimension, the drift, the
    initial state, and how a state maps to the scalar contraction signal
    a(t) in [0, 1] that :class:`~tfm_sim.forces.ForceModel` consumes. All
    other methods are optional extras that some integrators/diagnostics use
    when available, and are safely absent otherwise.
    """

    state_dim: int
    is_separable_hamiltonian: bool = False

    @abstractmethod
    def drift(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """f(t, x): the deterministic part of dx = f dt (+ g dW).

        ``state`` may carry a leading batch/time axis; the last axis is
        always the state dimension, and the return shape matches it.
        """

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """x(0), shape (state_dim,)."""

    @abstractmethod
    def contraction_signal(self, state: np.ndarray) -> np.ndarray:
        """Map a state (..., state_dim) to the scalar clock a(t) in [0, 1]."""

    def diffusion(self, t: np.ndarray, state: np.ndarray) -> Optional[np.ndarray]:
        """g(t, x): diffusion coefficient. None => purely deterministic (an ODE).

        A subclass that returns an array here (same leading shape as
        ``state``) can be integrated with :class:`~tfm_sim.integrators.EulerMaruyama`
        without touching anything else in the pipeline; that is the whole
        point of separating this from :meth:`drift`.
        """
        return None


class HarmonicOscillatorContraction(ContractionDynamics):
    """The undamped harmonic-oscillator clock of Eq. (3)-(7).

    State x = (c, z). c(t) is the contraction (+1 fully contracted, 0
    relaxed, -1 fully expanded); z is c's velocity divided by omega.
    ``contraction_signal`` returns "simulator B", a(t) = (1 + c(t)) / 2
    (Eq. 7), which is what the rest of the paper (Sec. 2.4 onward) uses.
    """

    state_dim = 2
    is_separable_hamiltonian = True

    def __init__(self, omega: float, phase: float = 0.0):
        if omega <= 0:
            raise ValueError(f"omega must be positive, got {omega}")
        self.omega = float(omega)
        self.phase = float(phase)

    def drift(self, t, state):
        c, z = state[..., 0], state[..., 1]
        return np.stack([self.omega * z, -self.omega * c], axis=-1)

    def initial_state(self):
        return np.array([np.sin(self.phase), np.cos(self.phase)])

    def contraction_signal(self, state):
        c = state[..., 0]
        return 0.5 * (1.0 + c)

    def __repr__(self) -> str:
        return f"HarmonicOscillatorContraction(omega={self.omega:.4g}, phase={self.phase:.4g})"
