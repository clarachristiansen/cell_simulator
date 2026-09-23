from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackingObservationModel:
    """x_hat_j(t_k) = r_j^0 + u_j(t_k) + eps_j(t_k), eps ~ N(0, sigma^2 I) (Eq. 11).

    ``rng`` is stored on the model (rather than passed per call) so that a
    whole simulation run is reproducible from one seed; pass a fresh
    :class:`numpy.random.Generator` if you want independent noise draws.
    """

    sigma_nm: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def observe(self, rest_positions: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Noisy observed positions, shape broadcast(rest_positions, displacement)."""
        noise = self.sigma_nm * self.rng.standard_normal(
            np.shape(displacement)
        )  # could set to 0 in the beginning when simulating several runs and matching to data from true parameters.
        return rest_positions + displacement + noise

    def force_estimate(self, observed_positions: np.ndarray, rest_positions: np.ndarray, k: float) -> np.ndarray:
        """The analyst's force estimate F_hat = k (x_hat - r^0) (Sec. 3).

        Its per-axis error has standard deviation k * sigma_nm.
        """
        return k * (observed_positions - rest_positions)

    def force_resolution_pN(self, k: float) -> float:
        """Per-axis standard deviation of the force estimate, k * sigma_nm.

        For the paper's defaults (k=4.8 pN/nm, sigma=2 nm) this is 9.6 pN.
        """
        return k * self.sigma_nm
