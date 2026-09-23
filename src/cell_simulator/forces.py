from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceModel:
    """F_j(t) = F_max * rho_j * a(t) * n_j under the cell, else 0 basically says
    force points inward (n_j) and scales with the contraction signal a(t) and
    the pillar distance to the center, rho_j times the maximum force F_max,
    (F_max is the force on a pillar at the very edge of the cell at full contraction).
    OBS: So a very simple model by now.
    u_j(t) = F_j(t) / k (Hooke's law inverted, Eq. 1/10).
    """

    F_max: float  # pN, edge force at full contraction
    k: float  # pN/nm, pillar spring constant

    def force(
        self,
        a: np.ndarray,
        rho: np.ndarray,
        n: np.ndarray,
        under_cell: np.ndarray,
    ) -> np.ndarray:
        """Force field, shape (..., P, 2).

        ``a`` may have any leading shape (e.g. () for one frame, (T,) for a
        time series); ``rho`` (P,), ``n`` (P, 2) and ``under_cell`` (P,) are
        the static geometry from :class:`~tfm_sim.geometry.Cell`.
        """
        a = np.asarray(a, dtype=float)
        rho = np.asarray(rho, dtype=float)
        n = np.asarray(n, dtype=float)
        under_cell = np.asarray(under_cell, dtype=bool)

        F_T = self.F_max * a[..., None] * rho  # (..., P)
        F_T = np.where(under_cell, F_T, 0.0)
        return F_T[..., None] * n  # (..., P, 2)

    def displacement(self, F: np.ndarray) -> np.ndarray:
        """u = F / k."""
        return F / self.k
