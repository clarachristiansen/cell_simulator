from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PillarGrid:
    """A set of pillar rest positions r_j^0, in nm.

    Use :meth:`square_lattice` for the paper's 15x15, 4 um pitch array;
    the ``positions`` attribute is a plain (P, 2) array so any other
    arrangement (e.g. a hexagonal lattice, or a lattice read from a real
    PillarTracker output) can be substituted just as easily.
    """

    positions: np.ndarray  # shape (P, 2), nm

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (P, 2)")

    @classmethod
    def square_lattice(
        cls, n_x: int, n_y: int, pitch_nm: float, origin_nm: tuple[float, float] = (0.0, 0.0)
    ) -> "PillarGrid":
        """A square lattice of n_x by n_y pillars with the given pitch.

        Matches the paper's "15 by 15, 4 um pitch" default.
        """
        ix, iy = np.meshgrid(np.arange(n_x), np.arange(n_y))
        xy = np.column_stack([ix.ravel(), iy.ravel()]).astype(float) * pitch_nm
        xy += np.asarray(origin_nm, dtype=float)
        return cls(positions=xy)

    @property
    def n_pillars(self) -> int:
        return self.positions.shape[0]


@dataclass(frozen=True)
class Cell:
    """A disc-shaped cell: centre c and radius R0."""

    center_nm: np.ndarray  # shape (2,)
    radius_nm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_nm", np.asarray(self.center_nm, dtype=float))
        if self.center_nm.shape != (2,):
            raise ValueError("center_nm must have shape (2,)")
        if self.radius_nm <= 0:
            raise ValueError("radius_nm must be positive")

    def vector_to_centre(self, positions: np.ndarray) -> np.ndarray:
        """c - r_j^0 for every position, shape (..., 2)."""
        return self.center_nm - positions

    def rho(self, positions: np.ndarray) -> np.ndarray:
        """Distance from the cell centre in units of the cell radius."""
        return np.linalg.norm(self.vector_to_centre(positions), axis=-1) / self.radius_nm

    def n(self, positions: np.ndarray) -> np.ndarray:
        """Unit vectors pointing from each position towards the cell centre."""
        v = self.vector_to_centre(positions)
        r = np.linalg.norm(v, axis=-1, keepdims=True)
        r = np.where(r == 0, 1.0, r)  # guard against a pillar exactly on the centre
        return v / r

    def under_cell(self, positions: np.ndarray) -> np.ndarray:
        """Boolean mask: is each position under the cell (rho <= 1)?"""
        return self.rho(positions) <= 1.0
