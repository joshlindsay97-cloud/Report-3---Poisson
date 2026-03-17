#!/usr/bin/env python3
"""
Task 1: Deterministic relaxation / over-relaxation (SOR) Poisson solver on a 2D square grid.

Conventions
- Array phi[i, j] where i is y-index (0 bottom -> n-1 top), j is x-index (0 left -> n-1 right).
- Dirichlet boundary conditions: boundary phi values are fixed each iteration.
- Update rule follows assignment hint:
    phi'_{i,j} = ω [ f_{i,j} + 1/4*(neighbors) ] + (1-ω) phi_{i,j}
Optimal ω:
    ω = 2 / (1 + sin(pi/N))
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SolveResult:
    phi: np.ndarray
    iters: int
    converged: bool
    max_update: float
    omega: float
    h: float


BoundaryCase = Literal["all0", "all100"]


def omega_optimal(n: int) -> float:
    """Optimal SOR parameter for an N×N square grid."""
    return 2.0 / (1.0 + math.sin(math.pi / n))


def apply_boundary(phi: np.ndarray, length_m: float, bc: BoundaryCase) -> None:
    value = 0.0 if bc == "all0" else 100.0
    phi[0, :] = value
    phi[-1, :] = value
    phi[:, 0] = value
    phi[:, -1] = value


def make_charge(n: int, length_m: float, case: str) -> np.ndarray:
    """
    Create a charge density field f (C m^-2).
    """
    f = np.zeros((n, n), dtype=float)
    if case == "none":
        return f
    if case == "spike":
        # single-site spike at center (testing only)
        i = (n - 1) // 2
        j = (n - 1) // 2
        f[i, j] = 1.0
        return f
    raise ValueError(f"Unknown charge case: {case}")


def solve_poisson_sor(
    n: int,
    length_m: float,
    f: np.ndarray,
    bc: BoundaryCase,
    tol: float = 1e-8,
    max_iters: int = 200_000,
    omega: float | None = None,
    phi0: np.ndarray | None = None,
) -> SolveResult:
    """
    Solve Poisson’s equation on an N×N grid using Gauss-Seidel style SOR.

    Parameters
    ----------
    n : int
        Grid points per side, includes boundary.
    length_m : float
        Physical side length (m). Grid spacing h = length_m/(n-1).
    f : ndarray
        Source term/charge density at grid sites (shape (n,n), units C m^-2).
    bc : {"all0","all100"}
        Simple boundary condition for Task 1 demo.
    tol : float
        Convergence tolerance based on max point update.
    max_iters : int
        Max iterations.
    omega : float | None
        Relaxation parameter; if None, uses optimal formula.
    phi0 : ndarray | None
        Initial guess (optional).

    Returns
    -------
    SolveResult
    """
    if f.shape != (n, n):
        raise ValueError(f"f must have shape {(n, n)}")

    h = length_m / (n - 1)
    w = omega_optimal(n) if omega is None else float(omega)

    phi = np.zeros((n, n), dtype=float) if phi0 is None else np.array(phi0, copy=True, dtype=float)
    apply_boundary(phi, length_m, bc)

    for it in range(1, max_iters + 1):
        max_update = 0.0

        for i in range(1, n - 1):
            for j in range(1, n - 1):
                old = phi[i, j]
                nb = 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1])
                new = w * (f[i, j] + nb) + (1.0 - w) * old
                phi[i, j] = new
                diff = abs(new - old)
                if diff > max_update:
                    max_update = diff

        apply_boundary(phi, length_m, bc)

        if max_update < tol:
            return SolveResult(phi=phi, iters=it, converged=True, max_update=max_update, omega=w, h=h)

    return SolveResult(phi=phi, iters=max_iters, converged=False, max_update=max_update, omega=w, h=h)
