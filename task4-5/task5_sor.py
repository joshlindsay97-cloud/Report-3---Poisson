#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
task4-5/task5_sor.py

Task 5: Deterministic SOR solver potentials for comparison against Green's/MC.

Outputs:
  out/task5_sor_results.csv

Run:
  python task5_sor.py --n 51 --length 1.0 --tol 1e-10 --max-iters 300000 --out out/task5_sor_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=51)
    p.add_argument("--length", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--max-iters", type=int, default=300_000)
    p.add_argument("--omega", type=float, default=None)
    p.add_argument("--out", type=str, default="out/task5_sor_results.csv")
    return p.parse_args()


def omega_optimal(n: int) -> float:
    """Approx optimal SOR parameter for an N×N square grid."""
    return 2.0 / (1.0 + math.sin(math.pi / n))


def boundary_mask(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def build_boundary_potential(n: int, length: float, case: str) -> np.ndarray:
    """
    Boundary cases (Dirichlet potentials, volts):
      A: all edges +100
      B: top/bottom +100, left/right -100
      C: top and left +200, bottom 0, right -400
    """
    phi_b = np.zeros((n, n), dtype=np.float64)
    L = float(length)
    h = L / (n - 1)

    for i in range(n):
        y = i * h
        for j in range(n):
            x = j * h
            if not (i == 0 or j == 0 or i == n - 1 or j == n - 1):
                continue

            on_bottom = (y == 0.0)
            on_top = (y == L)
            on_left = (x == 0.0)
            on_right = (x == L)

            if case == "A":
                phi_b[i, j] = 100.0
            elif case == "B":
                if on_top or on_bottom:
                    phi_b[i, j] = 100.0
                elif on_left or on_right:
                    phi_b[i, j] = -100.0
            elif case == "C":
                if on_top or on_left:
                    phi_b[i, j] = 200.0
                elif on_bottom:
                    phi_b[i, j] = 0.0
                elif on_right:
                    phi_b[i, j] = -400.0
            else:
                raise ValueError(f"Unknown boundary case: {case}")

    return phi_b


def build_charge_density(n: int, length: float, case: str) -> np.ndarray:
    """
    Charge density f[i,j] in C m^-2 (per assignment).
      0: zero everywhere
      U: 10 C total uniformly over 1 m^2 => 10 C m^-2 everywhere
      G: linear gradient top(1) -> bottom(0): f(y)=y/L
      E: exp(-10*r) centered at (L/2,L/2)
    """
    f = np.zeros((n, n), dtype=np.float64)
    L = float(length)
    h = L / (n - 1)

    if case == "0":
        return f

    if case == "U":
        f[:, :] = 10.0
        return f

    if case == "G":
        for i in range(n):
            y = i * h
            f[i, :] = y / L
        return f

    if case == "E":
        cx = 0.5 * L
        cy = 0.5 * L
        for i in range(n):
            y = i * h
            for j in range(n):
                x = j * h
                r = math.hypot(x - cx, y - cy)
                f[i, j] = math.exp(-10.0 * r)
        return f

    raise ValueError(f"Unknown charge case: {case}")


@dataclass(frozen=True)
class SORResult:
    phi: np.ndarray
    iters: int
    converged: bool
    max_update: float
    omega: float


def solve_sor(
    phi_b: np.ndarray,
    f: np.ndarray,
    *,
    length: float,
    omega: float,
    tol: float,
    max_iters: int,
) -> SORResult:
    """
    Solve discrete Poisson equation using in-place Gauss-Seidel SOR.

    Discrete stencil:
      (sum(neighbors) - 4*phi) / h^2 = f
    => phi_target = (sum(neighbors) - h^2*f) / 4

    SOR:
      phi_new = (1-omega)*phi_old + omega*phi_target
    """
    n = int(phi_b.shape[0])
    if phi_b.shape != (n, n) or f.shape != (n, n):
        raise ValueError("phi_b and f must have shape (n,n).")

    h = float(length) / (n - 1)
    bmask = boundary_mask(n)

    # Initial guess: boundary everywhere (good for Dirichlet problems)
    phi = np.array(phi_b, dtype=np.float64, copy=True)

    max_upd = float("inf")
    for it in range(1, max_iters + 1):
        max_upd = 0.0

        for i in range(1, n - 1):
            for j in range(1, n - 1):
                old = phi[i, j]
                nb_sum = phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]

                # *** KEY FIX: include h^2 scaling for Poisson ***
                phi_target = 0.25 * (nb_sum + (h * h) * f[i, j])

                new = (1.0 - omega) * old + omega * phi_target
                phi[i, j] = new

                d = abs(new - old)
                if d > max_upd:
                    max_upd = d

        # Enforce boundary exactly
        phi[bmask] = phi_b[bmask]

        if max_upd < tol:
            return SORResult(phi=phi, iters=it, converged=True, max_update=max_upd, omega=omega)

    return SORResult(phi=phi, iters=max_iters, converged=False, max_update=max_upd, omega=omega)


def idx_from_xy(n: int, length: float, x_m: float, y_m: float) -> Tuple[int, int]:
    """Map physical (x,y) in meters to nearest grid index (i,j)."""
    h = length / (n - 1)
    i = int(round(y_m / h))
    j = int(round(x_m / h))
    i = max(0, min(n - 1, i))
    j = max(0, min(n - 1, j))
    return i, j


def main() -> None:
    a = parse_args()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = float(a.omega) if a.omega is not None else omega_optimal(a.n)

    # Task 3 points in meters (for N=51, L=1.0 these are exact grid nodes)
    points_xy: Dict[str, Tuple[float, float]] = {
        "center": (0.50, 0.50),
        "corner": (0.02, 0.02),
        "face": (0.02, 0.50),
    }
    points_ij: Dict[str, Tuple[int, int]] = {
        name: idx_from_xy(a.n, a.length, x, y) for name, (x, y) in points_xy.items()
    }

    boundary_cases = {
        "A_all+100": "A",
        "B_tb+100_lr-100": "B",
        "C_top+left+200_bottom0_right-400": "C",
    }
    charge_cases = {
        "0_none": "0",
        "U_uniform_total10C": "U",
        "G_gradient_top1_bottom0": "G",
        "E_exp_center": "E",
    }

    rows = []
    for bc_name, bc_code in boundary_cases.items():
        phi_b = build_boundary_potential(a.n, a.length, bc_code)
        for ch_name, ch_code in charge_cases.items():
            f = build_charge_density(a.n, a.length, ch_code)

            res = solve_sor(
                phi_b,
                f,
                length=a.length,
                omega=w,
                tol=a.tol,
                max_iters=a.max_iters,
            )

            if not res.converged:
                print(
                    f"WARNING: not converged bc={bc_name} ch={ch_name} "
                    f"iters={res.iters} max_update={res.max_update:.3e}"
                )

            for pt, (i, j) in points_ij.items():
                rows.append(
                    {
                        "point": pt,
                        "boundary_case": bc_name,
                        "charge_case": ch_name,
                        "phi_sor_V": f"{float(res.phi[i, j]):.10g}",
                        "iters": str(res.iters),
                        "max_update": f"{res.max_update:.3e}",
                        "omega": f"{res.omega:.6g}",
                        "start_i": str(i),
                        "start_j": str(j),
                    }
                )

    with out_path.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
