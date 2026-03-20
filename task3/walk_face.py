#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Task 3 (face point): MPI random-walk Green's function estimator.

This script is configured (by default) for the "middle of a face" start point:
    (x, y) = (0.02 m, 0.50 m)
on a 1 m × 1 m domain with N=51 (so h=0.02 m).

Run (example):
    mpiexec -n 16 python walk_face.py --walkers 200000 --out out/greens_face.npz

Optional overrides:
    --start-x / --start-y / --n / --length / --seed / --out
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from mpi4py import MPI


@dataclass(frozen=True)
class Grid:
    n: int
    length: float

    @property
    def h(self) -> float:
        return self.length / (self.n - 1)

    def is_bnd(self, i: int, j: int) -> bool:
        n = self.n
        return i == 0 or j == 0 or i == n - 1 or j == n - 1

    def ij(self, x: float, y: float) -> Tuple[int, int]:
        i = int(round(y / self.h))
        j = int(round(x / self.h))
        i = max(0, min(self.n - 1, i))
        j = max(0, min(self.n - 1, j))
        return i, j


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=51)
    p.add_argument("--length", type=float, default=1.0)

    # Face point defaults: (0.02 m, 0.50 m)
    p.add_argument("--start-x", type=float, default=0.02)
    p.add_argument("--start-y", type=float, default=0.50)

    p.add_argument("--walkers", type=int, default=200000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", type=str, default="out/greens_face.npz")
    return p.parse_args()


def split(total: int, rank: int, size: int) -> int:
    base = total // size
    rem = total % size
    return base + (1 if rank < rem else 0)


def step(i: int, j: int, rng: np.random.Generator) -> Tuple[int, int]:
    r = int(rng.integers(0, 4))
    if r == 0:
        return i + 1, j
    if r == 1:
        return i - 1, j
    if r == 2:
        return i, j + 1
    return i, j - 1


def run(grid: Grid, start: Tuple[int, int], walkers: int, rng: np.random.Generator):
    n = grid.n
    visits = np.zeros((n, n), dtype=np.int64)
    exits = np.zeros((n, n), dtype=np.int64)
    steps_total = 0

    si, sj = start
    if grid.is_bnd(si, sj):
        raise ValueError(f"Start must be interior (not on boundary). start={start}")

    for _ in range(walkers):
        i, j = si, sj
        while True:
            visits[i, j] += 1
            steps_total += 1
            i, j = step(i, j, rng)
            if grid.is_bnd(i, j):
                exits[i, j] += 1
                break

    return visits, exits, steps_total


def main() -> None:
    a = args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid = Grid(n=a.n, length=a.length)
    start = grid.ij(a.start_x, a.start_y)

    local_n = split(a.walkers, rank, size)
    rng = np.random.default_rng(a.seed + 1000003 * rank)

    v_loc, e_loc, s_loc = run(grid, start, local_n, rng)

    v_glob = np.zeros_like(v_loc)
    e_glob = np.zeros_like(e_loc)
    s_glob = comm.allreduce(s_loc, op=MPI.SUM)

    comm.Reduce(v_loc, v_glob, op=MPI.SUM, root=0)
    comm.Reduce(e_loc, e_glob, op=MPI.SUM, root=0)

    if rank != 0:
        return

    h = grid.h
    nwalk = int(a.walkers)

    g_charge = 0.25 * (h * h) * (v_glob.astype(np.float64) / nwalk)
    g_exit_prob = e_glob.astype(np.float64) / nwalk
    g_exit_sig = np.sqrt(g_exit_prob * (1.0 - g_exit_prob) / nwalk)

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        n=a.n,
        length=a.length,
        h=h,
        start_i=int(start[0]),
        start_j=int(start[1]),
        walkers=nwalk,
        steps_total=int(s_glob),
        visits=v_glob,
        exits=e_glob,
        g_charge=g_charge,
        g_exit_prob=g_exit_prob,
        g_exit_sig=g_exit_sig,
    )

    print(f"saved {a.out}")
    print(f"n={a.n} h={h:.6g} start={start} walkers={nwalk} steps_total={s_glob}")


if __name__ == "__main__":
    main()
