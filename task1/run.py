#!/usr/bin/env python3
"""
Run Task 1 SOR solver.

Examples
--------
python run.py --n 51 --bc all0 --charge none
python run.py --n 101 --bc all100 --charge spike
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from solver import make_charge, solve_poisson_sor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=51)
    p.add_argument("--length", type=float, default=1.0)
    p.add_argument("--bc", choices=["all0", "all100"], default="all100")
    p.add_argument("--charge", choices=["none", "spike"], default="none")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iters", type=int, default=200_000)
    p.add_argument("--omega", type=float, default=None)
    args = p.parse_args()

    f = make_charge(args.n, args.length, args.charge)
    res = solve_poisson_sor(
        n=args.n,
        length_m=args.length,
        f=f,
        bc=args.bc,
        tol=args.tol,
        max_iters=args.max_iters,
        omega=args.omega,
    )

    i_c = (args.n - 1) // 2
    j_c = (args.n - 1) // 2

    print(f"n={args.n} h={res.h:.6g} omega={res.omega:.6g}")
    print(f"iters={res.iters} converged={res.converged} max_update={res.max_update:.3e}")
    print(f"phi(center)={res.phi[i_c, j_c]:.6g} V")

    os.makedirs("out", exist_ok=True)
    out_path = f"out/phi_n{args.n}_bc{args.bc}_charge{args.charge}.npz"
    np.savez_compressed(out_path, phi=res.phi, n=args.n, h=res.h, omega=res.omega)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
