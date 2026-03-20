#!/usr/bin/env python3

import numpy as np

CASES = [
    ("center", "out/greens_center.npz"),
    ("corner", "out/greens_corner.npz"),
    ("face",   "out/greens_face.npz"),
]


def boundary_mask(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def main() -> None:
    for name, path in CASES:
        d = np.load(path)
        n = int(d["n"])
        start_i = int(d["start_i"])
        start_j = int(d["start_j"])
        walkers = int(d["walkers"])

        p = d["g_exit_prob"]
        sig = d["g_exit_sig"]
        m = boundary_mask(n)

        print(f"{name}: file={path}")
        print(f"  start=(i={start_i}, j={start_j}) walkers={walkers}")
        print(f"  sum(exit prob)        = {float(p.sum()):.12f}")
        print(f"  mean sigma (boundary) = {float(sig[m].mean()):.6e}")
        print(f"  max  sigma (boundary) = {float(sig[m].max()):.6e}")
        print(f"  max exit prob (bnd)   = {float(p[m].max()):.6e}")
        print("")


if __name__ == "__main__":
    main()
