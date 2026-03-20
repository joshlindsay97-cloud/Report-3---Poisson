#!/usr/bin/env python3
"""
task1/plot.py

Plot a saved Task 1 .npz file.

Behavior
--------
- If the field is nearly constant (span < 1e-6 V), saves TWO plots:
    1) Absolute phi with a fixed scale (99.9–100.1 V) so it *looks* uniform.
    2) Deviation from mean in µV to show tiny numerical noise.

- Otherwise (nontrivial field), saves one plot of absolute phi with auto scale.

Usage
-----
# Plot a specific file
python plot.py out/phi_n51_bcall100_chargenone.npz
python plot.py out/phi_n51_bcall0_chargespike.npz

# Or omit argument to pick newest out/phi_*.npz
python plot.py
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="?", default=None, help="Optional .npz path (default: newest out/phi_*.npz)")
    return p.parse_args()


def newest_npz() -> str:
    files = glob.glob("out/phi_*.npz")
    if not files:
        raise SystemExit("No out/phi_*.npz files found")
    return max(files, key=os.path.getmtime)


def save_abs_phi(path: str, phi: np.ndarray, out_png: str, *, vmin=None, vmax=None, title: str) -> None:
    plt.figure()
    plt.imshow(phi, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Potential (V)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_dev_uV(path: str, phi: np.ndarray, out_png: str, title: str) -> None:
    dev_uV = (phi - phi.mean()) * 1e6
    plt.figure()
    plt.imshow(dev_uV, origin="lower")
    plt.colorbar(label="phi - mean(phi) (µV)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    a = parse_args()
    path = a.npz or newest_npz()

    d = np.load(path)
    phi = d["phi"].astype(float)

    mn = float(phi.min())
    mx = float(phi.max())
    span = mx - mn

    base = os.path.splitext(os.path.basename(path))[0]

    print("loaded:", path)
    print("phi min/max:", mn, mx, "span:", span)

    if span < 1e-6:
        # 1) Make it visually uniform
        out1 = os.path.join("out", f"{base}_abs_fixed.png")
        save_abs_phi(
            path,
            phi,
            out1,
            vmin=99.9,
            vmax=100.1,
            title=f"{path} (abs, fixed 99.9–100.1 V)",
        )
        print("saved:", out1)

        # 2) Show the tiny numerical noise
        out2 = os.path.join("out", f"{base}_dev_uV.png")
        save_dev_uV(path, phi, out2, title=f"{path} (deviation, µV)")
        print("saved:", out2)
    else:
        out = os.path.join("out", f"{base}_abs.png")
        save_abs_phi(path, phi, out, title=f"{path} (abs)")
        print("saved:", out)


if __name__ == "__main__":
    main()
