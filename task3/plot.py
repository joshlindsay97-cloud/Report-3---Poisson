#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import numpy as np
import matplotlib.pyplot as plt

CASES = [
    ("center", "out/greens_center.npz"),
    ("corner", "out/greens_corner.npz"),
    ("face",   "out/greens_face.npz"),
]

for name, path in CASES:
    d = np.load(path)

    # Charge-related Green
    plt.figure()
    plt.imshow(d["g_charge"], origin="lower")
    plt.colorbar(label="G_charge")
    plt.title(f"G_charge (start={name})")
    plt.tight_layout()
    plt.savefig(f"out/g_charge_{name}.png", dpi=200)

    # Boundary exit probability (Laplace weighting)
    plt.figure()
    plt.imshow(d["g_exit_prob"], origin="lower")
    plt.colorbar(label="Exit probability")
    plt.title(f"Boundary exit probabilities (start={name})")
    plt.tight_layout()
    plt.savefig(f"out/g_exit_prob_{name}.png", dpi=200)

print("saved plots in out/: g_charge_*.png and g_exit_prob_*.png")
