#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import numpy as np
import matplotlib.pyplot as plt

d = np.load("out/greens.npz")

plt.figure()
plt.imshow(d["g_charge"], origin="lower")
plt.colorbar(label="G_charge")
plt.title("G_charge (visit-based)")
plt.tight_layout()
plt.savefig("out/g_charge.png", dpi=200)

plt.figure()
plt.imshow(d["g_exit_prob"], origin="lower")
plt.colorbar(label="Exit probability")
plt.title("Boundary exit probabilities")
plt.tight_layout()
plt.savefig("out/g_exit_prob.png", dpi=200)

print("saved out/g_charge.png and out/g_exit_prob.png")
