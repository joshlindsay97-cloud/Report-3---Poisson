#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "out/phi_n51_bcall100_chargenone.npz"
OUT_PATH = "out/phi_plot.png"

data = np.load(DATA_PATH)
phi = data["phi"]

# For all100 + no charge, the meaningful view is the deviation from 100 V.
dphi = phi - 100.0

# Clamp colors so tiny ~1e-8 noise doesn't look like structure.
vlim = 1e-6  # tighten/loosen as you like (e.g. 1e-7, 1e-5)

plt.figure()
plt.imshow(dphi, origin="lower", vmin=-vlim, vmax=vlim)
plt.colorbar(label="phi - 100 (V)")
plt.title(f"phi - 100 (clipped to ±{vlim:g} V)")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
print(f"saved {OUT_PATH}")

print("max |phi-100| =", float(np.max(np.abs(dphi))))
