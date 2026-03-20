#!/usr/bin/env python3
"""
Merge Task 4 (Green/MC) results with Task 5 (SOR) results.

Inputs:
  out/task4_results.csv
  out/task5_sor_results.csv

Output:
  out/task5_comparison.csv
"""
from __future__ import annotations

import csv
from pathlib import Path


def load(path: Path, key_fields: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        k = tuple(r[kf] for kf in key_fields)
        out[k] = r
    return out


def main() -> None:
    base = Path("out")
    mc = load(base / "task4_results.csv", ["point", "boundary_case", "charge_case"])
    sor = load(base / "task5_sor_results.csv", ["point", "boundary_case", "charge_case"])

    out_rows = []
    for k, r_mc in mc.items():
        r_sor = sor.get(k)
        if r_sor is None:
            continue
        phi_mc = float(r_mc["phi_V"])
        sig = float(r_mc["sigma_boundary_V"])
        phi_s = float(r_sor["phi_sor_V"])
        diff = phi_s - phi_mc
        z = diff / sig if sig > 0 else float("nan")
        out_rows.append(
            {
                "point": r_mc["point"],
                "boundary_case": r_mc["boundary_case"],
                "charge_case": r_mc["charge_case"],
                "phi_mc_V": f"{phi_mc:.10g}",
                "sigma_mc_boundary_V": f"{sig:.6g}",
                "phi_sor_V": f"{phi_s:.10g}",
                "diff_sor_minus_mc_V": f"{diff:.6g}",
                "z_score_diff_over_sigma": f"{z:.3f}",
                "sor_iters": r_sor["iters"],
                "sor_max_update": r_sor["max_update"],
            }
        )

    out_path = base / "task5_comparison.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
