#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""

Outputs:
  out/task4_table.csv      (rounded to 4 sig figs for phi, 2 sig figs for sigma)
  out/task4_table.md       (markdown table for report)

Run:
  python make_table.py --in out/task4_results.csv --out-dir out
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Dict


TASK4_BC_ORDER = [
    "A_all+100",
    "B_tb+100_lr-100",
    "C_top+left+200_bottom0_right-400",
]

TASK4_CHARGE_ORDER = [
    "0_none",
    "U_uniform_total10C",
    "G_gradient_top1_bottom0",
    "E_exp_center",
]

POINT_ORDER = ["center", "corner", "face"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", type=str, default="out/task4_results.csv")
    p.add_argument("--out-dir", type=str, default="out")
    return p.parse_args()


def sigfig(x: float, n: int) -> str:
    """Format x to n significant figures (handles 0)."""
    if x == 0.0:
        return "0"
    sign = "-" if x < 0 else ""
    x = abs(x)
    exp = int(math.floor(math.log10(x)))
    places = max(0, n - exp - 1)
    fmt = f"{{:{''}.{places}f}}"
    return sign + fmt.format(x)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def to_markdown(rows: List[Dict[str, str]], fieldnames: List[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in fieldnames) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(in_path)

    # Keep only Task 4 rows (your file is already Task 4, but safe)
    wanted_bc = set(TASK4_BC_ORDER)
    wanted_ch = set(TASK4_CHARGE_ORDER)
    wanted_pt = set(POINT_ORDER)

    filtered = [
        r for r in rows
        if r["boundary_case"] in wanted_bc and r["charge_case"] in wanted_ch and r["point"] in wanted_pt
    ]

    # Sort in the required, readable order
    bc_rank = {k: i for i, k in enumerate(TASK4_BC_ORDER)}
    ch_rank = {k: i for i, k in enumerate(TASK4_CHARGE_ORDER)}
    pt_rank = {k: i for i, k in enumerate(POINT_ORDER)}

    filtered.sort(key=lambda r: (bc_rank[r["boundary_case"]], ch_rank[r["charge_case"]], pt_rank[r["point"]]))

    # Format numbers
    out_rows: List[Dict[str, str]] = []
    for r in filtered:
        phi = float(r["phi_V"])
        sig = float(r["sigma_boundary_V"])
        out_rows.append(
            {
                "boundary_case": r["boundary_case"],
                "charge_case": r["charge_case"],
                "point": r["point"],
                "phi_V": sigfig(phi, 5),              # 5 sig figs (>=4)
                "sigma_boundary_V": sigfig(sig, 2),   # 2 sig figs
            }
        )

    fields = ["boundary_case", "charge_case", "point", "phi_V", "sigma_boundary_V"]

    csv_out = out_dir / "task4_table.csv"
    md_out = out_dir / "task4_table.md"

    write_csv(csv_out, out_rows, fields)
    md_out.write_text(to_markdown(out_rows, fields), encoding="utf-8")

    print(f"wrote {csv_out}")
    print(f"wrote {md_out}")


if __name__ == "__main__":
    main()
