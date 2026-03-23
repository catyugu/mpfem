#!/usr/bin/env python3
"""Compare MPFEM transient text results with COMSOL reference results."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResultRow:
    x: float
    y: float
    z: float
    v: list[float]
    t: list[float]
    disp: list[float]


def parse_comsol_combined_file(file_path: Path) -> tuple[list[ResultRow], list[float]]:
    """Parse COMSOL-style file with all time steps in columns.

    Returns (rows, times) where times is the list of time values.
    """
    rows: list[ResultRow] = []
    times: list[float] = []

    with file_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    # Find header line with field names
    data_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Header line starts with "x" or "% x" or similar
        if (
            stripped.startswith("x")
            or stripped.startswith("% x")
            or stripped.lstrip("%").strip().startswith("x")
        ):
            data_line_idx = i
            # Parse time values from header
            # Format: "... V (V) @ t=0   T (K) @ t=0   solid.disp (m) @ t=0   V (V) @ t=10   ..."
            time_pattern = re.compile(r"@ t=([0-9.]+)")
            all_times = [float(t) for t in time_pattern.findall(line)]
            # Get unique times (each time appears 3 times for V, T, disp)
            times = sorted(set(all_times))
            break

    if data_line_idx is None:
        raise ValueError("Could not find data header line")

    # Parse data lines
    for line in lines[data_line_idx + 1 :]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])

        # Each time step has 3 values (V, T, disp)
        num_time_steps = len(times)
        if len(parts) < 3 + num_time_steps * 3:
            continue

        v = []
        t = []
        disp = []
        for i in range(num_time_steps):
            idx = 3 + i * 3
            v.append(float(parts[idx]))
            t.append(float(parts[idx + 1]))
            disp.append(float(parts[idx + 2]))

        rows.append(ResultRow(x=x, y=y, z=z, v=v, t=t, disp=disp))

    return rows, times


def parse_result_file_single(file_path: Path) -> list[ResultRow]:
    """Parse a single-time-step result file (legacy format)."""
    rows: list[ResultRow] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.startswith("x") or line.startswith("X"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            rows.append(
                ResultRow(
                    x=float(parts[0]),
                    y=float(parts[1]),
                    z=float(parts[2]),
                    v=[float(parts[3])],
                    t=[float(parts[4])],
                    disp=[float(parts[5])],
                )
            )
    return rows


def compute_metrics(
    reference: list[float], current: list[float]
) -> tuple[float, float, float, float]:
    """Compute L2, Linf, max relative, and L2 relative errors."""
    if len(reference) != len(current):
        raise ValueError(f"Length mismatch: {len(reference)} vs {len(current)}")

    n = len(reference)
    if n == 0:
        raise ValueError("No data points")

    sq_sum = 0.0
    ref_sq_sum = 0.0
    max_abs = 0.0
    max_rel = 0.0
    for i in range(n):
        diff = current[i] - reference[i]
        abs_diff = abs(diff)
        sq_sum += diff * diff
        ref_sq_sum += reference[i] * reference[i]
        if abs_diff > max_abs:
            max_abs = abs_diff

        denom = max(abs(reference[i]), 1e-16)
        rel = abs_diff / denom
        if rel > max_rel:
            max_rel = rel

    l2 = math.sqrt(sq_sum / n)
    if ref_sq_sum > 0.0:
        l2_relative = math.sqrt(sq_sum / ref_sq_sum)
    else:
        l2_relative = 0.0
    return l2, max_abs, max_rel, l2_relative


def coordinate_key(x: float, y: float, z: float) -> tuple[int, int, int]:
    """Create a coordinate-based key for alignment."""
    scale = 10**12
    return (int(round(x * scale)), int(round(y * scale)), int(round(z * scale)))


def align_by_coordinates(
    reference_rows: list[ResultRow], current_rows: list[ResultRow]
) -> tuple[list[ResultRow], list[ResultRow]]:
    """Align two result sets by coordinate values."""
    reference_map: dict[tuple[int, int, int], ResultRow] = {}
    for row in reference_rows:
        key = coordinate_key(row.x, row.y, row.z)
        if key in reference_map:
            raise ValueError(
                f"Duplicate coordinate in reference: ({row.x}, {row.y}, {row.z})"
            )
        reference_map[key] = row

    current_map: dict[tuple[int, int, int], ResultRow] = {}
    for row in current_rows:
        key = coordinate_key(row.x, row.y, row.z)
        if key in current_map:
            raise ValueError(
                f"Duplicate coordinate in current: ({row.x}, {row.y}, {row.z})"
            )
        current_map[key] = row

    if len(reference_map) != len(current_map):
        raise ValueError(
            f"Point count mismatch: reference={len(reference_map)} current={len(current_map)}"
        )

    missing_in_current = [key for key in reference_map if key not in current_map]
    if missing_in_current:
        raise ValueError(
            f"Current result missing {len(missing_in_current)} reference points"
        )

    missing_in_reference = [key for key in current_map if key not in reference_map]
    if missing_in_reference:
        raise ValueError(
            f"Reference result missing {len(missing_in_reference)} current points"
        )

    ordered_keys = sorted(reference_map.keys())
    return [reference_map[key] for key in ordered_keys], [
        current_map[key] for key in ordered_keys
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MPFEM transient and COMSOL results"
    )
    parser.add_argument(
        "reference", type=Path, help="COMSOL reference result.txt (combined format)"
    )
    parser.add_argument(
        "current", type=Path, help="MPFEM result file (combined format)"
    )
    parser.add_argument(
        "--tolerance-v",
        type=float,
        default=1e-5,
        help="Tolerance for electric potential relative L2 error",
    )
    parser.add_argument(
        "--tolerance-t",
        type=float,
        default=1e-4,
        help="Tolerance for temperature relative L2 error",
    )
    parser.add_argument(
        "--tolerance-disp",
        type=float,
        default=1e-2,
        help="Tolerance for displacement relative L2 error",
    )
    args = parser.parse_args()

    # Parse both files
    ref_rows, ref_times = parse_comsol_combined_file(args.reference)
    cur_rows, cur_times = parse_comsol_combined_file(args.current)

    print(
        f"Reference: {args.reference.name} ({len(ref_rows)} points, {len(ref_times)} time steps)"
    )
    print(
        f"Current: {args.current.name} ({len(cur_rows)} points, {len(cur_times)} time steps)"
    )

    if len(ref_times) != len(cur_times):
        print(
            f"WARNING: Time step count mismatch! Reference has {len(ref_times)}, current has {len(cur_times)}"
        )
        # Use minimum
        num_steps = min(len(ref_times), len(cur_times))
        ref_times = ref_times[:num_steps]
        cur_times = cur_times[:num_steps]

    # Align by coordinates
    ref_rows, cur_rows = align_by_coordinates(ref_rows, cur_rows)

    # Header
    print(
        "\nTime Step\tV L2\t\tV L2 Rel\tT L2\t\tT L2 Rel\tDisp L2\t\tDisp L2 Rel\tStatus"
    )
    print("-" * 115)

    all_passed = True
    for i, (ref_t, cur_t) in enumerate(zip(ref_times, cur_times)):
        # Get values at this time step
        ref_v = [row.v[i] for row in ref_rows]
        cur_v = [row.v[i] for row in cur_rows]
        ref_t_vals = [row.t[i] for row in ref_rows]
        cur_t_vals = [row.t[i] for row in cur_rows]
        ref_d = [row.disp[i] for row in ref_rows]
        cur_d = [row.disp[i] for row in cur_rows]

        v_metrics = compute_metrics(ref_v, cur_v)
        t_metrics = compute_metrics(ref_t_vals, cur_t_vals)
        d_metrics = compute_metrics(ref_d, cur_d)

        v_l2, v_max, v_rel, v_l2_rel = v_metrics
        t_l2, t_max, t_rel, t_l2_rel = t_metrics
        d_l2, d_max, d_rel, d_l2_rel = d_metrics

        # Check tolerances
        # For displacement at t=0 or when reference is near zero, use absolute error
        # because relative error becomes meaningless when reference ≈ 0
        ref_d_magnitude = (
            math.sqrt(sum(d * d for d in ref_d) / len(ref_d)) if ref_d else 0.0
        )
        if ref_d_magnitude < 1e-8:
            # Reference displacement is near zero (e.g., initial condition or numerical noise)
            # Use absolute L2 error with generous tolerance
            d_ok = d_l2 < 1e-6  # Generous absolute tolerance for near-zero reference
        else:
            d_ok = d_l2_rel < args.tolerance_disp

        v_ok = v_l2_rel < args.tolerance_v
        t_ok = t_l2_rel < args.tolerance_t

        status = "PASS" if (v_ok and t_ok and d_ok) else "FAIL"
        if not (v_ok and t_ok and d_ok):
            all_passed = False

        print(
            f"t={cur_t:.0f}\t\t{v_l2_rel:.2e}\t{'OK' if v_ok else 'FAIL':s}\t{t_l2_rel:.2e}\t{'OK' if t_ok else 'FAIL':s}\t{d_l2_rel:.2e}\t{'OK' if d_ok else 'FAIL':s}\t{status}"
        )

    print("-" * 100)
    if all_passed:
        print("All time steps PASSED validation")
        return 0
    else:
        print("Some time steps FAILED validation")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
