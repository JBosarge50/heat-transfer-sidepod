#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import pandas as pd
from typing import Tuple, List

# ───────────────────────── 0. USER INPUTS ──────────────────────────
# Paths
csv_path = (r"C:\Users\pxzuk\OneDrive\Documents\OneDrive - Mississippi State University"
            r"\Mississippi State FSAE\FSAE 2026\Design\Powertrain\Tunes\VE Calculations"
            r"\BHP by MAP and RPM.csv")

base_out_dir = (r"C:\Users\pxzuk\OneDrive\Documents\OneDrive - Mississippi State University"
                r"\Mississippi State FSAE\FSAE 2026\Design\Powertrain\Cooling\Heat Transfer Calculations")

# (bulk core velocity [ft/s], required vehicle free-stream [ft/s]) pairs
# Edit as needed. If you want free stream = bulk velocity, set both values equal.
cases: List[Tuple[float, float]] = [
    (20.0, 20.0),
    (30.0, 30.0),
    (40.0, 40.0),
]

# Pass/Fail only assessed where post-gear vehicle speed is in [0.8, 1.2] * V_free_stream
speed_tolerance = 0.20

# ──────────────────────── 0a. INPUT CSV LOAD ───────────────────────
if not os.path.isfile(csv_path):
    print(f"ERROR: CSV not found at: {csv_path}")
    sys.exit(1)

try:
    bhp_df = pd.read_csv(csv_path)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
    print(f"ERROR: Failed to parse CSV: {e}")
    sys.exit(1)
except PermissionError as e:
    print(f"ERROR: Permission denied reading CSV: {e}")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"ERROR: CSV missing: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected failure reading CSV: {e}")
    sys.exit(1)

# ───────────────────────── USER CONSTANTS ──────────────────────────
# Coolant Flow Rate Values - Experimentally measured [CFM]
coolant_ref = {
    2000: 0.3878,
    3000: 0.4277,
    4000: 0.6685,
    5000: 0.6950,
    6000: 0.8288,
    7000: 0.9493,
    8000: 0.9892,
    9000: 1.0428,
    10000: 1.1096
}

# Operating / assumed conditions
T_c_inlet = 210.0    # F
T_air = 100.0        # F
thermal_frac = 0.32
mech_eff = 0.28
rad_angle_deg = 90
A_core_face = 0.84   # ft^2 (projected core face area)

# Driveline (for post-gear vehicle speeds)
gear_ratios = {1: 28/14, 2: 26/16, 3: 24/18, 4: 24/21, 5: 21/22}
primary_reduction = 70/29
final_drive = 2.75
tire_diam_in = 10

# ───────────────────────── 1. GEOMETRY ────────────────────────────
# Radiator Inputs
n_tube = 11
tube_h = 0.05558
tube_w = 0.006583
tube_L = 1.666
n_rows = 2
fin_d = 0.00625
fin_t = 0.000167
eta_fin = 0.88
wall_t = 0.001583
k_w = 120

# Calculated Radiator Values
A_front = A_core_face * math.sin(math.radians(rad_angle_deg))
A_flow = n_tube * tube_w * tube_h * n_rows
D_h_c = 4 * (tube_h * tube_w) / (2 * (tube_h + tube_w))
A_c_tot = n_tube * n_rows * (2 * (tube_h + tube_w) * tube_L)
A_w_tot = A_c_tot
core_W = tube_L
core_H = A_front / core_W
FPI = 1.0 / (fin_d * 12.0)
N_fins = int(FPI * core_W * 12)
core_depth = n_rows * (t_hp := tube_h) + 2*n_rows*wall_t + n_rows*fin_t
W_fin = core_depth
tube_h_o = tube_h + 2*wall_t
tube_w_o = tube_w + 2*wall_t
A_fin_tot = 2.0 * N_fins * core_H * W_fin
A_bare = n_tube * n_rows * (2 * (tube_h_o + tube_w_o) * tube_L)
A_a_tot = eta_fin * A_fin_tot + A_bare

# Single-segment model (uniform bulk velocity)
A_front_seg = A_front
A_a_seg = A_a_tot
A_c_seg = A_c_tot
A_w_seg = A_w_tot

# ───────────────────────── 2. PROPERTIES ──────────────────────────
# Coolant
rho_c = 59.8
mu_c = 0.00022
k_c = 0.37
C_p_c = 1.00
mu_c_hr = mu_c * 3600
Pr_c = C_p_c * mu_c_hr / k_c

# Air (fixed at T_air; update to temperature-dependent if desired)
rho_air = 0.0749
mu_air = 1.2e-05
k_air = 0.0137
C_p_air = 0.24
Pr_air = C_p_air * (mu_air * 3600) / k_air
D_h_air = tube_h_o

# ───────────────────────── 3. HELPERS ─────────────────────────────
rpm_ref = np.array(sorted(coolant_ref.keys()), dtype=float)
cfm_ref = np.array([coolant_ref[r] for r in rpm_ref], dtype=float)
slope_lo = (cfm_ref[1] - cfm_ref[0]) / (rpm_ref[1] - rpm_ref[0])
slope_hi = (cfm_ref[-1] - cfm_ref[-2]) / (rpm_ref[-1] - rpm_ref[-2])


def coolant_cfm(rpm: float) -> float:
    """Linear interpolate / extrapolate coolant flow [cfm] at a given rpm."""
    if rpm < rpm_ref[0]:
        return cfm_ref[0] + slope_lo * (rpm - rpm_ref[0])
    if rpm > rpm_ref[-1]:
        return cfm_ref[-1] + slope_hi * (rpm - rpm_ref[-1])
    return float(np.interp(rpm, rpm_ref, cfm_ref))


def Nu_MB_louver(Re: float, Pr: float) -> float:
    j = 0.6522 * Re**-0.5403 * (1 + 5.269e-5 * Re**1.340) * Pr**(1/3) * (1 + 0.504 / Pr**(2/3))
    return j * Re * Pr**(1/3)


def eps_crossflow_unmixed(NTU: float, C_r: float) -> float:
    if C_r == 0.0:
        return 1.0 - math.exp(-NTU)
    term = (1.0 / C_r) * (1.0 - math.exp(-C_r * (NTU**0.78))) * (NTU**0.22)
    return 1.0 - math.exp(-term)


tire_circ_ft = math.pi * (tire_diam_in / 12)


def vehicle_speed(engine_rpm: float, gear: int) -> float:
    total_ratio = gear_ratios[gear] * primary_reduction * final_drive
    wheel_rps = (engine_rpm / 60) / total_ratio
    return wheel_rps * tire_circ_ft  # ft/s


# ───────────────────────── 4. CORE CALC (filtered) ─────────────────
def run_filtered_case(v_air_bulk_fts: float, v_free_stream_fts: float) -> pd.DataFrame:
    """
    Evaluate only those MAP/RPM bins where ANY gear's vehicle speed
    lies within +/-speed_tolerance of v_free_stream_fts.
    For those bins, compute Pass/Fail and margin using the uniform bulk velocity.
    """
    v_low = (1.0 - speed_tolerance) * v_free_stream_fts
    v_high = (1.0 + speed_tolerance) * v_free_stream_fts

    rows_out = []

    for _, row in bhp_df.iterrows():
        rpm = float(row["RPM_bin"])

        # Check speed window across all gears
        speeds_by_gear = {g: vehicle_speed(rpm, g) for g in gear_ratios}
        in_window_gears = [g for g, v in speeds_by_gear.items() if v_low <= v <= v_high]
        if not in_window_gears:
            # Skip this operating point (no gear meets the free-stream window)
            continue

        # Heat load required
        bhp = float(row["average_BHP"])
        Q_req = (bhp / mech_eff) * thermal_frac * 2545  # Btu/hr

        # Coolant-side film coefficient
        cf_cfm = coolant_cfm(rpm)
        v_c = (cf_cfm / 60.0) / A_flow                  # ft/s
        Re_c = rho_c * v_c * D_h_c / mu_c
        Nu_c = 0.023 * (Re_c**0.8) * (Pr_c**0.4)
        h_c = Nu_c * k_c / D_h_c

        m_dot_c = rho_c * v_c * A_flow * 3600           # lb/hr
        C_c = m_dot_c * C_p_c

        # Air-side film coefficient (Mon-Berg louver correlation)
        v_air = float(v_air_bulk_fts)
        Re_a = rho_air * v_air * D_h_air / mu_air
        Nu_a = Nu_MB_louver(Re_a, Pr_air)
        h_a = Nu_a * k_air / D_h_air

        R_air = 1.0 / (h_a * A_a_seg)
        R_wall = wall_t / (k_w * A_w_seg)
        R_cool = 1.0 / (h_c * A_c_seg)
        UA_seg = 1.0 / (R_air + R_wall + R_cool)

        # Air heat capacity rate
        m_dot_a = rho_air * v_air * A_front_seg * 3600  # lb/hr
        C_a = m_dot_a * C_p_air

        C_min = min(C_c, C_a)
        C_r = C_min / max(C_c, C_a)
        NTU = UA_seg / C_min
        eps = eps_crossflow_unmixed(NTU, C_r)

        # Single-segment (uniform bulk velocity) heat transfer
        T_c_in = T_c_inlet
        Q_total = eps * C_min * (T_c_in - T_air)

        margin_pct = (Q_total - Q_req) / Q_req * 100.0
        result_flag = "Pass" if Q_total >= Q_req else "Fail"

        # Prepare output row
        out = [
            row["MAP_bin"],                 # MAP_bin
            rpm,                            # RPM_bin
            row["count"],                   # Count
            result_flag,                    # Result
            margin_pct,                     # Margin_pct
            v_air_bulk_fts,                 # BulkVelocity_fts (for traceability)
            v_free_stream_fts,              # FreeStream_fts
            ",".join(map(str, in_window_gears))  # GearsWithinWindow
        ]

        # Per-gear speeds (ft/s)
        for g in gear_ratios:
            out.append(speeds_by_gear[g])

        rows_out.append(tuple(out))

    cols_base = ["MAP_bin", "RPM_bin", "Count", "Result", "Margin_pct",
                 "BulkVelocity_fts", "FreeStream_fts", "GearsWithinWindow"]
    cols_speed = [f"Speed_g{g}" for g in gear_ratios]  # g = 1..5
    df_out = pd.DataFrame(rows_out, columns=cols_base + cols_speed)
    return df_out


# ───────────────────────── 5. RUN & SAVE ──────────────────────────
os.makedirs(base_out_dir, exist_ok=True)

for v_bulk, v_fs in cases:
    df = run_filtered_case(v_bulk, v_fs)
    v_label = f"vb{int(round(v_bulk)):03d}"
    fs_label = f"vf{int(round(v_fs)):03d}"
    out_path = os.path.join(base_out_dir, f"CoolingMargin_{v_label}_{fs_label}.csv")
    df.to_csv(out_path, index=False)
    kept = len(df)
    print(f"CSV written: {out_path}  (rows kept: {kept})")