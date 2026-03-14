# -*- coding: utf-8 -*-
"""
Master Sidepod Optimizer (Bernoulli-only, Standalone) + Heat-Transfer Pass/Fail
-------------------------------------------------------------------------------
- Standalone: no external imports of your other scripts. All inputs are inlined here.
- Keeps full geometry/taper checks, using contraction/expansion half-angle limits and total length.
- Sweeps theta (tilt), inlet/outlet area ratios relative to radiator face area.
- Computes core face velocity with Bernoulli: u_core = V_inf * (A_in / A_core_face).
- Runs e-NTU crossflow heat-transfer per BHP bin; assigns operating points to gears by speed band.
- Ranks designs by count-weighted failure rate across all gears (tie-break: lowest theta >= theta_min).
- Generates per-gear plots for the best geometry only, plus summary CSVs and a best-design report.

Formatting mirrors your CFD script with numbered sections for clarity.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

# ───────────────────────── 0. USER INPUTS ──────────────────────────
# If you have your real BHP CSV, set its path here. Else a synthetic dataset is used.
BHP_CSV = (r"C:\Users\pxzuk\OneDrive\Documents\OneDrive - Mississippi State University"
           r"\Mississippi State FSAE\FSAE 2026\Design\Powertrain\Tunes\VE Calculations"
           r"\BHP by MAP and RPM.csv")

HERE = os.path.abspath(os.path.dirname(
    __file__)) if '__file__' in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(HERE, "integrated_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUTPUT_DIR, "integrated_sidepod_summary.csv")
BEST_REPORT_TXT = os.path.join(OUTPUT_DIR, "integrated_best_design_report.txt")
BEST_PARAMS_CSV = os.path.join(OUTPUT_DIR, "best_design_parameters.csv")

# ───────────────────────── 1. VEHICLE & SIDEPOD ────────────────────
# Vehicle Information (VERBATIM from your CFD script)
# Free-stream target speed [ft/s]
V_inf = 50.0
speed_tolerance = 0.2                          # fractional band around V_inf
gear_ratios = {1: 28/14, 2: 26/16, 3: 24/18, 4: 24/21, 5: 21/22}
primary_reduction = 70/29
final_drive = 2.75
tire_diam_in = 10.0
tire_circ_ft = math.pi * (tire_diam_in / 12.0)

# Sidepod (face areas and angle bounds as in your CFD file)
# Note: In this optimizer we parameterize A_in and A_out as ratios of A_core_face (below).
# The baseline values from your CFD file are kept for reference.
A_in_baseline = 0.55                          # [ft^2] baseline inlet face area
A_out_baseline = 0.84                         # [ft^2] baseline outlet face area
expansion_angle_deg = 7                       # downstream half-angle limit
contraction_angle_deg = 30.0                  # upstream half-angle limit

# Radiator face reference (VERBATIM constants used in your CFD script)
rad_angle_deg = 90                            # used to project A_front from A_core_face
A_core_face = 0.84                            # [ft^2] radiator face area (normal to flow)
A_front = A_core_face * math.sin(math.radians(rad_angle_deg))  # [ft^2] projected front area used on air side

# Geometry packaging constraints (from your sidepod geometry script)
L_tot_in = 24.0                               # total axial length available [in]
# Optional buffers used in your geometry calculator (kept for completeness)
center_height_buffer_in = 1.0                 # [in]
center_length_buffer_in = 4.0                 # [in]

# Optimizer search space
theta_min_deg = 45
theta_max_deg = 80.0
theta_step_deg = 2.0
# inlet/outlet area ratios relative to A_core_face
r_in_min,  r_in_max,  r_in_step = 0.35, 0.7, 0.05
r_out_min, r_out_max, r_out_step = 0.50, .9, 0.05
prefer_lowest_theta = True

# ───────────────────────── 2. RADIATOR GEOMETRY ────────────────────
# VERBATIM geometry & fin/tube data (units in ft unless stated)
n_tube = 11
tube_h = 0.05558        # ft   (about 0.667 in)
tube_w = 0.006583       # ft   (about 0.079 in)
tube_L = 1.666          # ft   core width (about 20.0 in)
n_rows = 2
fin_d = 0.00625         # ft   fin pitch (about 0.075 in) => FPI below
fin_t = 0.000167        # ft   fin thickness (about 0.002 in)
eta_fin = 0.88          # fin efficiency
wall_t = 0.001583       # ft   (about 0.019 in) wall thickness
k_w = 120.0             # Btu/(hr*ft*F) wall conductivity

# Derived: frontal area used in air-side mass flow and convection
A_flow = n_tube * tube_w * tube_h             # [ft^2] coolant flow area
D_h_c = 4.0 * (tube_h * tube_w) / (2.0 * (tube_h + tube_w)
                                   )          # [ft] coolant hydraulic diameter
tire_note = None

# Face sizing to compute air-side areas including fins
core_W = tube_L                               # [ft] face width
core_H = A_front / core_W                     # [ft] face height
FPI = 1.0 / (fin_d * 12.0)                    # fins/in
core_depth = n_rows * (tube_h + 2*wall_t + fin_t)   # [ft] flow depth across rows

# Outer tube dims for bare area (wetted perimeter on air side)
tube_h_o = tube_h + 2*wall_t
tube_w_o = tube_w + 2*wall_t

# Fin total area (both sides of each fin across face width)
N_fins = int(FPI * core_W * 12.0)
W_fin = core_depth
A_fin_tot = 2.0 * N_fins * core_H * W_fin

# Bare area (tubes external surfaces exposed to air)
A_bare = n_tube * n_rows * (2.0 * (tube_h_o + tube_w_o) * tube_L)

# Totals used in e-NTU model
A_a_tot = eta_fin * A_fin_tot + A_bare        # [ft^2] air-side area
A_c_tot = n_tube * n_rows * (2.0 * (tube_h + tube_w) * tube_L)   # [ft^2] coolant-side area
A_w_tot = A_c_tot                             # [ft^2] wall conduction area equals coolant perimeter * L

# Segmentation (keep 1 segment as in your CFD file; can increase if desired)
N_seg = 1
A_front_seg = A_front / N_seg
A_a_seg = A_a_tot / N_seg
A_c_seg = A_c_tot / N_seg
A_w_seg = A_w_tot / N_seg

# Air-side Dh
D_h_air = tube_h_o

# ───────────────────────── 3. PROPERTIES ───────────────────────────
# Coolant (VERBATIM)
rho_c = 59.8                 # lbm/ft^3
mu_c = 0.00022               # lbm/(ft*s)
k_c = 0.37                   # Btu/(hr*ft*F)
C_p_c = 1.00                 # Btu/(lbm*F)
mu_c_hr = mu_c * 3600.0
Pr_c = C_p_c * mu_c_hr / k_c

# Air (VERBATIM)
rho_air = 0.0749
mu_air = 1.2e-05
k_air = 0.0137
C_p_air = 0.24
Pr_air = C_p_air * (mu_air * 3600.0) / k_air

# Temperatures and engine heat split (VERBATIM)
T_air_F = 100.0              # F
T_c_inlet_F = 210.0          # F
thermal_frac = 0.32
mech_eff = 0.28

# Coolant map (VERBATIM from your CFD script) - assumed as volumetric flow [cfm].
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

# ───────────────────────── 4. HELPER FUNCTIONS ────────────────────
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
    """Air-side Colburn-j based correlation rearranged for Nusselt via j*Re*Pr^(1/3)."""
    j = 0.6522 * Re**-0.5403 * \
        (1 + 5.269e-5 * Re**1.340) * Pr**(1/3) * (1 + 0.504 / Pr**(2/3))
    return j * Re * Pr**(1/3)


def eps_crossflow_unmixed(NTU: float, C_r: float) -> float:
    if C_r == 0.0:
        return 1.0 - math.exp(-NTU)
    term = (1.0 / C_r) * (1.0 - math.exp(-C_r * NTU**0.78)) * NTU**0.22
    return 1.0 - math.exp(-term)


def vehicle_speed(engine_rpm: float, gear: int) -> float:
    total_ratio = gear_ratios[gear] * primary_reduction * final_drive
    wheel_rps = (engine_rpm / 60.0) / total_ratio
    return wheel_rps * tire_circ_ft     # ft/s


def projected_center_dims(theta_deg: float) -> Tuple[float, float]:
    """Center-section projected H and L in inches (only used for taper angles/packaging checks)."""
    theta = math.radians(theta_deg)
    # convert core_H(ft) to inches + buffer
    H_c_in = (A_front / core_W) * 12.0 + center_height_buffer_in
    # center length buffer only (geometry-independent here)
    L_c_in = (A_front / core_W) * 0.0 + center_length_buffer_in
    # Note: face area A_front comes from CFD constants; theta influences packaging preference but not A_front numerically here.
    return H_c_in, L_c_in


def bernoulli_core_velocity(V_free_ft_s: float, A_in_ft2: float, A_core_ft2: float) -> float:
    """u_core from Bernoulli + continuity. (Losses neglected by design request.)"""
    if A_core_ft2 <= 0.0:
        return 0.0
    return V_free_ft_s * (A_in_ft2 / A_core_ft2)


def required_axial_length(H_up_in: float, H_dn_in: float, alpha_lim_deg: float) -> float:
    """Minimum axial length [in] to transition between two heights under half-angle limit."""
    if alpha_lim_deg <= 0.0:
        return float('inf')
    return abs(H_dn_in - H_up_in) / math.tan(math.radians(alpha_lim_deg))


def compute_taper_lengths(H_in_in: float, H_c_in: float, H_out_in: float,
                          alpha_contr_deg: float, alpha_diff_deg: float,
                          L_total_in: Optional[float]) -> Tuple[float, float, bool]:
    """
    Compute upstream/downstream axial lengths either from packaging (L_total_in)
    or as the minimum required by the specific contraction/expansion angle limits.
    Returns (L_up_in, L_dn_in, feasible).
    """
    L_up_req = required_axial_length(H_in_in,  H_c_in,  alpha_contr_deg)
    L_dn_req = required_axial_length(H_c_in,   H_out_in, alpha_diff_deg)
    if L_total_in is None:
        return L_up_req, L_dn_req, True
    if L_up_req + L_dn_req <= L_total_in + 1e-9:
        extra = L_total_in - (L_up_req + L_dn_req)
        return L_up_req + 0.5*extra, L_dn_req + 0.5*extra, True
    return L_up_req, L_dn_req, False

# ───────────────────────── 5. DATA LOADING ─────────────────────────


def load_bhp_csv(path: Optional[str]) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        needed = {"RPM_bin", "average_BHP", "MAP_bin", "count"}
        if not needed.issubset(df.columns):
            missing = needed - set(df.columns)
            raise ValueError(f"BHP CSV missing columns: {missing}")
        return df.copy()

    # Synthetic fallback matching your CFD structure
    rng = np.random.default_rng(7)
    rpms = np.arange(4000, 10500, 250)
    rows = []
    for rpm in rpms:
        for map_bin in [60, 70, 80, 90, 100]:
            # nominal bhp curve
            bhp = 15 + 0.014*(rpm-4000) + 0.18*(map_bin-60)
            cnt = int(rng.integers(1, 6))
            rows.append({"RPM_bin": rpm, "average_BHP": bhp,
                        "MAP_bin": map_bin, "count": cnt})
    return pd.DataFrame(rows)

# ───────────────────────── 6. THERMAL MODEL ───────────────────────


@dataclass
class GeometryResult:
    theta_deg: float
    A_core_ft2: float
    A_in_ft2: float
    A_out_ft2: float
    H_in_in: float
    H_c_in: float
    H_out_in: float
    L_up_in: float
    L_dn_in: float
    alpha_contr_deg: float
    alpha_diff_deg: float
    ratio_A_in: float
    ratio_A_out: float


def heat_transfer_pass_for_row(u_core_ft_s: float, rpm: float, bhp: float) -> Tuple[bool, float]:
    """Return (pass, margin%) for one operating point, e-NTU crossflow model."""
    Q_req = (bhp / mech_eff) * thermal_frac * 2545.0         # Btu/hr

    # Coolant-side convection via Dittus-Boelter on hydraulic diameter
    cf_cfm = coolant_cfm(rpm)
    v_c = (cf_cfm / 60.0) / max(A_flow, 1e-9)                # ft/s
    Re_c = rho_c * v_c * D_h_c / max(mu_c, 1e-12)
    Nu_c = 0.023 * max(Re_c, 1.0)**0.8 * max(Pr_c, 1e-9)**0.4
    h_c = Nu_c * k_c / max(D_h_c, 1e-9)

    # Segment-by-segment UA and e-NTU
    T_c_in = T_c_inlet_F
    Q_total = 0.0
    for _ in range(N_seg):
        v_air = float(u_core_ft_s)
        Re_a = rho_air * v_air * max(D_h_air, 1e-9) / max(mu_air, 1e-12)
        Nu_a = Nu_MB_louver(Re_a, Pr_air)
        h_a = Nu_a * k_air / max(D_h_air, 1e-9)

        R_air = 1.0 / max(h_a * A_a_seg, 1e-12)
        R_wall = wall_t / max(k_w * A_w_seg, 1e-12)
        R_cool = 1.0 / max(h_c * A_c_seg, 1e-12)
        UA_seg = 1.0 / (R_air + R_wall + R_cool)

        m_dot_a = rho_air * v_air * A_front_seg * 3600.0     # lbm/hr
        C_a = m_dot_a * C_p_air

        m_dot_c = rho_c * v_c * A_flow * 3600.0              # lbm/hr
        C_c = m_dot_c * C_p_c

        C_min = min(C_a, C_c)
        C_max = max(C_a, C_c)
        C_r = C_min / max(C_max, 1e-12)
        NTU = UA_seg / max(C_min, 1e-12)
        eps = eps_crossflow_unmixed(NTU, C_r)

        q_seg = eps * C_min * (T_c_in - T_air_F)
        T_c_in = T_c_in - q_seg / max(C_c, 1e-12)
        Q_total += q_seg

    margin_pct = (Q_total - Q_req) / max(Q_req, 1e-12) * 100.0
    return (Q_total >= Q_req), margin_pct

# ───────────────────────── 7. EVALUATION ──────────────────────────


def evaluate_geometry(theta_deg: float, r_in: float, r_out: float, bhp_df: pd.DataFrame):
    # Areas (ft^2)
    A_core_ft2 = A_core_face
    A_in_ft2 = max(1e-9, r_in * A_core_ft2)
    A_out_ft2 = max(1e-9, r_out * A_core_ft2)

    # Heights based on constant face width core_W (ft -> in)
    H_c_in = (A_core_ft2 / core_W) * 12.0 + center_height_buffer_in
    H_in_in = (A_in_ft2 / core_W) * 12.0
    H_out_in = (A_out_ft2 / core_W) * 12.0

    # Compute required taper lengths; check packaging feasibility
    L_up_in, L_dn_in, feasible = compute_taper_lengths(H_in_in, H_c_in, H_out_in,
                                                       contraction_angle_deg, expansion_angle_deg,
                                                       L_tot_in)
    if not feasible:
        # infeasible: return "worst" failure rate for ranking, with empty per-gear arrays
        gear_results = {g: {"mask": np.array([], dtype=bool),
                            "fails": np.array([], dtype=bool),
                            "margins": np.array([], dtype=float),
                            "weights": np.array([], dtype=int)}
                        for g in gear_ratios.keys()}
        geom = GeometryResult(theta_deg, A_core_ft2, A_in_ft2, A_out_ft2,
                              H_in_in, H_c_in, H_out_in, L_up_in, L_dn_in,
                              contraction_angle_deg, expansion_angle_deg, r_in, r_out)
        return 1.0, gear_results, geom

    # Check actual angles using the assigned lengths
    alpha_contr = math.degrees(
        math.atan(abs(H_c_in - H_in_in)/max(L_up_in, 1e-9)))
    alpha_diff = math.degrees(
        math.atan(abs(H_out_in - H_c_in)/max(L_dn_in, 1e-9)))
    if alpha_contr > contraction_angle_deg + 1e-9 or alpha_diff > expansion_angle_deg + 1e-9:
        gear_results = {g: {"mask": np.array([], dtype=bool),
                            "fails": np.array([], dtype=bool),
                            "margins": np.array([], dtype=float),
                            "weights": np.array([], dtype=int)}
                        for g in gear_ratios.keys()}
        geom = GeometryResult(theta_deg, A_core_ft2, A_in_ft2, A_out_ft2,
                              H_in_in, H_c_in, H_out_in, L_up_in, L_dn_in,
                              alpha_contr, alpha_diff, r_in, r_out)
        return 1.0, gear_results, geom

    # Bernoulli core velocity
    u_core = bernoulli_core_velocity(V_inf, A_in_ft2, A_core_ft2)

    # Evaluate thermal pass/fail per BHP bin
    passes = []
    margins = []
    for _, row in bhp_df.iterrows():
        rpm = float(row["RPM_bin"])
        bhp = float(row["average_BHP"])
        ok, m = heat_transfer_pass_for_row(u_core, rpm, bhp)
        passes.append(ok)
        margins.append(m)
    passes = np.array(passes, dtype=bool)
    margins = np.array(margins, dtype=float)

    # Gear-wise weighting based on relative speed band like your CFD script
    v_low, v_hi = (1.0 - speed_tolerance) * V_inf, (1.0 + speed_tolerance) * V_inf
    failure_weight_total = 0
    weight_total = 0
    gear_results = {}
    rpms = bhp_df["RPM_bin"].to_numpy(float)
    counts = bhp_df["count"].to_numpy(int)

    for g in sorted(gear_ratios.keys()):
        speeds = np.array([vehicle_speed(rpm, g) for rpm in rpms], dtype=float)
        mask = (speeds >= v_low) & (speeds <= v_hi)
        if not np.any(mask):
            gear_results[g] = {"mask": mask, "fails": np.array([], dtype=bool),
                               "margins": np.array([], dtype=float), "weights": np.array([], dtype=int)}
            continue

        fails = (~passes) & mask
        w = counts[mask]
        failure_weight_total += w[fails[mask]].sum() if np.any(fails[mask]) else 0
        weight_total += w.sum()

        gear_results[g] = {
            "mask": mask,
            "fails": (~passes) & mask,
            "margins": margins[mask],
            "weights": counts[mask]
        }

    failure_rate_total = (failure_weight_total / weight_total) if weight_total > 0 else 1.0

    geom = GeometryResult(theta_deg, A_core_ft2, A_in_ft2, A_out_ft2,
                          H_in_in, H_c_in, H_out_in, L_up_in, L_dn_in,
                          alpha_contr, alpha_diff, r_in, r_out)
    return failure_rate_total, gear_results, geom

# ───────────────────────── 8. OPTIMIZER ───────────────────────────


def optimize(bhp_df: pd.DataFrame):
    thetas = np.arange(theta_min_deg, theta_max_deg + 1e-9, theta_step_deg)
    r_ins = np.arange(r_in_min, r_in_max + 1e-9, r_in_step)
    r_outs = np.arange(r_out_min, r_out_max + 1e-9, r_out_step)

    best_key = None
    best = None
    records = []

    for theta in thetas:
        for rin in r_ins:
            for rout in r_outs:
                failure_rate, gear_results, geom = evaluate_geometry(
                    theta, rin, rout, bhp_df)
                records.append({
                    "theta_deg": geom.theta_deg,
                    "ratio_A_in": geom.ratio_A_in,
                    "ratio_A_out": geom.ratio_A_out,
                    "failure_rate": failure_rate,
                    "alpha_contr_deg": geom.alpha_contr_deg,
                    "alpha_diff_deg": geom.alpha_diff_deg,
                    "A_core_ft2": geom.A_core_ft2,
                    "A_in_ft2": geom.A_in_ft2,
                    "A_out_ft2": geom.A_out_ft2,
                    "H_in_in": geom.H_in_in,
                    "H_c_in": geom.H_c_in,
                    "H_out_in": geom.H_out_in,
                    "L_up_in": geom.L_up_in,
                    "L_dn_in": geom.L_dn_in,
                })
                # tie-breaker: lower theta if failure_rate equal
                theta_pen = geom.theta_deg if prefer_lowest_theta else 0.0
                key = (failure_rate, theta_pen)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (failure_rate, geom, gear_results)

    df_results = pd.DataFrame.from_records(records).sort_values(
        by=["failure_rate", "theta_deg", "ratio_A_in", "ratio_A_out"],
        ascending=[True, True, True, True]
    )
    return best, df_results

# ───────────────────────── 9. PLOTTING ────────────────────────────


def plot_best_geometry(bhp_df: pd.DataFrame, best_geom: GeometryResult, best_gear_results: Dict[int, Dict[str, np.ndarray]]):
    """Generate per-gear plots similar to the CFD script for the best geometry only."""
    rpms = bhp_df["RPM_bin"].to_numpy(float)
    counts = bhp_df["count"].to_numpy(int)

    # Recompute per-bin pass/fail/margins for the best geometry to get arrays aligned with df order
    u_core = bernoulli_core_velocity(
        V_inf, best_geom.A_in_ft2, best_geom.A_core_ft2)
    pass_list, margin_list = [], []
    for rpm, bhp in zip(bhp_df["RPM_bin"].to_numpy(float), bhp_df["average_BHP"].to_numpy(float)):
        ok, m = heat_transfer_pass_for_row(u_core, rpm, bhp)
        pass_list.append(ok)
        margin_list.append(m)
    passes = np.array(pass_list, dtype=bool)
    margins = np.array(margin_list, dtype=float)

    # Global scatter: margin vs RPM (size about count), and pass/fail map vs RPM
    plt.figure()
    plt.scatter(rpms, margins, s=np.clip(counts*10, 10, 200), alpha=0.7)
    plt.xlabel("Engine RPM")
    plt.ylabel("Margin (%)")
    plt.title("Best geometry: Margin vs RPM (marker size about count)")
    plt.grid(True)
    plt.savefig(os.path.join(
        OUTPUT_DIR, "best_global_margin_vs_rpm.png"), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(rpms, (~passes).astype(int),
                s=np.clip(counts*10, 10, 200), alpha=0.7)
    plt.xlabel("Engine RPM")
    plt.ylabel("Failure (1) / Pass (0)")
    plt.title("Best geometry: Pass/Fail vs RPM (marker size about count)")
    plt.grid(True)
    plt.savefig(os.path.join(
        OUTPUT_DIR, "best_global_passfail_vs_rpm.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Per-gear failure line plot (selected by relative speed band)
    v_low, v_hi = (1.0 - speed_tolerance) * V_inf, (1.0 + speed_tolerance) * V_inf
    rpms = bhp_df["RPM_bin"].to_numpy(float)
    for g, info in best_gear_results.items():
        mask = info["mask"]
        if mask.size == 0 or not np.any(mask):
            continue
        fails = info["fails"]
        x = np.arange(np.count_nonzero(mask)) + 1
        y = fails[mask].astype(int)

        plt.figure()
        plt.title(
            f"Gear {g}: Failures (1) vs Passes (0) near V_inf={V_inf:.0f} ft/s (+/-{speed_tolerance*100:.0f}%)")
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel("Operating point index (selected by speed band)")
        plt.ylabel("Failure (1) / Pass (0)")
        plt.ylim(-0.2, 1.2)
        plt.grid(True)
        out_png = os.path.join(OUTPUT_DIR, f"gear_{g}_failures.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()

# ───────────────────────── 10. MAIN ───────────────────────────────


def main():
    bhp_df = load_bhp_csv(BHP_CSV)
    (best_failure_rate, best_geom, best_gear_results), results_df = optimize(bhp_df)

    # Save outputs
    results_df.to_csv(SUMMARY_CSV, index=False)
    with open(BEST_REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("Best design (Bernoulli-only, Standalone):\n")
        for k, v in asdict(best_geom).items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"\nTotal weighted failure rate: {best_failure_rate:.4f}\n")
    pd.DataFrame([asdict(best_geom)]).to_csv(BEST_PARAMS_CSV, index=False)

    # Console
    print("Best design parameters (Bernoulli-only, Standalone):")
    print(pd.DataFrame([asdict(best_geom)]).to_string(index=False))
    print(f"\nTotal weighted failure rate: {best_failure_rate:.4f}")
    print(f"\nSummary CSV: {SUMMARY_CSV}")
    print(f"Best design report: {BEST_REPORT_TXT}")
    print(f"Best design parameters CSV: {BEST_PARAMS_CSV}")

    # Plots for best geometry only
    plot_best_geometry(bhp_df, best_geom, best_gear_results)


if __name__ == "__main__":
    main()