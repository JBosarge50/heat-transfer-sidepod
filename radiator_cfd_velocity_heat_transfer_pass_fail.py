import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple


# Script to determine pass fail conditions at every operating point as determined by 2025 Autocross and Endurance Events.

# ───────────────────────── 0. USER INPUTS ──────────────────────────
# Load BHP data: UPDATE THIS TO REFLECT USER ON LOCAL SYSTEM, OTHERWISE FILE WILL NOT BE READ
csv_path = (r"C:\Users\pxzuk\OneDrive\Documents\OneDrive - Mississippi State University"
            r"\Mississippi State FSAE\FSAE 2026\Design\Powertrain\Tunes\VE Calculations"
            r"\BHP by MAP and RPM.csv")

# Fail fast if the file is missing
if not os.path.isfile(csv_path):
    print(f"ERROR: CSV not found at: {csv_path}")
    sys.exit(1)

# Attempt to read; exit on any read/parse issues
try:
    bhp_df = pd.read_csv(csv_path)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
    print(f"ERROR: Failed to parse CSV: {e}")
    sys.exit(1)
except PermissionError as e:
    print(f"ERROR: Permission denied reading CSV: {e}")
    sys.exit(1)
except FileNotFoundError as e:  # in case the file disappears between the check and read
    print(f"ERROR: CSV missing: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected failure reading CSV: {e}")
    sys.exit(1)


bhp_df = pd.read_csv(csv_path)


# Coolant Flow Rate Values - Experimentally measured from the car. Units: CFM
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

# Velocity profile and other constants - v_air_profile is CFD input, CHECK UNITS. The rest are assumed values or other inputs. rad_angle_deg is based on the sidepod geometry.
N_seg = 1
v_air_profile = np.array([30], dtype=float)        # ft/s
T_c_inlet = 210.0                                  # F
T_air = 100.0                                      # F
thermal_frac = 0.32
mech_eff = 0.28
rad_angle_deg = 90
A_core_face = 0.84                                 # ft^2
assert len(v_air_profile) == N_seg

# Vehicle Information
V_inf_mph = 30
V_inf = V_inf_mph * 1.4667  # Free Stream Velocity ft/s
speed_tolerance = 0.2
gear_ratios = {1: 28/14, 2: 26/16, 3: 24/18, 4: 24/21, 5: 21/22}
primary_reduction = 70/29
final_drive = 2.75
tire_diam_in = 10

# Sidepod Information
A_in = 0.55           # ft^2    inlet opening area
A_out = 0.84          # ft^2    outlet opening area
expansion_angle = 14
contraction_angle = 30

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
core_depth = n_rows * (tube_h + 2*wall_t + fin_t)
W_fin = core_depth
A_fin_tot = 2.0 * N_fins * core_H * W_fin
tube_h_o = tube_h + 2*wall_t
tube_w_o = tube_w + 2*wall_t
A_bare = n_tube * n_rows * (2 * (tube_h_o + tube_w_o) * tube_L)
A_a_tot = eta_fin * A_fin_tot + A_bare
# per-segment areas
A_front_seg = A_front / N_seg
A_a_seg = A_a_tot / N_seg
A_c_seg = A_c_tot / N_seg
A_w_seg = A_w_tot / N_seg

# ───────────────────────── 2. PROPERTIES ──────────────────────────
# Coolant
rho_c = 59.8
mu_c = 0.00022
k_c = 0.37
C_p_c = 1.00
mu_c_hr = mu_c * 3600
Pr_c = C_p_c * mu_c_hr / k_c
# Air
rho_air = 0.0749
mu_air = 1.2e-05
k_air = 0.0137
C_p_air = 0.24
Pr_air = C_p_air * (mu_air * 3600) / k_air
D_h_air = tube_h_o
# ───────────────────────── 3. HELPER FUNCTIONS ────────────────────
rpm_ref = np.array(sorted(coolant_ref.keys()), dtype=float)
cfm_ref = np.array([coolant_ref[r] for r in rpm_ref], dtype=float)
slope_lo = (cfm_ref[1] - cfm_ref[0]) / (rpm_ref[1] - rpm_ref[0])
slope_hi = (cfm_ref[-1] - cfm_ref[-2]) / (rpm_ref[-1] - rpm_ref[-2])

# UPDATE THIS TO DYNAMICALLY CALCULATE AIR PROPERTIES BASED ON TEMP
'''
def air_props_english(T_F: float = 68.0) -> Tuple[float, float]:
    """
    Air density rho [lbm/ft^3] and dynamic viscosity mu [lbm/(ft*s)]
    at 1 atm as a function of temperature in F.

    - Density: ideal-gas scaling from the ISA sea-level reference
               (rho_0 = 0.0023769 slug/ft^3 = 0.07647 lbm/ft^3 at 59 F).
    - Viscosity: Sutherland's law with
                 mu_0 = 1.20e-5 lbm/(ft*s) at 59 F (T_0 = 518.67 R),
                 S = 198.72 R (Sutherland constant for air).

    Parameters
    ----------
    T_F : float
        Temperature in degrees Fahrenheit (default 68 F).

    Returns
    -------
    rho_lbm_ft3 : float
        Density in lbm/ft^3.
    mu_lbm_fts : float
        Dynamic viscosity in lbm/(ft*s).
    """

    # --- conversion to absolute Rankine ---------------------------
    T_R   = T_F + 459.67                     # [R]
    T_ref = 518.67                           # 59 F in R

    # --- density (ideal gas, p = const = 1 atm) -------------------
    rho_ref_slug = 0.0023769                # [slug/ft^3] at T_ref
    g_c          = 32.174                   # [lbm*ft/(lbf*s^2)]
    rho_ref      = rho_ref_slug * g_c       # => 0.07647 lbm/ft^3

    rho = rho_ref * (T_ref / T_R)           # rho proportional to 1/T (ideal gas)

    # --- viscosity (Sutherland) -----------------------------------
    mu_ref = 1.20e-5                        # [lbm/(ft*s)] at T_ref
    S      = 198.72                         # Sutherland const [R]

    mu = mu_ref * (T_R / T_ref)**1.5 * (T_ref + S) / (T_R + S)

    return rho, mu

'''


def coolant_cfm(rpm: float) -> float:
    """Linear interpolate / extrapolate coolant flow [cfm] at a given rpm."""
    if rpm < rpm_ref[0]:
        return cfm_ref[0] + slope_lo * (rpm - rpm_ref[0])
    if rpm > rpm_ref[-1]:
        return cfm_ref[-1] + slope_hi * (rpm - rpm_ref[-1])
    return float(np.interp(rpm, rpm_ref, cfm_ref))


def Nu_MB_louver(Re: float, Pr: float) -> float:
    j = 0.6522 * Re**-0.5403 * \
        (1 + 5.269e-5 * Re**1.340) * Pr**(1/3) * (1 + 0.504 / Pr**(2/3))
    return j * Re * Pr**(1/3)


def eps_crossflow_unmixed(NTU: float, C_r: float) -> float:
    if C_r == 0.0:
        return 1.0 - math.exp(-NTU)
    term = (1.0 / C_r) * (1.0 - math.exp(-C_r * NTU**0.78)) * NTU**0.22
    return 1.0 - math.exp(-term)


# ───────────────────────── 4. MAIN LOOP ───────────────────────────
results = []

for _, row in bhp_df.iterrows():
    rpm = float(row["RPM_bin"])
    bhp = float(row["average_BHP"])
    map_bin = row["MAP_bin"]         # kept as label only

    Q_req = (bhp / mech_eff) * thermal_frac * 2545            # Btu/hr
    cf_cfm = coolant_cfm(rpm)

    # Coolant-side film coefficient
    v_c = (cf_cfm / 60.0) / A_flow                            # ft/s
    Re_c = rho_c * v_c * D_h_c / mu_c
    Nu_c = 0.023 * Re_c**0.8 * Pr_c**0.4
    h_c = Nu_c * k_c / D_h_c

    m_dot_c = rho_c * v_c * A_flow * 3600                     # lb/hr
    C_c = m_dot_c * C_p_c

    T_c_in = T_c_inlet
    Q_total = 0.0

    for j in range(N_seg):
        v_air = float(v_air_profile[j])

        Re_a = rho_air * v_air * D_h_air / mu_air
        Nu_a = Nu_MB_louver(Re_a, Pr_air)
        h_a = Nu_a * k_air / D_h_air

        R_air = 1.0 / (h_a * A_a_seg)
        R_wall = wall_t / (k_w * A_w_seg)
        R_cool = 1.0 / (h_c * A_c_seg)
        UA_seg = 1.0 / (R_air + R_wall + R_cool)

        m_dot_a = rho_air * v_air * A_front_seg * 3600
        C_a = m_dot_a * C_p_air

        C_min = min(C_c, C_a)
        C_r = C_min / max(C_c, C_a)
        NTU = UA_seg / C_min
        eps = eps_crossflow_unmixed(NTU, C_r)

        q_seg = eps * C_min * (T_c_in - T_air)
        T_c_in = T_c_in - q_seg / C_c
        Q_total += q_seg

    margin_pct = (Q_total - Q_req) / Q_req * 100.0
    results.append((map_bin, rpm, row["count"],
                    "Pass" if Q_total >= Q_req else "Fail",
                    margin_pct))

# ───────────────────────── 4-bis. VEHICLE SPEEDS  ─────────────────────────
tire_circ_ft = math.pi * (tire_diam_in / 12)


def vehicle_speed(engine_rpm: float, gear: int) -> float:
    total_ratio = gear_ratios[gear] * primary_reduction * final_drive
    wheel_rps = (engine_rpm / 60) / total_ratio
    return wheel_rps * tire_circ_ft     # ft/s


# add one speed column per gear
for g in gear_ratios:
    col = f"Speed_g{g}"
    for i, row in enumerate(results):
        rpm = row[1]
        speed = vehicle_speed(rpm, g)
        results[i] = (*row, speed)      # extend the tuple

# ───────────────────────── 5. REPORT ──────────────────────────────
print(f"{'MAP':>5} | {'RPM':>6} | {'Count':>4} | {'Result':>4} | {'Margin (%)':>11}")
print("-" * 49)
for m, r, c, flag, pct, *_ in results:     # "*_" swallows the speed columns
    print(f"{m:>5} | {int(r):>6} | {int(c):>4} | {flag:>4} | {pct:>11.2f}")


# ───────────────────────── 5-bis.  SAVE TABLE TO CSV  ───────────────────

cols_base = ["MAP_bin", "RPM_bin", "Count", "Result", "Margin_pct"]
cols_speed = [f"Speed_g{g}" for g in gear_ratios]      # g = 1..5
df_out = pd.DataFrame(results, columns=cols_base + cols_speed)

out_path = (r"C:\Users\pxzuk\OneDrive\Documents\OneDrive - Mississippi State University\Mississippi State FSAE\FSAE 2026\Design\Powertrain\Cooling\Heat Transfer Calculations/testingSpeed.csv")

# ensure folder exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_out.to_csv(out_path, index=False)

print(f"\nCSV written to: {out_path}")

# ───────────────────────── 6. PLOTS ───────────────────────────────
maps = [r[0] for r in results]
rpms = [r[1] for r in results]
counts = [r[2] for r in results]
flags = [r[3] for r in results]
margins = [r[4] for r in results]
speeds = {g: [r[5+g-1] for r in results] for g in gear_ratios}  # dict of lists
colors = ['green' if f == 'Pass' else 'red' for f in flags]

# 3-D scatter: RPM x MAP x Margin
fig3d = plt.figure(figsize=(7, 5))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.scatter(rpms, maps, margins, c=colors, marker='o')

# -- translucent reference plane at 0 % margin ---------------------
x_plane = [min(rpms), max(rpms)]
y_plane = [min(maps), max(maps)]
X, Y = np.meshgrid(x_plane, y_plane)
Z = np.zeros_like(X)                      # z = 0 %
ax3d.plot_surface(X, Y, Z,
                  color='lightcoral', alpha=0.18, linewidth=0, antialiased=False)

# axis labels and title
ax3d.set_xlabel("RPM")
ax3d.set_ylabel("MAP_bin")
ax3d.set_zlabel("Margin (%)")
ax3d.set_title("Cooling Margin vs RPM & MAP")

# ---------- additional 3-D plot: RPM x Margin x Count -------------
counts = [row[2] for row in results]     # z-axis values
fig_cnt = plt.figure(figsize=(7, 5))
ax_cnt = fig_cnt.add_subplot(111, projection='3d')
ax_cnt.scatter(rpms, margins, counts, c=colors, marker='o')

ax_cnt.set_xlabel("RPM")
ax_cnt.set_ylabel("Margin (%)")
ax_cnt.set_zlabel("Count")
ax_cnt.set_title("Operating Density vs RPM & Cooling Margin")

plt.show()

'''# Optional 2-D pass/fail map (comment out if not needed)
fig2d, ax2d = plt.subplots(figsize=(6, 5))
ax2d.scatter(rpms, maps, c=colors, marker='o')
ax2d.set_xlabel("RPM")
ax2d.set_ylabel("MAP_bin")
ax2d.set_title("Pass/Fail Map (green = pass, red = fail)")

plt.show()
'''

# ──────────────────────── Gear-filtered 3-D plots ────────────────────────
# Global axis limits for consistency across all gear figures
rpm_min,  rpm_max = min(rpms),  max(rpms)
map_min,  map_max = min(maps),  max(maps)
count_min, count_max = 0, max(counts)

v_low, v_hi = (1 - speed_tolerance) * V_inf, (1 + speed_tolerance) * V_inf

for g, ratio in gear_ratios.items():
    filt = [v_low <= v <= v_hi for v in speeds[g]]
    if not any(filt):
        continue

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111, projection='3d')

    # scatter points that satisfy the speed window
    ax.scatter(
        np.array(rpms)[filt],           # X : RPM
        np.array(maps)[filt],           # Y : MAP_bin
        np.array(counts)[filt],         # Z : Count
        c=np.array(colors)[filt],
        marker='o'
    )

    # fixed, dataset-wide axis limits
    ax.set_xlim(rpm_min,   rpm_max)
    ax.set_ylim(map_min,   map_max)
    ax.set_zlim(count_min, count_max)

    ax.set_xlabel("RPM")
    ax.set_ylabel("MAP_bin")
    ax.set_zlabel("Count")
    ax.set_title(
        f"Gear {g}: bins with vehicle speed {v_low:.0f}-{v_hi:.0f} ft/s")

    # -- translucent vertical plane at RPM where vehicle speed = V_inf --
    rpm_eq = V_inf * ratio * primary_reduction * final_drive * 60 / tire_circ_ft
    Yp, Zp = np.meshgrid([map_min, map_max], [count_min, count_max])
    Xp = np.full_like(Yp, rpm_eq)
    ax.plot_surface(Xp, Yp, Zp, color='lightcoral', alpha=0.18, linewidth=0)

plt.show()
