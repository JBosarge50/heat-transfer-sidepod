# Darcy-Forchheimer coefficients for radiator core:
# - Outputs BOTH SimScale (d, f) and Fluent (1/alpha, C2) porous media inputs.
# - Uses Ergun-based correlations with a chosen hydraulic length scale.

import math
import numpy as np

# ───────────────────────── USER INPUTS ─────────────────────────
V_inf = 58.0         # ft/s   free-stream test speed (not used directly here)
A_in = 0.55          # ft^2   inlet opening area
A_out = 0.75         # ft^2   outlet opening area
fin_pitch = 0.00625*12*25.4   # mm   (= 1.9 mm) estimate from geometry
Cd_inlet = 1.05      # external form-drag coefficient (not used here)
Cd_plate = 1.17      # vertical flat-plate Cd normal to flow (not used here)
N_seg = 1
v_air_profile = np.array([30.0], dtype=float)   # ft/s (not used here)
T_c_inlet = 210.0    # F (not used here)
T_air = 100.0        # F (not used here)
thermal_frac = 0.32  # (not used here)
mech_eff = 0.28      # (not used here)
rad_angle_deg = 60.0
A_core_face = 0.84   # ft^2
# lbm/ft^3 (used only if converting to SI; not needed for d,f from Ergun)
rho_air = 0.075
g = 32.2             # ft/s^2
assert len(v_air_profile) == N_seg

expansion_angle = 14    # deg (not used here)
contraction_angle = 30  # deg (not used here)

# Radiator core geometry (imperial -> will convert to SI)
n_tube = 11
tube_h = 0.05558     # ft
tube_w = 0.006583    # ft
tube_L = 1.666       # ft
n_rows = 2
fin_d = 0.0119       # ft   (fin pitch along flow-normal, about spacing)
fin_t = 0.000167     # ft   (fin thickness)
eta_fin = 0.88
wall_t = 0.001583    # ft   (tube wall thickness)
k_w = 120.0          # W/(m*K) (not used here)

# Air properties (only needed if converting from dp-u fits; kept for completeness)
rho_air = 0.0749     # lbm/ft^3
mu_air = 1.2e-05     # lbm/(ft*s)
k_air = 0.0137       # BTU/(hr*ft*F) (not used here)
C_p_air = 0.24       # BTU/(lbm*F)   (not used here)

# ──────────── UNIT CONVERSION CONSTANTS ────────────
ft_to_m = 0.3048
in_to_m = 0.0254
mm_to_m = 0.001
lbm_ft3_to_kg_m3 = 0.453592 / (ft_to_m**3)
lbm_ft_s_to_kg_m_s = 0.453592 / ft_to_m

# ──────────── CONVERT PRIMARY INPUTS TO SI ─────────
V_inf *= ft_to_m                         # m/s
v_air_profile = v_air_profile * ft_to_m  # m/s
A_in *= ft_to_m**2                       # m^2
A_out *= ft_to_m**2                      # m^2
A_core_face *= ft_to_m**2                # m^2
tube_h *= ft_to_m                        # m
tube_w *= ft_to_m                        # m
tube_L *= ft_to_m                        # m
fin_d *= ft_to_m                         # m
fin_t *= ft_to_m                         # m
wall_t *= ft_to_m                        # m
rho_air_SI = 0.0749 * lbm_ft3_to_kg_m3   # kg/m^3
mu_air_SI = 1.2e-05 * lbm_ft_s_to_kg_m_s # kg/(m*s)
g_SI = 32.2 * ft_to_m                    # m/s^2

# ───────────────────────── GEOMETRY ─────────────────────────
A_front = A_core_face * math.sin(math.radians(rad_angle_deg))
A_flow = n_tube * tube_w * tube_h
D_h_c = 4 * (tube_h * tube_w) / (2 * (tube_h + tube_w))
A_c_tot = n_tube * n_rows * (2 * (tube_h + tube_w) * tube_L)
A_w_tot = A_c_tot
core_W = tube_L
core_H = A_core_face / core_W
FPI = 1.0 / (fin_d * 39.3701)                 # fins/in (informational)
N_fins = int(FPI * core_W * 39.3701)
core_depth = n_rows * (tube_h + 2*wall_t + fin_t)   # m
W_fin = core_depth
A_fin_tot = 2.0 * N_fins * core_H * W_fin
tube_h_o = tube_h + 2*wall_t
tube_w_o = tube_w + 2*wall_t
A_bare = n_tube * n_rows * (2*(tube_h_o + tube_w_o) * tube_L)
A_a_tot = eta_fin * A_fin_tot + A_bare

# ───────────────── POROSITY & LENGTH SCALE ─────────────────
# Porosity model (same as original; ensure dimensionless)
porosity = 1.0 - (fin_t / fin_d) - (n_tube * wall_t) / core_H
phi = porosity

# Choice of hydraulic length scale for Ergun (modeling choice for finned cores)
# m, tunable; consider passage hydraulic diameter if available
hydraulic_len_scale = 0.8 * fin_d

# ────────────────── ERGUN-BASED COEFFICIENTS ──────────────────
# Per-length coefficients:
# k_perm [m^2] from Ergun viscous term; beta [1/m] (Fluent convention) from inertial term
k_perm = (phi**3 * hydraulic_len_scale**2) / (150.0 * (1.0 - phi)**2)  # m^2
# 1/m (Fluent C2 form)
beta = (1.75 * (1.0 - phi)) / (phi**3 * hydraulic_len_scale)

# ────────────────── OUTPUTS FOR SIMSCALE VS FLUENT ──────────────────
# SimScale uses dp = mu d L u + (rho/2) f L u^2  =>  d [1/m^2], f [1/m]
d_sim = 1.0 / k_perm                                # 1/m^2
# 1/m (note 3.5 = 2*1.75)
f_sim = (3.5 * (1.0 - phi)) / (phi**3 * hydraulic_len_scale)

# Fluent uses dp = mu (1/alpha) L u + rho C2 L u^2  =>  (1/alpha) [1/m^2], C2 [1/m]
D11_fluent = 1.0 / k_perm                           # 1/m^2
C2_fluent = beta                                    # 1/m

# ───────────────────────── PRINT RESULTS ─────────────────────────
print(
    "Porous-media coefficients (Ergun-based):\n"
    f"  - Porosity phi..................... {phi:9.5f}\n"
    f"  - Hydraulic length scale dh........ {hydraulic_len_scale:11.4e}  m\n\n"
    "SimScale Darcy-Forchheimer inputs (dp = mu*d*L*u + (rho/2)*f*L*u^2):\n"
    f"  - d  (Darcy, viscous).............. {d_sim:11.4e}  1/m^2\n"
    f"  - f  (Forchheimer, inertial)....... {f_sim:11.4e}  1/m\n"
    "  - Local flow direction e1.......... (1, 0, 0)  # set to face-normal in your CAD\n"
    "  - Cross-flow (e2,e3): scale d,f up (ex. x10) for anisotropy if desired.\n\n"
    "ANSYS Fluent porous-zone inputs (dp = mu*(1/alpha)*L*u + rho*C2*L*u^2):\n"
    f"  - 1/alpha (viscous resistance)..... {D11_fluent:11.4e}  1/m^2\n"
    f"  - C2      (inertial resistance).... {C2_fluent:11.4e}  1/m\n"
)

# ───────────────── OPTIONAL: FIT CONVERSION HELPERS ─────────────────


def fit_to_simscale(A, B, mu, rho, L):
    """
    Convert a dp = A*u + B*u^2 fit across thickness L into SimScale per-length coefficients.
    A : linear fit coefficient      [Pa*s/m]    (units depend on how the fit was formed)
    B : quadratic fit coefficient   [Pa*s^2/m^2] (same note as above)
    mu: dynamic viscosity           [Pa*s]
    rho: density                    [kg/m^3]
    L : porous thickness used in fit [m]
    Returns: d [1/m^2], f [1/m]
    """
    d = A / (mu * L)
    f = (2.0 * B) / (rho * L)
    return d, f

# Example usage (commented):
# A_fit, B_fit = 0.0, 0.0   # from your dp-u regression
# d_from_fit, f_from_fit = fit_to_simscale(A_fit, B_fit, mu_air_SI, rho_air_SI, core_depth)
# print(f"From fit -> d = {d_from_fit:.4e} 1/m^2, f = {f_from_fit:.4e} 1/m")