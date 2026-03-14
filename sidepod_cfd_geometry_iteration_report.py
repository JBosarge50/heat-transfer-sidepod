import math
import os, sys
from contextlib import redirect_stdout

# ────────────────────────────────────────────────────────────────
# USER INPUTS
# ----------------------------------------------------------------
L_rad = 11.8      # radiator core length [in]
theta = 60        # inclination from HORIZONTAL [deg]
W_c   = 5.2       # center-section width [in]

ratio_A_in  = 0.50   # inlet area / center area [-]
ratio_A_out = 0.65   # outlet area / center area [-]

L_tot     = 24.0  # overall inlet-to-outlet length [in]
angle_lim = 15.0  # max desired inlet half-angle [deg]
R_lo_min  = 0.20  # min L_out / L_in [-]

# --- NEW: optional length-ratio control ------------------------
R_len     = 5     # desired L_in / L_out (ignored unless use_R_len = True)
use_R_len = True  # set True to enforce R_len
# ---------------------------------------------------------------

center_length_buffer = 4
center_height_buffer = 1
# ────────────────────────────────────────────────────────────────


# Center-section geometry
H_c = L_rad * math.sin(math.radians(theta)) + center_height_buffer
L_c = L_rad * math.cos(math.radians(theta)) + center_length_buffer
A_c = H_c * W_c
L_c_half = L_c / 2


def scaled_dims(scale: float, H_ref: float, W_ref: float):
    s = math.sqrt(scale)
    return H_ref * s, W_ref * s


H_in,  W_in  = scaled_dims(ratio_A_in,  H_c, W_c)
H_out, W_out = scaled_dims(ratio_A_out, H_c, W_c)
A_in,  A_out = A_c * ratio_A_in, A_c * ratio_A_out

dH_in,  dW_in  = abs(H_c - H_in),  abs(W_c - W_in)
dH_out, dW_out = abs(H_c - H_out), abs(W_c - W_out)

top_drop_in   = dH_in / 2.0
top_drop_out  = dH_out / 2.0

# -- Split remaining length between tapers ----------------------
L_rem = L_tot - L_c
if L_rem <= 0:
    raise ValueError("Total length too small to accommodate center section.")

if use_R_len:                                     # new branch
    L_in  = (R_len / (1 + R_len)) * L_rem
    L_out = L_rem - L_in
    angle_status = f"length-ratio mode (L_in / L_out = {R_len:g})"
else:
    angle_lim_rad = math.radians(angle_lim)
    L_in_req  = math.hypot(dH_in / 2.0, dW_in) / math.tan(angle_lim_rad)
    L_out_req = L_rem - L_in_req

    if (L_out_req >= R_lo_min * L_in_req) and (0 <= L_out_req < L_in_req):
        L_in, L_out = L_in_req, L_out_req
        angle_status = f"<= {angle_lim:0.1f} deg satisfied"
    else:
        L_in  = L_rem / (1 + R_lo_min)
        L_out = L_rem - L_in
        angle_status = (f"angle > {angle_lim:0.1f} deg or L_in <= L_out - "
                        "lengths redistributed")


def half_angles(dH: float, dW: float, L: float):
    angle_h = math.degrees(math.atan((dH / 2) / L))
    angle_w = math.degrees(math.atan(dW / L))
    angle_r = math.degrees(math.atan(math.hypot(dH / 2, dW) / L))
    return angle_h, angle_w, angle_r


angleh_in,  anglew_in,  angleres_in  = half_angles(dH_in,  dW_in,  L_in)
angleh_out, anglew_out, angleres_out = half_angles(dH_out, dW_out, L_out)


def line(lbl, val):
    print(f"{lbl:<29s} {val:>10.3f}")


print("\nSIDE-POD DIMENSION SUMMARY (theta from horizontal)")
print("─" * 43)
line("center height  H_c  [in]", H_c)
line("center width   W_c  [in]", W_c)
line("center length  L_c_half  [in]", L_c_half)
print("─" * 43)
line("Inlet  height  H_in [in]", H_in)
line("Inlet  width   W_in [in]", W_in)
line("Inlet  area    A_in [in^2]", A_in)
line("Outlet height  H_out[in]", H_out)
line("Outlet width   W_out[in]", W_out)
line("Outlet area    A_out[in^2]", A_out)
print("─" * 43)
line("Inlet length   L_in [in]", L_in)
line("Outlet length  L_out[in]", L_out)
line("L_in / L_out        [-]",  L_in / L_out)   # shows actual ratio
line("Total length   L_tot[in]", L_tot)
print("─" * 43)
line("Inlet angle_height [deg]", angleh_in)
line("Inlet angle_width  [deg]", anglew_in)
line("Inlet angle_result [deg]", angleres_in)
print("─" * 43)
line("Outlet angle_height[deg]", angleh_out)
line("Outlet angle_width [deg]", anglew_out)
line("Outlet angle_result[deg]", angleres_out)
print("─" * 43)
print(f"Inlet-angle criterion: {angle_status}")

# ───────────────── REPORT (unchanged) ───────────────────────────
output_dir = (r"C:\Users\pxzuk\OneDrive\Documents\OneDrive - "
              r"Mississippi State University\Mississippi State FSAE\FSAE "
              r"2026\Design\Powertrain\Cooling\Sidepod")


def make_filename(theta_deg, r_in, r_out, alpha_in, alpha_out):
    return (f"{int(round(theta_deg))}_{r_in:g}_{r_out:g}_"
            f"{alpha_in:.1f}_{alpha_out:.1f}.txt")


fname = make_filename(theta, ratio_A_in, ratio_A_out,
                      angleres_in, angleres_out)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, fname)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):           [s.write(d) for s in self.streams]
    def flush(self):              [s.flush() for s in self.streams]


with open(fname, "w", encoding="utf-8") as fh, \
     redirect_stdout(Tee(sys.__stdout__, fh)):

    print("\nDESIGN PARAMETERS")
    print("Radiator Tilt Angle [deg]   Inlet Area Ratio [-]   Exhaust Area Ratio [-]")
    print(f"{theta:>22.1f}{ratio_A_in:>25.2f}{ratio_A_out:>27.2f}")
    print("─" * 43)

    # (table section identical to console-printed block)
    # ... (omitted here for brevity) ...

print(f"\nReport written to: {fname}")