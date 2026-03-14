# FSAE SCRIPT NAMES AND DESCRIPTIONS

**1) `radiator_cfd_velocity_heat_transfer_pass_fail.py`**  
Uses CFD-derived air velocity input at the radiator face to perform a detailed heat-transfer pass/fail analysis at each engine operating point from the BHP map. The script calculates required heat rejection, models radiator performance using the supplied airflow input, determines cooling margin for each RPM/MAP bin, associates points with gear-dependent vehicle speed, exports results to CSV, and generates plots for cooling-performance assessment.

**2) `radiator_geometry_based_porous_media_estimator.py`**  
Estimates radiator porous-media resistance coefficients directly from known core geometry using Ergun-based relations, without requiring measured pressure-drop data. The script outputs equivalent porous-media inputs for SimScale and ANSYS Fluent, making it useful for early CFD setup when radiator dimensions are known but experimental flow-loss data is unavailable.

**3) `sidepod_bernoulli_starting_point_optimizer.py`**  
Performs a first-pass sidepod design sweep using a Bernoulli-based bulk velocity estimate to screen inlet area ratio, outlet area ratio, and radiator tilt angle. The script checks basic geometry feasibility, estimates radiator-face airflow from area ratios, evaluates cooling pass/fail across operating points, ranks candidate geometries by weighted failure rate, and identifies a practical starting geometry for later CFD refinement.

**4) `sidepod_cfd_geometry_iteration_report.py`**  
Rapidly calculates sidepod inlet, center, and outlet dimensions along with taper lengths and resulting half-angles for a given geometry setup. The script is intended as a quick iteration tool for generating and comparing candidate sidepod shapes before building or refining CFD models, and it outputs a concise report for each tested configuration.

**5) `radiator_uniform_bulk_velocity_target_face_velocity_sweep.py`**  
Preliminary pre-CFD cooling pass/fail sweep that applies a single uniform bulk air velocity across the radiator core and filters operating points by a matching free-stream vehicle speed window. This is mainly an early screening tool for identifying a reasonable target radiator face velocity before moving to more detailed CFD-informed heat-transfer analysis.
