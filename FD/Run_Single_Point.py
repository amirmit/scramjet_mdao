"""Single-point run at opt_input.py's initial guess (x0) through the real
Run_ObjFunc -- no case-config machinery, no SNOPT. Same xdict/Opt_params
shape Opt_Main.py hands to Run_ObjFunc, built here directly from opt_input.py
so it always matches the currently configured case.

Prints the objective-function evaluation and its sensitivities (Run_SensFunc).

    source /home/s4679920/MDAO/MDOenv/bin/activate
    cd /home/s4679920/MDAO/scramjet_mdao/single_point
    python Run_Single_Point.py
"""

from utilities import get_number
import re

import numpy as np

from Run_ObjFunc import Run_ObjFunc
from Run_SensFunc import Run_SensFunc
import opt_input

###################################
######### Design point ###########
###################################
# x0 of every DV in opt_input.py's DV dict. Edit values here to poke at a
# specific design point.
DV_VALUES = {name: dv["x0"] for name, dv in opt_input.DV.items()}

# Optional: path to a text file holding HyperMDAO's Run_Single_Point output
# (lines like "  dCD/dp: inlet_length_1=-0.0097, ..."); if set, the maximum
# relative difference per quantity is printed against it.
OLD_SENS_FILE = None

###################################
############ Run params ##########
###################################
# opt_input.opt_params carries flight_duration/eta_pr/engine_visc_flg/
# aero_visc_flg/cf_model/cf_params (and the optional mesh_params/
# save_geometry/geometry_name); nozzle_surrogate/aoa_deg live at
# opt_input's top level -- merge both, mirroring Opt_Main.py.
Opt_params = dict(
    opt_input.opt_params,
    nozzle_surrogate=opt_input.nozzle_surrogate,
    aoa_deg=opt_input.aoa_deg,
)

print(f"xdict: {DV_VALUES}")

# ---- Run_ObjFunc -----------------------------------------------------
wind_coeffs, quantities, out4sens = Run_ObjFunc(DV_VALUES, Opt_params)

print("\n--- Run_ObjFunc results ---")
CD, CL, CM = wind_coeffs
print(f"wind_coeffs: CD={get_number(CD):.5f}  CL={get_number(CL):.5f}  CM={get_number(CM):.5f}")
print("quantities:")
for name, value in sorted(quantities.items()):
    print(f"  {name}: {get_number(value):.6g}")

# ---- Component detail (read straight from the real result dict) -------
result = out4sens["result"]
print("\ncomponent body-axis forces (Fx, Fz) [N]:")
for name, comp in result["components"].items():
    if comp is not None:
        print(f"  {name}: {get_number(comp.force.x):.6g}, {get_number(comp.force.z):.6g}")

outflow = result["models"]["engine"].outflow
print(
    f"\nengine outflow: M={get_number(outflow.mach):.5g}  "
    f"P={get_number(outflow.p):.6g}  T={get_number(outflow.T):.6g}"
)
print(f"nozzle P_exit: {get_number(result['models']['nozzle'].P_exit):.6g}")


if __name__ == "__main__":
    pass
# ---- Run_SensFunc ------------------------------------------------------
# sens = Run_SensFunc(out4sens, Opt_params)
#
# print("\n--- Run_SensFunc results ---")
# for name, grad in sens.items():
#     print(f"  d{name}/dp: " + ", ".join(f"{p}={g:.4g}" for p, g in grad.items()))
#
# if OLD_SENS_FILE:
#     old = {}
#     for line in open(OLD_SENS_FILE):
#         m = re.match(r"\s+d(\w+)/dp: (.*)", line)
#         if m:
#             old[m.group(1)] = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in m.group(2).split(", ")}
#     print("\n--- max|new-old| / max|old| per quantity (vs OLD_SENS_FILE) ---")
#     for name, o in old.items():
#         if name in sens:
#             new = np.array([sens[name][p] for p in o])
#             ref = np.array(list(o.values()))
#             print(f"  {name}: {np.abs(new - ref).max() / np.abs(ref).max():.3e}")
