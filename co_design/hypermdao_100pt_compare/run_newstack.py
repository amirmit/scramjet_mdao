"""New-stack (this repo, run under MDOenv) side of the 100-point
inlet/cowl/engine/nozzle comparison against HyperMDAO.

For each point in points.csv (see generate_inputs.py): builds SJ2D geometry
from the point's 13 geometry DVs (via geometry_generation.generate_scramjet_sj2d),
then calls co_design.scramjet_point.evaluate_scramjet_point at the point's
flight condition (per the user's explicit request to use that function).

Extracts, per point:
  - inlet/cowl_in/engine/nozzle force (Fx, Fz) -- value + sensitivity to
    each of the 13 geometry DVs (FloatWithSens.number/.sens).
  - engine exit ("outflow") Mach/pressure/temperature -- value + sensitivity.
  - nozzle P_exit -- value + sensitivity.

Sensitivity w.r.t. equivalence_ratio is NOT available here: evaluate_scramjet_point
takes equivalence_ratio as a plain float (case parameter), not an FWS-tracked
design variable, so it never gets its own sensitivity axis -- see
PROJECT_STATE.md's "known loose end" and this comparison's own README.md.

Output: results_newstack.csv, one row per (point_id, component, quantity),
long format, with a `sens_<dv_name>` column per geometry DV -- same schema
HyperMDAO/RunFolder_dev/scramjet2D_comparison/run_100pt_compare.py writes,
so compare_report.py can merge the two directly.

Failed points (geometry build or physics chain raising) are logged to
failures_newstack.csv and skipped, not fatal to the batch.

USAGE
-----
    source /home/s4679920/MDAO/MDOenv/bin/activate
    cd /home/s4679920/MDAO/scramjet_mdao/co_design/hypermdao_100pt_compare
    python run_newstack.py
"""
import os
import sys
import traceback

import numpy as np
import pandas as pd

# co_design/*.py modules (Scramjet2D, geometry_generation, scramjet_point,
# multistage_manager) use flat same-directory imports, so co_design/ itself
# must be on sys.path -- this file lives one level below it.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autodiff import FloatWithSens
from utilities import get_number
from geometry_generation import generate_scramjet_sj2d
from scramjet_point import evaluate_scramjet_point

from hyperProp.Nozzle.nozzle_surrogate_model import load_surrogate_model
from hypervisc import get_friction_model

GEOM_DV_NAMES = [
    "inlet_length_1", "inlet_length_2", "inlet_theta_1", "inlet_theta_2",
    "throat_height", "engine_tilt_ang", "cowl_LE", "engine_length",
    "isolator_length", "engine_h5_h3", "nozzle_length", "nozzle_angle",
    "nozzle_mid_frac",
]

NOZZLE_SURROGATE_VERSION = "v_3_1"
CF_MODEL_NAME = "constant"
CF_PARAMS = dict(cf_value=0.001)


def _row(point_id, component, quantity, val, dv_names, status="ok", error=""):
    row = {
        "point_id": point_id, "component": component, "quantity": quantity,
        "status": status, "error": error,
    }
    if isinstance(val, FloatWithSens):
        row["value"] = val.number
        for i, name in enumerate(dv_names):
            row[f"sens_{name}"] = val.sens[i]
    else:
        row["value"] = get_number(val)
        for name in dv_names:
            row[f"sens_{name}"] = np.nan
    return row


def run_point(row_in, nozzle_surrogate, cf_model):
    point_id = int(row_in["point_id"])

    active_dv_dict = {name: float(row_in[name]) for name in GEOM_DV_NAMES}
    geom_out = generate_scramjet_sj2d(
        active_dv_dict=active_dv_dict, overrides={}, mesh_params=None, save=False,
    )
    vehicle = geom_out["vehicle"]
    fws = geom_out["FWS_param_dict"]
    context = geom_out["context"]
    dv_names = context.variable_names

    S_ref = float(get_number(vehicle.mesh_data.calculate_volume())) ** (2 / 3)

    result = evaluate_scramjet_point(
        alpha=np.deg2rad(float(row_in["alpha_deg"])),
        mach=float(row_in["mach"]),
        altitude=float(row_in["altitude_m"]),
        equivalence_ratio=float(row_in["equivalence_ratio"]),
        vehicle=vehicle,
        fws=fws,
        context=context,
        overrides={},
        nozzle_surrogate=nozzle_surrogate,
        cf_model=cf_model,
        S_ref=S_ref,
        verbosity=0,
    )

    out_rows = []
    comp_map = {"inlet": "inlet", "cowl": "cowl_in", "engine": "engine", "nozzle": "nozzle"}
    for comp_label, comp_key in comp_map.items():
        force = result["components"][comp_key].force
        out_rows.append(_row(point_id, comp_label, "Fx", force.x, dv_names))
        out_rows.append(_row(point_id, comp_label, "Fz", force.z, dv_names))

    outflow = result["models"]["engine"].outflow
    out_rows.append(_row(point_id, "engine", "M", outflow.mach, dv_names))
    out_rows.append(_row(point_id, "engine", "P", outflow.p, dv_names))
    out_rows.append(_row(point_id, "engine", "T", outflow.T, dv_names))

    nozzle_model = result["models"]["nozzle"]
    out_rows.append(_row(point_id, "nozzle", "P_exit", nozzle_model.P_exit, dv_names))

    return out_rows


def main():
    points = pd.read_csv("points.csv")

    nozzle_surrogate = load_surrogate_model(NOZZLE_SURROGATE_VERSION)
    cf_model = get_friction_model(CF_MODEL_NAME, **CF_PARAMS)

    all_rows = []
    failures = []
    for _, row_in in points.iterrows():
        point_id = int(row_in["point_id"])
        try:
            all_rows.extend(run_point(row_in, nozzle_surrogate, cf_model))
            print(f"[newstack] point {point_id:3d} OK")
        except Exception as e:
            failures.append({
                "point_id": point_id, "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[newstack] point {point_id:3d} FAILED: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv("results_newstack.csv", index=False)
    print(f"\nWrote {len(df)} rows ({df['point_id'].nunique()} points ok) to results_newstack.csv")

    if failures:
        fdf = pd.DataFrame(failures)
        fdf.to_csv("failures_newstack.csv", index=False)
        print(f"{len(failures)} points FAILED -- see failures_newstack.csv")


if __name__ == "__main__":
    main()
