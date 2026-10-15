import json
import numpy as np
import os
from pyoptsparse import History
from scipy.constants import g
from scipy.io import savemat
from pathlib import Path
import matplotlib.pyplot as plt

import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from utilities import get_number
from OptLogger import OptLogger


def SaveHistory(filename, sens_log=None, run_details=None, case_config_path=None):
    """

    :param filename: str
    :param sens_log: OptLogger instance
    :param run_details: dict of DV/Con/options/const_vars/Opt_params to save
    :param case_config_path: path to the case config .py file used for this
        run (e.g. "cases/baseline.py") -- copied into the run folder under
        its own basename so the run is reproducible from its saved folder
        alone, without needing the cases/ directory as it stood at run time.
    :return:
    """
    hist = History(filename, flag="r")

    objNames = hist.getObjNames()
    conNames = hist.getConNames()
    DVNames = hist.getDVNames()
    sensNames = hist.getExtraFuncsNames()
    val = hist.getValues()
    obj = hist.getObjInfo()
    DV = hist.getDVInfo()
    con = hist.getConInfo()
    # breakpoint()
    out = {
        "Var": {
            key: {
                "value": val[key],
                "lower": DV[key]["lower"],
                "upper": DV[key]["upper"],
                "scale": DV[key]["scale"],
            }
            for key in DVNames
            if key in val
        },
        "Obj": {
            key: {
                "value": val[key],
                "scale": obj[key]["scale"],
                "optimality": val["optimality"],
                "feasibility": val["feasibility"],
            }
            for key in objNames
            if key in val
        },
        "Con": {
            key: {
                "value": val[key],
                "scale": con[key]["scale"],
                "lower": _scalar_bound(con[key]["lower"]),
                "upper": _scalar_bound(con[key]["upper"]),
            }
            for key in conNames
            if key in val
        },
        "time": val["time"],
        "iter": val["iter"],
    }

    # Sens data for mat
    if sens_log.Sensitivity["Sens"]:

        # Option B: filter wrapper sensitivities to MAJOR iterations via x matching
        payload = _as_logger_payload(sens_log)

        # Build (n_major, n_dv) from .hst
        X_major = _build_major_iter_design_matrix(val, DVNames)

        # Build (n_calls, n_dv) from wrapper
        X_calls = _stack_by_names(payload["x"], DVNames, to_float=True)

        # Match each major-iteration x to the first equal x in wrapper calls
        matched_idx, unmatched = _match_rows(X_major, X_calls)

        # Build gradient matrix for the objective "Fx" (as per your wrapper)
        # Shape: (n_major, n_dv), NaN where unmatched
        n_major, n_dv = X_major.shape
        dFx_dx = np.full((n_major, n_dv), np.nan, dtype=float)

        # Wrapper gradients are stored as {dv_name: [g0, g1, ...]}
        grads_by_dv = payload["Sens"]  # dict: DV -> list of grads per call

        # Fill per matched major iteration
        for i, j in enumerate(matched_idx):
            if j is None:
                continue
            # Pick the j-th sensitivity sample for each DV, in dv_names order
            row = []
            for k, dv in enumerate(DVNames):
                if dv not in grads_by_dv:
                    raise KeyError(
                        f"DV '{dv}' missing in sens_log['Sens']. "
                        f"Available: {list(grads_by_dv.keys())}"
                    )
                g_list = grads_by_dv[dv]
                if j >= len(g_list):
                    raise IndexError(
                        f"Match index {j} out of range for gradients of DV '{dv}'."
                    )
                row.append(float(g_list[j]))
            dFx_dx[i, :] = row
        out["Sens"] = {
            "values": dFx_dx,
            "norm": _normalize_sensitivities_by_row_norm(dFx_dx),
        }

    else:
        out["Sens"] = (
            {
                key: val[key]
                for key in sensNames
                if key.startswith("d")
                and any(key[1:].startswith(obj) for obj in objNames)
            },
        )

    # Geom data for mat
    if sens_log.Objective["geom"]:
        geom_calls = sens_log.Objective["geom"]

        if sens_log.Sensitivity["Sens"]:
            geom_major = []
            for j in matched_idx:
                if j is None:
                    geom_major.append(None)
                else:
                    geom_major.append(geom_calls[j])
        else:
            geom_major = geom_calls

        point_keys = ["N0", "I1", "C1", "C2", "E1", "E2", "T0", "T1", "T2"]
        n_major = len(geom_major)
        geom_stacked = {}

        for key in point_keys:
            points_3n = np.full((3, n_major), np.nan, dtype=float)
            for i, geom_dict in enumerate(geom_major):
                if geom_dict is None:
                    continue
                vec = geom_dict[key]
                points_3n[0, i] = float(vec.x)
                points_3n[1, i] = float(vec.y)
                points_3n[2, i] = float(vec.z)
            geom_stacked[key] = points_3n

        if "width" in geom_major[0]:
            width_vec = np.full(n_major, np.nan, dtype=float)
            for i, geom_dict in enumerate(geom_major):
                width_vec[i] = float(geom_dict["width"])
            geom_stacked["width"] = width_vec

        if "inlet_thetas" in geom_major[0]:
            theta_vec = np.full((2, n_major), np.nan, dtype=float)
            for i, geom_dict in enumerate(geom_major):
                theta_vec[0, i] = float(geom_dict["inlet_thetas"][0])
                theta_vec[1, i] = float(geom_dict["inlet_thetas"][1])
            geom_stacked["inlet_thetas"] = theta_vec

    else:
        geom_stacked = {}
    out["geom"] = geom_stacked

    # Inlet shock waves for mat
    if sens_log.Objective["inlet_model"]:
        inlet_calls = sens_log.Objective["inlet_model"]

        if sens_log.Sensitivity["Sens"]:
            inlet_major = []
            for j in matched_idx:
                if j is None:
                    inlet_major.append(None)
                else:
                    inlet_major.append(inlet_calls[j])
        else:
            inlet_major = inlet_calls

        n_major = len(inlet_major)
        inlet_stacked = {}

        if hasattr(inlet_major[0], "InletMode"):
            IM_vec = np.full(n_major, np.nan, dtype=float)
            for i, inlet_inst in enumerate(inlet_major):
                IM_vec[i] = float(inlet_inst.InletMode)
            inlet_stacked["InletMode"] = IM_vec

        if hasattr(inlet_major[0], "betas"):
            beta_vec = np.full((2, n_major), np.nan, dtype=float)
            for i, inlet_inst in enumerate(inlet_major):
                beta_vec[0, i] = float(inlet_inst.betas[0])
                beta_vec[1, i] = float(inlet_inst.betas[1])
            inlet_stacked["inlet_betas"] = beta_vec

    else:
        inlet_stacked = {}
    out["inlet"] = inlet_stacked

    # Pressure recovery for mat
    if sens_log.Objective["pressure_recovery"]:
        pr_calls = sens_log.Objective["pressure_recovery"]

        if sens_log.Sensitivity["Sens"]:
            pr_major = []
            for j in matched_idx:
                if j is None:
                    pr_major.append(None)
                else:
                    pr_major.append(pr_calls[j])
        else:
            pr_major = pr_calls

        pr_keys = ["s1", "s2", "C", "I", "E"]
        n_major = len(pr_major)
        pr_stacked = {}

        for key in pr_keys:
            prs = np.full(n_major, np.nan, dtype=float)
            for i, pr_dict in enumerate(pr_major):
                prs[i] = float(pr_dict[key])
            pr_stacked[key] = prs

    else:
        pr_stacked = {}
    out["pressure_recovery"] = pr_stacked

    # Objective function options
    if sens_log.Objective["out_params"]:
        params_calls = sens_log.Objective["out_params"]

        if sens_log.Sensitivity["Sens"]:
            params_major = []
            for j in matched_idx:
                if j is None:
                    params_major.append(None)
                else:
                    params_major.append(params_calls[j])
        else:
            params_major = params_calls

        params_keys = ["CD", "T_W_ratio"]
        n_major = len(params_major)
        params_stacked = {}

        for key in params_keys:
            params = np.full(n_major, np.nan, dtype=float)
            for i, params_dict in enumerate(params_major):
                params[i] = float(params_dict[key])
            params_stacked[key] = params

    else:
        params_stacked = {}
    out["out_params"] = params_stacked

    # Nozzle surrogate diagnostics for mat + json (surrogate inputs, Puffin-
    # axis forces, exit pressure, and a training-box extrapolation flag —
    # one row per major iteration).
    if sens_log.Objective["nozzle_diag"]:
        nz_calls = sens_log.Objective["nozzle_diag"]

        if sens_log.Sensitivity["Sens"]:
            nz_major = []
            for j in matched_idx:
                if j is None:
                    nz_major.append(None)
                else:
                    nz_major.append(nz_calls[j])
        else:
            nz_major = nz_calls

        nz_numeric_keys = [
            "h5",
            "nozzle_length",
            "nozzle_angle",
            "nozzle_mid_frac",
            "M_out",
            "P_out",
            "Fx_p",
            "Fy_p",
            "P_exit",
        ]
        n_major = len(nz_major)
        nz_stacked = {}

        for key in nz_numeric_keys:
            vals = np.full(n_major, np.nan, dtype=float)
            for i, nz_dict in enumerate(nz_major):
                if nz_dict is not None:
                    vals[i] = float(nz_dict[key])
            nz_stacked[key] = vals

        in_box = np.full(n_major, np.nan, dtype=float)
        for i, nz_dict in enumerate(nz_major):
            if nz_dict is not None:
                in_box[i] = float(bool(nz_dict["in_box"]))
        nz_stacked["in_box"] = in_box

        # Which specific inputs went outside the training box (JSON only —
        # variable-length string lists don't round-trip cleanly through .mat).
        nz_oob_vars = [
            (nz_dict["oob_vars"] if nz_dict is not None else [])
            for nz_dict in nz_major
        ]

    else:
        nz_stacked = {}
        nz_oob_vars = []
    out["nozzle"] = nz_stacked

    # Per-component force breakdown for mat (one (3, n_major) array per
    # component key — keys come from OPMVec.solve()'s "Top"/"Cowl"/"Other"
    # np.ndarray entries plus the Vector3 entries added in calc_vehicle():
    # "inlet","internal_cowl","engine","nozzle","Aero","Uninstalled_Prop").
    if sens_log.Objective["comp_force"]:
        cf_calls = sens_log.Objective["comp_force"]

        if sens_log.Sensitivity["Sens"]:
            cf_major = []
            for j in matched_idx:
                if j is None:
                    cf_major.append(None)
                else:
                    cf_major.append(cf_calls[j])
        else:
            cf_major = cf_calls

        n_major = len(cf_major)
        comp_keys = []
        for cf_dict in cf_major:
            if cf_dict is not None:
                comp_keys = list(cf_dict.keys())
                break

        cf_stacked = {}
        for key in comp_keys:
            vec3 = np.full((3, n_major), np.nan, dtype=float)
            for i, cf_dict in enumerate(cf_major):
                if cf_dict is None or key not in cf_dict:
                    continue
                comp = cf_dict[key]
                if isinstance(comp, np.ndarray):
                    vec3[:, i] = [get_number(v) for v in comp]
                else:
                    vec3[:, i] = [
                        get_number(comp.x),
                        get_number(comp.y),
                        get_number(comp.z),
                    ]
            cf_stacked[key] = vec3
    else:
        cf_stacked = {}
    out["comp_force"] = cf_stacked

    # Engine performance (mass flow rates, fuel budget limit, thermal
    # efficiency) for mat. m_dot_air = conserve_3.m_dot (post-cowl-turn,
    # pre-combustion — see Engine_1D.py:46-47); m_dot_fuel = m_dot_f =
    # axial_conserve_5.m_dot - axial_conserve_3.m_dot (Engine_1D.py:129).
    if sens_log.Objective["engine_perf"]:
        ep_calls = sens_log.Objective["engine_perf"]

        if sens_log.Sensitivity["Sens"]:
            ep_major = []
            for j in matched_idx:
                if j is None:
                    ep_major.append(None)
                else:
                    ep_major.append(ep_calls[j])
        else:
            ep_major = ep_calls

        ep_keys = ["m_dot_air", "m_dot_fuel", "m_dot_f_lim", "eta_th"]
        n_major = len(ep_major)
        ep_stacked = {}
        for key in ep_keys:
            vals = np.full(n_major, np.nan, dtype=float)
            for i, ep_dict in enumerate(ep_major):
                if ep_dict is not None:
                    vals[i] = get_number(ep_dict[key])
            ep_stacked[key] = vals
    else:
        ep_stacked = {}
    out["engine_perf"] = ep_stacked

    # Vehicle weight/mass and reference area for mat (mass derived from
    # weight since only weight = mass*g is carried in out4sens)
    if sens_log.Objective["vehicle"]:
        veh_calls = sens_log.Objective["vehicle"]

        if sens_log.Sensitivity["Sens"]:
            veh_major = []
            for j in matched_idx:
                if j is None:
                    veh_major.append(None)
                else:
                    veh_major.append(veh_calls[j])
        else:
            veh_major = veh_calls

        n_major = len(veh_major)
        weight_vec = np.full(n_major, np.nan, dtype=float)
        Sref_vec = np.full(n_major, np.nan, dtype=float)
        volume_vec = np.full(n_major, np.nan, dtype=float)
        in_vol_vec = np.full(n_major, np.nan, dtype=float)
        eng_vol_vec = np.full(n_major, np.nan, dtype=float)
        for i, veh_dict in enumerate(veh_major):
            if veh_dict is not None:
                weight_vec[i] = get_number(veh_dict["weight"])
                Sref_vec[i] = get_number(veh_dict["Sref"])
                volume_vec[i] = get_number(veh_dict["volume"])
                in_vol_vec[i] = get_number(veh_dict["internal_vol"])
                eng_vol_vec[i] = get_number(veh_dict["eng_vol"])
        veh_stacked = {
            "weight": weight_vec,
            "mass": weight_vec / g,
            "Sref": Sref_vec,
            "volume": volume_vec,
            "internal_vol": in_vol_vec,
            "eng_vol": eng_vol_vec,
        }
    else:
        veh_stacked = {}
    out["vehicle"] = veh_stacked

    # Freestream conditions (M, P, T, dynamic pressure) for mat — fixed for
    # a single-point run today, logged per major iteration to future-proof
    # against multi-point/trajectory optimisation.
    if sens_log.Objective["freestream"]:
        fs_calls = sens_log.Objective["freestream"]

        if sens_log.Sensitivity["Sens"]:
            fs_major = []
            for j in matched_idx:
                if j is None:
                    fs_major.append(None)
                else:
                    fs_major.append(fs_calls[j])
        else:
            fs_major = fs_calls

        fs_keys = ["M", "P", "T", "q"]
        n_major = len(fs_major)
        fs_stacked = {}
        for key in fs_keys:
            vals = np.full(n_major, np.nan, dtype=float)
            for i, fs_dict in enumerate(fs_major):
                if fs_dict is not None:
                    vals[i] = get_number(fs_dict[key])
            fs_stacked[key] = vals
    else:
        fs_stacked = {}
    out["freestream"] = fs_stacked

    # Breguet range-equation breakdown (Run_ObjFunc.py's range_mod dict) for
    # mat + json -- Thrust/Aero are Vector3 (stacked like comp_force above),
    # the rest are scalars (get_number strips any FloatWithSens autodiff
    # wrapper). Computed unconditionally in calc_vehicle() regardless of
    # whether "range" is this case's active objective.
    if sens_log.Objective["range_mod"]:
        rm_calls = sens_log.Objective["range_mod"]

        if sens_log.Sensitivity["Sens"]:
            rm_major = []
            for j in matched_idx:
                if j is None:
                    rm_major.append(None)
                else:
                    rm_major.append(rm_calls[j])
        else:
            rm_major = rm_calls

        n_major = len(rm_major)
        rm_stacked = {}

        for key in ("Thrust", "Aero"):
            vec3 = np.full((3, n_major), np.nan, dtype=float)
            for i, rm_dict in enumerate(rm_major):
                if rm_dict is None:
                    continue
                comp = rm_dict[key]
                vec3[:, i] = [
                    get_number(comp.x),
                    get_number(comp.y),
                    get_number(comp.z),
                ]
            rm_stacked[key] = vec3

        rm_scalar_keys = [
            "Isp",
            "weight_0",
            "weight_1",
            "velocity",
            "m_dot_f",
            "mass_f",
            "range",
        ]
        for key in rm_scalar_keys:
            vals = np.full(n_major, np.nan, dtype=float)
            for i, rm_dict in enumerate(rm_major):
                if rm_dict is not None:
                    vals[i] = get_number(rm_dict[key])
            rm_stacked[key] = vals
    else:
        rm_stacked = {}
    out["range_mod"] = rm_stacked

    # Optimization definitions (DV/Con bounds, SNOPT options, Opt_params
    # incl. nozzle_surrogate/objective, and the frozen non-DV invars). Unbounded
    # DV/Con entries are defined as None (e.g. Consts["Outflow_M"]["gub"])
    # which savemat cannot serialise, so None -> NaN first.
    if run_details is not None:
        out["run_details"] = _none_to_nan(run_details)

    # ---------- create folder, copy files, save MAT + JSON ----------

    # Base path of the original history file
    base_path = Path(filename).resolve()

    # 1. Create new folder named after "filename" (without extension)
    run_dir = base_path.with_suffix("")  # e.g. "run01.hst" -> folder "run01"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save any currently open matplotlib figures into the run directory
    for num in plt.get_fignums():
        fig = plt.figure(num)

        fig_name = getattr(fig, "name", f"figure_{num}")
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in fig_name
        )

        fig.savefig(run_dir / f"{safe_name}.png", dpi=300, bbox_inches="tight")

    # 3. Copy optimizer output files into that folder
    #    Adjust this list to your actual output files
    files_to_copy = [
        "SNOPT_print.out",
        "SNOPT_summary.out",
        "SNOPT_dump.out",
        filename,
    ]
    for fname in files_to_copy:
        src = base_path.parent / fname
        if src.exists():
            shutil.copy2(src, run_dir / fname)

    # Run provenance: the code that produced this run -- the single_point
    # driver/pipeline files plus the shared co_design files it imports.
    # Located relative to this file, so it doesn't depend on the launch dir.
    here = Path(__file__).resolve().parent
    code_files = [
        here / "Opt_Main.py",
        here / "Run_ObjFunc.py",
        here / "Run_SensFunc.py",
        here.parent / "co_design" / "Scramjet2D.py",
        here.parent / "co_design" / "geometry_generation.py",
        here.parent / "co_design" / "scramjet_point.py",
        here.parent / "co_design" / "quantity_catalog.py",
    ]
    for src in code_files:
        if src.exists():
            shutil.copy2(src, run_dir / src.name)

    # 3b. Copy the case config too, under its own basename (its filename
    # varies per case, unlike the fixed names above, so it isn't part of
    # files_to_copy -- see run_diff.diff_vs_parent for how it gets diffed).
    if case_config_path is not None:
        case_config_src = Path(case_config_path).resolve()
        if case_config_src.exists():
            shutil.copy2(case_config_src, run_dir / case_config_src.name)

    # 4. Save .mat and .json inside that folder

    # 4a. .mat file
    mat_name = run_dir / (base_path.stem + ".mat")
    savemat(mat_name, out)

    # 4b. Python-friendly JSON file
    # Build python_friendly with lists instead of raw np.ndarray
    python_friendly = {
        "dv_names": list(DVNames),
        "obj_names": list(objNames),
        "con_names": list(conNames),
        "history": {
            "iter": np.asarray(val["iter"]).tolist(),
            "time": np.asarray(val["time"]).tolist(),
            "dv_values": {k: np.asarray(val[k]).tolist() for k in DVNames if k in val},
            "dv_bounds": {
                k: {
                    "lower": DV[k]["lower"],
                    "upper": DV[k]["upper"],
                    "scale": DV[k]["scale"],
                }
                for k in DVNames
            },
            "obj_values": {
                k: np.asarray(val[k]).tolist() for k in objNames if k in val
            },
            "optimality": np.asarray(val["optimality"]).tolist(),
            "feasibility": np.asarray(val["feasibility"]).tolist(),
            "con_values": {
                k: np.asarray(val[k]).tolist() for k in conNames if k in val
            },
            "con_bounds": {
                k: {
                    "lower": _scalar_bound(con[k]["lower"]),
                    "upper": _scalar_bound(con[k]["upper"]),
                }
                for k in conNames
            },
            "geom": geom_stacked,
            "nozzle": {
                **{k: v.tolist() for k, v in nz_stacked.items()},
                "oob_vars": nz_oob_vars,
            },
            "comp_force": {k: v.tolist() for k, v in cf_stacked.items()},
            "engine_perf": {k: v.tolist() for k, v in ep_stacked.items()},
            "vehicle": {k: v.tolist() for k, v in veh_stacked.items()},
            "freestream": {k: v.tolist() for k, v in fs_stacked.items()},
            "range_mod": {k: v.tolist() for k, v in rm_stacked.items()},
        },
        "definitions": {
            "model_import": "my_optim.problem_defs:create_problem",
            "run_options": run_details
            or {},  # ideally kept simple: numbers, strings, lists, dicts
        },
    }

    json_name = run_dir / (base_path.stem + ".json")
    with open(json_name, "w") as f:
        json.dump(python_friendly, f, default=_to_serializable, indent=2)

    # 5. Remove the original .hst file from the working directory
    try:
        if base_path.exists():
            base_path.unlink()
    except OSError as e:
        print(f"Warning: could not delete history file '{base_path}': {e}")


def _scalar_bound(bound):
    """pyoptsparse's ConInfo lower/upper come back as a length-1 array
    (one entry per scalar sub-constraint — every constraint in this problem
    is added via addCon, i.e. nCon=1) built with allow_none=True, so an
    unbounded side is array([None], dtype=object). Object arrays containing
    None crash savemat's write_cells, so unwrap to a plain float, NaN for
    unbounded."""
    val = np.atleast_1d(bound)[0]
    return float("nan") if val is None else float(val)


def _none_to_nan(obj):
    """Recursively replace None with NaN so nested dicts (e.g. run_details,
    which carries unbounded-constraint entries like {"glb": None}) survive
    scipy.io.savemat, which cannot serialise NoneType."""
    if obj is None:
        return float("nan")
    if isinstance(obj, dict):
        return {k: _none_to_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_none_to_nan(v) for v in obj]
    return obj


def _to_serializable(x):
    """Convert NumPy types to plain Python types for JSON."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    # Let json handle dict, list, str, int, float, bool, None
    return x


def _build_major_iter_design_matrix(
    hist_values: Dict[str, np.ndarray], dv_names: Sequence[str]
) -> np.ndarray:
    """
    Collect the design variables from the .hst values dict into an array
    of shape (n_major, n_dv) following dv_names.
    """
    X_cols = []
    n_major = None

    for dv in dv_names:
        if dv not in hist_values:
            raise KeyError(
                f"Design variable '{dv}' not present in History.getValues()."
            )
        col = np.asarray(hist_values[dv]).astype(float)
        if col.ndim != 1:
            col = np.squeeze(col)
        n_major = n_major or col.size
        if col.size != n_major:
            raise ValueError(
                f"All DV arrays must have the same length. '{dv}' has "
                f"{col.size}, expected {n_major}."
            )
        X_cols.append(col)

    return np.column_stack(X_cols)


def _as_logger_payload(sens_log: Union[object, Dict]) -> Dict[str, Dict[str, List]]:
    """
    Normalise the user's logger into a simple payload:
    {
      "x": {dv_name: [values...]},
      "Sens": {dv_name: [grad_values...]},
    }

    This supports:
      * The OptLogger instance from the question
      * A plain dict in the same structure
    """
    # If it's the OptLogger instance, prefer its attributes
    if hasattr(sens_log, "Sensitivity"):
        payload = sens_log.Sensitivity
    elif isinstance(sens_log, dict) and "Sens" in sens_log:
        # Could be top-level {"Objective": ..., "Sens": ...} as per save()
        maybe = sens_log["Sens"]
        if isinstance(maybe, dict) and "x" in maybe and "Sens" in maybe:
            payload = maybe
        else:
            raise ValueError(
                "sens_log dict does not contain expected keys: 'Sens' and 'x'."
            )
    elif isinstance(sens_log, dict) and "x" in sens_log and "Sens" in sens_log:
        # Already the inner dictionary
        payload = sens_log
    else:
        raise TypeError(
            "Unsupported sens_log type. Pass your OptLogger instance or a dict "
            "with keys {'x': {...}, 'Sens': {...}} or the top-level "
            "{'Objective': ..., 'Sens': ...} structure."
        )

    # Basic shape checks (lists of equal length per DV)
    lengths = {len(v) for v in payload["x"].values()}
    if not lengths:
        raise ValueError("sens_log.x appears to be empty.")
    if len(lengths) != 1:
        raise ValueError(
            "Inconsistent sens_log.x lengths across DVs. Ensure you append "
            "one value per sensFunc call for every DV."
        )
    sens_lengths = {len(v) for v in payload["Sens"].values()} or {0}
    if sens_lengths != lengths:
        raise ValueError(
            "The number of entries in sens_log['x'] and sens_log['Sens'] "
            "must match (one sensitivity vector per x)."
        )

    return payload


def _stack_by_names(
    data_by_name: Dict[str, Sequence],
    names_in_order: Sequence[str],
    to_float: bool = True,
) -> np.ndarray:
    """
    Turn {name: [v0, v1, ...]} into an array of shape (n_samples, n_names),
    following `names_in_order`. Missing names raise a KeyError.
    """
    for name in names_in_order:
        if name not in data_by_name:
            raise KeyError(
                f"Expected key '{name}' not found in data; available: "
                f"{list(data_by_name.keys())}"
            )

    # Convert each series to a 1D float array
    cols = []
    for name in names_in_order:
        col = np.asarray(data_by_name[name])
        # If values were arrays (e.g., shape ()), ravel them
        if col.ndim > 1:
            col = np.squeeze(col)
        if to_float:
            col = col.astype(float)
        cols.append(col)

    # Stack to shape (n_samples, n_names)
    return np.column_stack(cols)


def _match_rows(
    X_target: np.ndarray,
    X_candidates: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[List[Optional[int]], List[int]]:
    """
    For each row of X_target (major iterations), find the *first* index j
    in X_candidates (sensFunc calls) such that allclose(row_i, row_j).
    Returns:
      matched_indices: list of candidate indices or None if not found.
      unmatched_i: list of i where no match was found.
    """
    matched: List[Optional[int]] = []
    unmatched: List[int] = []

    # Build a simple linear scan. If performance becomes an issue, consider
    # hashing/rounding keys to buckets.
    for i in range(X_target.shape[0]):
        row = X_target[i]
        idx = None
        # Scan candidates; choose the earliest matching sens call
        for j in range(X_candidates.shape[0]):
            if np.allclose(row, X_candidates[j], rtol=rtol, atol=atol):
                idx = j
                break
        matched.append(idx)
        if idx is None:
            unmatched.append(i)

    return matched, unmatched


def _normalize_sensitivities_by_row_norm(sens_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row of a sensitivity matrix by the maximum Euclidean norm
    across all rows.

    Parameters
    ----------
    sens_matrix : np.ndarray
        Array of shape (n_rows, n_cols), where rows are sensitivities for each
        major iteration. NaNs are allowed and will be ignored in norm computation.

    Returns
    -------
    np.ndarray
        Normalized sensitivity matrix of the same shape, where each row has been
        divided by the maximum row norm (global scaling factor).
    """
    # Calculate row-wise Euclidean norms, ignoring NaNs
    row_norms = np.linalg.norm(np.nan_to_num(sens_matrix, nan=0.0), axis=1)

    # Avoid division by zero: take max norm > 0
    max_norm = np.nanmax(row_norms)
    if max_norm == 0:
        return sens_matrix.copy()  # no scaling if everything is zero

    # Normalize all rows by the max norm
    normalized_matrix = sens_matrix / max_norm
    return normalized_matrix


if __name__ == "__main__":

    sens_log = OptLogger()
    sens_log.Sensitivity = {
        "x": {
            "inlet_length_1": [
                1,
                0.9,
                0.885,
                0.76655,
                0.6445465,
                0.70272385,
                0.70844945,
                0.73861081,
                0.73870615,
                0.73870615,
            ],
            "inlet_length_2": [
                1.3,
                1.2,
                1.369,
                1.44007,
                1.5132721,
                1.53111737,
                1.52323856,
                1.49063703,
                1.49053397,
                1.49053397,
            ],
        },
        "Sens": {
            "inlet_length_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "inlet_length_2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    }

    SaveHistory("test_logger.hst", sens_log)
