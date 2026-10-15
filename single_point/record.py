"""Builders for the per-evaluation record (out4sens) that OptLogger, PostOpt
and the plots read. Every value is read straight off the objects
evaluate_scramjet_point already produced -- nothing is re-derived here.

Key names match HyperMDAO's out4sens so the logger/PostOpt/plots keep their
field names: geom, inlet_model, cowl_model, engine_model, comp_force, out4mat,
nozzle_diag (plus alpha).

Differences from HyperMDAO's record:
  * comp_force has no "Other" (the new panel method only runs TOP and COWL).
    "Aero" = Top + Cowl + inlet and "Uninstalled_Prop" = cowl_in (internal_cowl)
    + engine + nozzle, i.e. they match the optimisation's own buckets
    (result["aero"] and result["propulsion"]). HyperMDAO's Aero was Top + Cowl
    + Other and its Uninstalled_Prop included the inlet force, so the logged
    eta_th (Uninstalled_Prop.x * v / (m_dot_f * h_pr)) is not comparable with
    old runs' eta_th. Friction is kept separately as "Visc_force".
  * Pt/Tt come from utilities.FlowState's own properties.
"""

from utilities import get_number


def build_comp_force(result):
    """Component forces (Vector3) under HyperMDAO's Comp_force key names."""
    c = result["components"]
    comp_force = {
        "Top": c["top"].force,
        "Cowl": c["cowl_out"].force,
        "inlet": c["inlet"].force,
        "internal_cowl": c["cowl_in"].force,
        "engine": c["engine"].force,
        "nozzle": c["nozzle"].force,
        "Visc_force": c["friction"].force,
    }
    comp_force["Aero"] = comp_force["Top"] + comp_force["Cowl"] + comp_force["inlet"]

    comp_force["Uninstalled_Prop"] = (
        comp_force["nozzle"]
        + comp_force["internal_cowl"]
        + comp_force["engine"]
    )
    return comp_force


def build_out4mat(result, comp_force):
    """Per-station flow properties: freestream, s1, s2 (inlet), C (cowl
    entry), I (cowl exit / engine entry), E (engine exit)."""
    inlet = result["models"]["inlet"]
    cowl = result["models"]["cowl"]
    engine = result["models"]["engine"]

    states = {
        "freestream": inlet.states[0],
        "s1": inlet.states[1],
        "s2": inlet.states[2],
        "C": cowl.flowstate_2,
        "I": cowl.flowstate_3,
        "E": engine.outflow,
    }
    fs = states["freestream"]

    out4mat = {}
    for key, s in states.items():
        out4mat[key] = {
            "M": s.mach,
            "T": s.T,
            "P": s.p,
            "Pt": s.Pt,
            "Tt": s.Tt,
            "h": s.gmodel.Cp * s.T,
            "v": s.v,
            "direction": s.direction,
            "pressure_ratio": s.p / fs.p,
            "temperature_ratio": s.T / fs.T,
            "pressure_recovery": s.Pt / fs.Pt,
        }

    stations = {
        "C": cowl.conserve_2,
        "I": cowl.conserve_3,
        "E": engine.axial_conserve_5,
    }
    for key, sta in stations.items():
        out4mat[key]["m_dot"] = sta.m_dot
        out4mat[key]["mom"] = sta.mom
        out4mat[key]["tE"] = sta.tE

    out4mat["E"]["m_dot_f"] = engine.m_dot_f
    out4mat["E"]["eta_th"] = (
        comp_force["Uninstalled_Prop"].x * fs.v / (engine.m_dot_f * engine.h_pr)
    )
    return out4mat


def build_nozzle_diag(nozzle, nozzle_bounds):
    """Nozzle surrogate inputs, nominal forces, exit pressure, and whether
    the inputs sit inside the surrogate's training box."""
    inputs = nozzle.surrogate_inputs
    oob_vars = [
        k
        for k, v in inputs.items()
        if not (nozzle_bounds[k][0] <= v <= nozzle_bounds[k][1])
    ]
    return {
        **inputs,
        "Fx_p": nozzle.Fx_p_nom,
        "Fy_p": nozzle.Fy_p_nom,
        "P_exit": get_number(nozzle.P_exit),
        "in_box": len(oob_vars) == 0,
        "oob_vars": oob_vars,
    }


def build_record(result, nozzle_bounds, alpha):
    """Everything OptLogger/PostOpt/plots need beyond `quantities`, merged
    into out4sens by calc_vehicle."""
    models = result["models"]
    comp_force = build_comp_force(result)
    return {
        "geom": models["eng_model"]._geom,
        "inlet_model": models["inlet"],
        "cowl_model": models["cowl"],
        "engine_model": models["engine"],
        "comp_force": comp_force,
        "out4mat": build_out4mat(result, comp_force),
        "nozzle_diag": build_nozzle_diag(models["nozzle"], nozzle_bounds),
        "alpha": alpha,
    }


# What goes into the funcs["out4sens"] that pyoptsparse pickles into the
# history file on every call -- the equivalents of what HyperMDAO's out4sens
# held (freestream, inlet/cowl/engine models, geom, comp_force, out4mat,
# nozzle_diag, weight, Sref, eng_vol, range_mod, ...), plus the two things the
# new Run_SensFunc needs (quantities, context). Everything else (notably the
# full evaluate_scramjet_point `result`, ~6.7 MB per call) is left out.
SAVED_OUT4SENS = (
    "quantities",
    "context",
    "freestream",
    "geom",
    "inlet_model",
    "cowl_model",
    "engine_model",
    "comp_force",
    "out4mat",
    "nozzle_diag",
    "alpha",
    "weight",
    "S_ref",
    "total_vol",
    "eng_vol",
    "m_dot_f_lim",
    "range_mod",
)


def slim_out4sens(out4sens):
    """The subset of out4sens that is logged/saved (see SAVED_OUT4SENS)."""
    return {k: out4sens[k] for k in SAVED_OUT4SENS}


def check_physical(quantities, out4mat):
    """Sanity check of the converged analysis; returns a list of problems (empty if fine).

    Every quantity value and every `.sens` entry must be finite, and static pressure and
    temperature must be finite and > 0 at every station. Catches NaN/inf and negative-p/T
    solutions now that the gas model no longer raises. It does NOT detect a solver that
    stopped at a non-root with positive, finite states (e.g. a thermally choked engine);
    that needs the solver residual (checked where the solve happens).
    """
    import numpy as np

    bad = []
    for name, q in quantities.items():
        if not np.isfinite(float(get_number(q))):
            bad.append(f"{name} is not finite")
        sens = getattr(q, "sens", None)
        if sens is not None and not np.all(np.isfinite(np.asarray(sens, dtype=float))):
            bad.append(f"d{name}/dDV is not finite")
    for station, d in out4mat.items():
        for key, label in (("P", "pressure"), ("T", "temperature")):
            v = float(get_number(d[key]))
            if not np.isfinite(v) or v <= 0:
                bad.append(f"{label} at station {station} = {v:.6g}")
    return bad
