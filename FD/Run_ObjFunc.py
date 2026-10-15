import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.constants import g

# geometry_generation.py / scramjet_point.py / quantity_catalog.py live in
# the sibling co_design/ folder -- this repo's real, actively-developed SJ2D
# pipeline. (CLAUDE.md's repo table and the old sys.path target here both
# still describe a vehicle/Eng_Build.py location that was never created --
# see PROJECT_STATE.md and this session's plan; worth fixing CLAUDE.md once
# this port settles.)
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "co_design"))

from hyperProp.Nozzle.nozzle_surrogate_model import load_surrogate_model
from hypervisc import get_friction_model

from geometry_generation import generate_scramjet_sj2d
from scramjet_point import evaluate_scramjet_point
from Scramjet2D import sum_aero_results
from quantity_catalog import CATALOG
from problem_catalog import check_quantities
from record import build_record, check_physical

##############################################################################
# Nozzle surrogate registry. Each version ("v_3_1" = M6.5, "v_3_2" = M10, ...)
# is loaded and cached at most once per process, the first time a case
# config selects it via Opt_params["nozzle_surrogate"] -- not at import
# time, so switching versions is a config change, not a code change. Mirrors
# the old single_point registry pattern, against the new-stack loader
# (hyperProp.Nozzle.nozzle_surrogate_model.load_surrogate_model) instead of
# HyperPro's pickle-based one -- the per-version config module still carries
# FREESTREAM_MACH/P/T and BOUNDS under the same names as before.
##############################################################################

_SURROGATE_CACHE = {}


class EvaluationFailure(RuntimeError):
    """A design point that can't be evaluated (geometry, inlet, or the
    cowl/engine/nozzle chain failed). Opt_Main.objfunc catches this -- and
    only this -- and returns fail=True so SNOPT backs off to a shorter step
    instead of the whole run dying. Anything else (a coding error) is not
    caught and still crashes with its traceback."""


def get_surrogate(version):
    """Returns (NozzleSurrogateModel, config_module) for `version` (e.g.
    "v_3_1", "v_3_2"). Cached: repeated calls for the same version within a
    process reuse the already-deserialized model instead of reloading the
    pickle.
    """
    if version not in _SURROGATE_CACHE:
        cfg = importlib.import_module(f"hyperProp.Nozzle.models.config_{version}")
        model = load_surrogate_model(version)
        _SURROGATE_CACHE[version] = (model, cfg)
    return _SURROGATE_CACHE[version]


def _set_package_logs(on):
    """hypervehicle/hypergrad print INFO status lines (mesh generation, integrators) on every
    evaluation. Silenced (WARNING and above only) unless opt_params["package_logs"] is True.
    Applied on every call, not at import: the packages reset their own level to INFO when imported."""
    level = logging.INFO if on else logging.WARNING
    for name in ("hypervehicle", "hypergrad"):
        logging.getLogger(name).setLevel(level)


def Run_ObjFunc(xdict, Opt_params, save_flg=False):
    """
    :param xdict: Design variables dictionary -- exactly the active DVs.
        pyoptsparse's own xdict already matches generate_scramjet_sj2d's
        active_dv_dict shape 1:1; unlike old-stack's Eng_Build (whose
        **kwargs required every field populated), no nominal/fallback dict
        is needed here.
    :param Opt_params: shared run constants (nozzle_surrogate, aoa_deg,
        altitude, fixed_dvs, default_dvs, cf_model, cf_params,
        engine_visc_flg, aero_visc_flg, eta_pr, flight_duration, ...)
    :param save_flg: Returns run constants for saving record
    :return: (wind_coeffs, quantities, out4sens). wind_coeffs is the total
        (aero + propulsion [+ friction if aero_visc_flg]) AeroResults
        wind_coeffs(aoa) tuple (CD, CL, CM).
    """

    verbosity = 0
    _set_package_logs(Opt_params.get("package_logs", False))

    # The selected nozzle surrogate version supplies the freestream Mach
    # (and, via nozzle_cfg, the fixed pressure/temperature that condition
    # was trained at) so a surrogate/flight-condition mismatch can't be
    # expressed -- same convention as old-stack's get_surrogate(). Altitude
    # isn't in the surrogate config itself (evaluate_scramjet_point builds
    # freestream from altitude via Atmosphere, not directly from P/T), so it
    # comes from Opt_params, defaulting to the condition v_3_1/v_3_2 were
    # both trained at (30 km -- see hyperProp/Nozzle/models/README.MD).
    nozzle_surrogate, nozzle_cfg = get_surrogate(Opt_params["nozzle_surrogate"])
    aoa = np.deg2rad(Opt_params.get("aoa_deg", 0.0))
    mach = Opt_params.get("mach", nozzle_cfg.FREESTREAM_MACH)
    altitude = Opt_params.get("altitude", 30e3)

    # Three-state DV handling (case_config.py's per-DV `active` flag,
    # threaded through as Opt_params["fixed_dvs"]/["default_dvs"]): active
    # DVs come from xdict directly; a fixed (active=False) DV goes into
    # `overrides`; a "default" DV is simply absent from both, so
    # Parametric_Scramjet2D.check_defaults()'s own hardcoded default
    # applies. Note: equivalence_ratio has no such hardcoded default (unlike
    # the 13 geometry params), so it must be active or fixed, never
    # "default".
    fixed_dvs = Opt_params.get("fixed_dvs", {})
    active_dv_dict = {name: float(val) for name, val in xdict.items()}
    overrides = {name: float(val) for name, val in fixed_dvs.items()}

    # Engine constants, passed explicitly (not left to
    # Parametric_Scramjet2D.check_defaults()) so the case config controls
    # them -- same as HyperMDAO's invars. Neither is a geometry name, so
    # generate_scramjet_sj2d ignores them and evaluate_scramjet_point merges
    # them into the engine-model dict. Engine friction defaults to OFF.
    overrides["viscosityFlg"] = Opt_params.get("engine_visc_flg", False)
    overrides["eta_pr"] = Opt_params.get("eta_pr", 0.8)
    aero_visc_flg = Opt_params.get("aero_visc_flg", True)
    flight_duration = Opt_params.get("flight_duration", 10)

    if save_flg:
        const_vars = dict(overrides)
        const_vars["mach_inf"] = mach
        const_vars["p_inf"] = nozzle_cfg.FREESTREAM_P
        const_vars["T_inf"] = nozzle_cfg.FREESTREAM_T
        const_vars["altitude"] = altitude
        const_vars["aoa"] = aoa
        const_vars["flight_duration"] = flight_duration
        const_vars["aero_visc_flg"] = aero_visc_flg
        const_vars["nozzle_surrogate"] = Opt_params["nozzle_surrogate"]
        return const_vars

    cf_model = get_friction_model(
        Opt_params.get("cf_model", "constant"),
        **Opt_params.get("cf_params", dict(cf_value=0.001)),
    )

    wind_coeffs, quantities, out4sens = calc_vehicle(
        active_dv_dict=active_dv_dict,
        overrides=overrides,
        alpha=aoa,
        mach=mach,
        altitude=altitude,
        nozzle_surrogate=nozzle_surrogate,
        cf_model=cf_model,
        aero_visc_flg=aero_visc_flg,
        flight_duration=flight_duration,
        nozzle_bounds=nozzle_cfg.BOUNDS,
        mesh_params=Opt_params.get("mesh_params"),
        save_geometry=Opt_params.get("save_geometry", False),
        geometry_name=Opt_params.get("geometry_name"),
        verbosity=verbosity,
    )

    return wind_coeffs, quantities, out4sens


def calc_vehicle(
    active_dv_dict,
    overrides,
    alpha,
    mach,
    altitude,
    nozzle_surrogate,
    cf_model,
    aero_visc_flg=True,
    flight_duration=10,
    nozzle_bounds=None,
    mesh_params=None,
    save_geometry=False,
    geometry_name=None,
    verbosity=0,
):
    """
    :param active_dv_dict: the DVs the outer optimiser controls, promoted to
        FWS together (geometry + equivalence_ratio when active -- see
        PROJECT_STATE.md decision 2). Including equivalence_ratio here (not
        just the 13 geometry DVs) is what gives it a real sensitivity
        column; PROJECT_STATE.md decision 12 excludes it from FWS only for
        the deck-building/trajectory-control path (generate_scramjet_decks),
        which doesn't apply to single-point optimisation.
    :param overrides: fixed (active=False) DVs, plain floats.
    :param alpha, mach, altitude: the one flight condition to evaluate at.
    :param nozzle_surrogate: loaded NozzleSurrogateModel (get_surrogate()).
    :param cf_model: loaded hypervisc friction model (get_friction_model()).
    :param aero_visc_flg: whether the friction bucket is added to the total
        force used for CD/T_W_ratio/L_W_ratio/range (evaluate_scramjet_point
        always computes it; the friction-only entries in CATALOG are
        unaffected by this flag).
    :param flight_duration: fuel-budget duration [s] for m_dot_f_ratio.
    :param nozzle_bounds: the nozzle surrogate's BOUNDS (for the in-box diagnostic).
    :param mesh_params: geometry mesh settings (e.g. {"mesh_resolution": 2});
        None -> Scramjet2D_Comp's own default.
    :param save_geometry: write the generated vehicle to a .vtk file.
    :param geometry_name: name for the vehicle / saved file.
    :return: (wind_coeffs, quantities, out4sens)
    """
    try:
        geom_out = generate_scramjet_sj2d(
            active_dv_dict,
            overrides=overrides,
            mesh_params=mesh_params,
            save=save_geometry,
            name=geometry_name,
        )
    except Exception as e:
        raise EvaluationFailure(f"Geometry generation failed: {e}") from e
    vehicle = geom_out["vehicle"]
    fws = geom_out["FWS_param_dict"]
    context = geom_out["context"]

    # S_ref convention matches generate_scramjet_decks / PROJECT_STATE.md
    # decision 4 -- FWS-aware, computed once per design point.
    S_ref = vehicle.mesh_data.calculate_volume() ** (2 / 3)

    # equivalence_ratio must come from fws (active) or overrides (fixed) --
    # see the docstring above and this session's equivalence_ratio
    # discussion. Passing the FWS value (not a plain float) here is what
    # lets its sensitivity propagate through the engine chain the same way
    # the 13 geometry DVs' do.
    equivalence_ratio = fws.get("equivalence_ratio", overrides.get("equivalence_ratio"))

    try:
        result = evaluate_scramjet_point(
            alpha=alpha,
            mach=mach,
            altitude=altitude,
            equivalence_ratio=equivalence_ratio,
            vehicle=vehicle,
            fws=fws,
            context=context,
            overrides=overrides,
            nozzle_surrogate=nozzle_surrogate,
            cf_model=cf_model,
            S_ref=S_ref,
            verbosity=verbosity,
        )
    except Exception as e:
        raise EvaluationFailure(f"Flow/propulsion evaluation failed: {e}") from e

    if result["propulsion_error"] is not None:
        # evaluate_scramjet_point tolerates a failed propulsion chain (the
        # co-design deck grid repairs such nodes); a single-point evaluation
        # has nothing to fall back on, so fail loudly instead.
        raise EvaluationFailure(f"Propulsion chain failed: {result['propulsion_error']}")

    eng_model = result["models"]["eng_model"]
    engine = result["models"]["engine"]
    nozzle = result["models"]["nozzle"]
    freestream = result["freestream"]
    geom = eng_model._geom

    total_vol = vehicle.mesh_data.calculate_volume()
    mass = eng_model.calculate_mass()

    eng_vol = 0.5 * np.abs(
        (geom["C1"].x * geom["C2"].z) - (geom["C1"].z * geom["C2"].x)
        + (geom["C2"].x * geom["E2"].z) - (geom["C2"].z * geom["E2"].x)
        + (geom["E2"].x * geom["E1"].z) - (geom["E2"].z * geom["E1"].x)
        + (geom["E1"].x * geom["C1"].z) - (geom["E1"].z * geom["C1"].x)
    ) * geom["width"]
    internal_vol = total_vol - eng_vol

    # calculate_fuel_mass(self, cells) reads cells.vol, matching HyperMDAO's
    # CellArray -- new-stack's volume comes from a method call
    # (vehicle.mesh_data.calculate_volume()), not a .vol attribute, so this
    # minimal shim bridges the two without changing the ported method.
    fuel_mass = eng_model.calculate_fuel_mass(internal_vol)
    weight = mass * g

    ###########################################################################
    # Total force via AeroResults' own wind-axis conversion.
    # result["aero"] already includes the inlet force (top + cowl_out + inlet);
    # result["propulsion"] is cowl_in + engine + nozzle.
    ###########################################################################
    buckets = [result["aero"], result["propulsion"]]
    if aero_visc_flg:
        buckets.append(result["friction"])
    total = sum_aero_results(
        buckets, freestream, result["aero"].moment_ref_point
    ).update_input(S_ref=S_ref)
    wind_coeffs = total.wind_coeffs(aoa=alpha)  # (CD, CL, CM)
    D, L, _ = total.wind_forces(aoa=alpha)

    L_W_ratio = L / weight
    T_W_ratio = D / weight

    ###########################################################################
    # Fuel budget
    ###########################################################################
    m_dot_f = engine.m_dot_f
    m_dot_f_lim = fuel_mass / flight_duration
    m_dot_f_ratio = m_dot_f / m_dot_f_lim
    # mf_dV = -m_dot_f / T_W_ratio
    mf_dV = -m_dot_f / (T_W_ratio * g)

    ###########################################################################
    # Geometry-derived
    ###########################################################################
    tail_z = geom["N0"].z - geom["T0"].z
    I1, T0 = geom["I1"], geom["T0"]
    theta_nose = np.arccos(
        I1.dot(T0) / (np.sqrt(I1.dot(I1)) * np.sqrt(T0.dot(T0)))
    )

    P_exit_ratio = nozzle.P_exit / freestream.p

    ###########################################################################
    # Range (Breguet; aoa assumed 0, as in HyperMDAO):
    #   R = V * Isp * (L+Tz) / sqrt(D^2 + Tz^2) * ln(W0/W1),  Isp = T/(g*m_dot_f)
    # Thrust = cowl_in + engine + nozzle; Aero = aero bucket (incl. inlet)
    # + friction. NOTE: HyperMDAO's Aero was Top+Cowl+Other panels + inlet,
    # so a range mismatch is first suspect for this grouping.
    ###########################################################################
    Thrust = result["propulsion"].force
    Aero = result["aero"].force
    if aero_visc_flg:
        Aero = Aero + result["friction"].force
    weight_1 = weight - fuel_mass * g

    # Isp = np.sqrt(Thrust.dot(Thrust)) / (g * m_dot_f)
    # range_ = (
    #         freestream.v * Isp
    #         * (-Aero.z - Thrust.z) / np.sqrt(Aero.x ** 2 + Thrust.z ** 2)
    #         * np.log(weight / weight_1)
    # )
    Isp = np.abs(Thrust.x) / (g * m_dot_f)
    range_ = (
        freestream.v * Isp
        * (-Aero.z - Thrust.z) / np.sqrt(Aero.x**2)
        * np.log(weight / weight_1)
    )

    # quantity_catalog.CATALOG (shared with co-design -- deliberately NOT
    # extended here: scramjet_deck builds one interpolant per entry) plus
    # the HyperMDAO-parity quantities computed above.
    quantities = {name: q.extract(result, alpha) for name, q in CATALOG.items()}
    quantities["mass"] = mass
    quantities["weight"] = weight
    quantities["fuel_mass"] = fuel_mass
    quantities["CD"] = wind_coeffs[0]
    quantities["CL"] = wind_coeffs[1]
    quantities["T_W_ratio"] = T_W_ratio
    quantities["L_W_ratio"] = L_W_ratio
    quantities["m_dot_f_ratio"] = m_dot_f_ratio
    quantities["mf_dV"] = mf_dV
    quantities["range"] = range_
    quantities["Outflow_M"] = quantities["M_out"]
    quantities["Outflow_P"] = quantities["P_out"]
    quantities["Outflow_T"] = quantities["T_out"]
    quantities["P_exit_ratio"] = P_exit_ratio
    quantities["tail_z"] = tail_z
    quantities["internal_vol"] = internal_vol
    quantities["theta_nose"] = theta_nose
    quantities["volume"] = total_vol
    check_quantities(quantities)

    range_mod = {
        "Thrust": Thrust,
        "Aero": Aero,
        "Isp": Isp,
        "weight_0": weight,
        "weight_1": weight_1,
        "velocity": freestream.v,
        "m_dot_f": m_dot_f,
        "mass_f": fuel_mass,
        "range": range_,
    }

    # NOTE: Opt_Main hands pyoptsparse only record.slim_out4sens(out4sens) --
    # pyoptsparse pickles every value in funcs into its history file, so
    # nothing unpicklable (e.g. the Vehicle) may be in that subset.
    out4sens = {
        "quantities": quantities,
        "result": result,
        "eng_model": eng_model,
        "fws": fws,
        "context": context,
        "freestream": freestream,
        "total": total,
        "wind_coeffs": wind_coeffs,
        "S_ref": S_ref,
        "mass": mass,
        "weight": weight,
        "fuel_mass": fuel_mass,
        "total_vol": total_vol,
        "eng_vol": eng_vol,
        "m_dot_f_lim": m_dot_f_lim,
        "range_mod": range_mod,
    }
    out4sens.update(build_record(result, nozzle_bounds, alpha))

    problems = check_physical(quantities, out4sens["out4mat"])
    if problems:
        raise EvaluationFailure("Non-physical solution: " + "; ".join(problems[:6]))

    return wind_coeffs, quantities, out4sens


if __name__ == "__main__":
    pass
