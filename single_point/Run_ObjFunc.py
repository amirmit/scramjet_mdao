import importlib
import os
import sys
from pathlib import Path
import numpy as np
from scipy.constants import g

# Eng_Build.py lives in the sibling vehicle/ folder, shared with co_design/
# (see README "Shared vehicle definition") rather than copied per-driver.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vehicle"))

import HyperPro.nozzle_surrogate
from Eng_Build import Parametric_EngModel
from HyperPro.nozzle_surrogate import NozzleSurrogate
from HyperPro.utilities import get_number
from hypervehicle.geometry import Vector3
from hypervisc import get_friction_model, friction_drag
from pysagas import CellArray
from pysagas.cfd import OPMVec
from pysagas.flow import FlowState, InFlowStateVec, FlowResults

from problem_catalog import CATALOG

##############################################################################
# Nozzle surrogate registry. Each version ("v3_1" = M6.5, "v3_2" = M10, ...)
# is loaded and cached at most once per process, the first time a case
# config selects it via Opt_params["nozzle_surrogate"] -- not at import
# time, so switching versions is a config change, not a code change. The
# selected version's config.py also supplies the freestream conditions and
# the surrogate's training-box BOUNDS (see get_surrogate() below).
##############################################################################

_HYPERPRO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(HyperPro.nozzle_surrogate.__file__))
)
# SurrogateGen sits alongside, not inside, the installed HyperPro package --
# imported by path, same as the pre-registry code did for v3_1 only.
if _HYPERPRO_ROOT not in sys.path:
    sys.path.insert(0, _HYPERPRO_ROOT)

_SURROGATE_CACHE = {}


def get_surrogate(version):
    """Returns (NozzleSurrogate, config_module) for `version` (e.g. "v3_1",
    "v3_2"). Cached: repeated calls for the same version within a process
    reuse the already-deserialized model instead of reloading the pickle.
    """
    if version not in _SURROGATE_CACHE:
        cfg = importlib.import_module(f"SurrogateGen.nozzle_v3.{version}.config")
        pkl_path = os.path.join(
            _HYPERPRO_ROOT,
            "SurrogateGen",
            "nozzle_v3",
            version,
            "nozzle_v3_surrogate.pkl",
        )
        _SURROGATE_CACHE[version] = (NozzleSurrogate(pkl_path=pkl_path), cfg)
    return _SURROGATE_CACHE[version]


def Run_ObjFunc(xdict, Opt_params, save_flg=False):
    """
    :param xdict: Design variables dictionary
    :param Opt_params: shared run constants (nozzle_surrogate, aoa_deg, flight_duration, eta_pr, ...)
    :param save_flg: Returns run constants for saving record
    :return:
    """

    ####### Control #########
    verbosity = 0

    ############ flow state ##############
    # The selected nozzle surrogate version supplies the freestream
    # (Mach/pressure/temperature) so a surrogate/flight-condition mismatch
    # can't be expressed -- see get_surrogate() and each version's config.py.
    nozzle_surrogate, nozzle_cfg = get_surrogate(Opt_params["nozzle_surrogate"])
    aoa = np.deg2rad(Opt_params.get("aoa_deg", 0.0))
    freestream = FlowState(
        mach=nozzle_cfg.FREESTREAM_MACH,
        pressure=nozzle_cfg.FREESTREAM_P,
        temperature=nozzle_cfg.FREESTREAM_T,
        aoa=np.rad2deg(aoa),
    )


    ######## input ###########
    invars = {
        #################
        #### Nominal ####
        #################
        "inlet_length_1": 1.2,
        "inlet_length_2": 0.88,
        "inlet_theta_1": np.deg2rad(10),
        "inlet_theta_2": np.deg2rad(6.93),
        "throat_height": 0.1,
        "engine_tilt_ang": np.deg2rad(0),
        "cowl_LE": 0.1,
        "engine_length": 0.5,
        "isolator_length": 0.5,
        "engine_h5_h3": 1.0,
        "nozzle_length": 1.0,
        "nozzle_angle": 0.15788088456676624,
        "nozzle_mid_frac": 0.5,
        ######## Engine #########
        "equivalence_ratio": 0.2,
        "flight_duration": Opt_params["flight_duration"],  # sec
        "eta_pr": Opt_params["eta_pr"],
        ####### Flow #########
        "freestream": freestream,
        ####### Control ########
        "viscosityFlg": Opt_params.get("engine_visc_flg", False),
    }
    # Three-state DV handling (case_config.py's per-DV `active` flag,
    # threaded through here as Opt_params["fixed_dvs"]/["default_dvs"]):
    # active DVs come from xdict as usual; a fixed (active=False) DV
    # overrides this nominal dict with its config x0; a "default" DV is
    # dropped from invars entirely so Eng_Build's own __init__ default for
    # that attribute applies instead of either nominal value here.
    fixed_dvs = Opt_params.get("fixed_dvs", {})
    default_dvs = Opt_params.get("default_dvs", ())
    invars = {
        key: (
            float(xdict[key])
            if key in xdict
            else float(fixed_dvs[key]) if key in fixed_dvs else value
        )
        for key, value in invars.items()
        if key not in default_dvs
    }

    ######## Sensitivities ############
    # parameters = xdict.copy()
    parameters = {key: float(value) for key, value in xdict.items()}

    ####### Return constants for saving the optimization case ################
    if save_flg:
        const_vars = {
            key: invars[key]
            for key in invars
            if key not in parameters and key != "freestream"
        }
        const_vars["mach_inf"] = nozzle_cfg.FREESTREAM_MACH
        const_vars["p_inf"] = nozzle_cfg.FREESTREAM_P
        const_vars["T_inf"] = nozzle_cfg.FREESTREAM_T
        const_vars["aoa"] = aoa
        const_vars["nozzle_surrogate"] = Opt_params["nozzle_surrogate"]
        return const_vars

    ########## Calculate Vehicle ##########
    overall_force, quantities, out4sens = calc_vehicle(
        invars=invars,
        parameters=parameters,
        nozzle_surrogate=nozzle_surrogate,
        nozzle_bounds=nozzle_cfg.BOUNDS,
        save_vtk=False,
        aero_visc_flg=Opt_params.get("aero_visc_flg", True),
        cf_model_name=Opt_params.get("cf_model", "constant"),
        cf_params=Opt_params.get("cf_params", dict(cf_value=0.001)),
    )

    return overall_force, quantities, out4sens


def calc_vehicle(
    invars,
    parameters,
    nozzle_surrogate,
    nozzle_bounds,
    save_vtk=False,
    freestream=None,
    verbosity=0,
    aero_visc_flg=True,
    cf_model_name="constant",
    cf_params=None,
):
    """
    :param invars: input variables for run (constants)
    :param parameters: design variables for sensitivity calculation
    :param nozzle_surrogate: NozzleSurrogate instance from get_surrogate()
    :param nozzle_bounds: that surrogate's training-box BOUNDS dict
    :param save_vtk: Save a VTK file (for SinglePiont post processing)
    :param freestream: freestream variables (only for running post processing)
    :param verbosity:
    :param aero_visc_flg: external-body friction drag on/off (separate from
        invars["viscosityFlg"], which controls Engine_1D's internal
        isolator/combustor skin friction)
    :param cf_model_name: name passed to hypervisc.get_friction_model()
    :param cf_params: kwargs passed to hypervisc.get_friction_model()
    :return:
    """
    if cf_params is None:
        cf_params = dict(cf_value=0.001)
    if freestream is None:
        freestream = invars["freestream"]
    else:
        invars["freestream"] = freestream

    ################################
    ######## hypervehicle #########
    ###############################
    parametric_generator = Parametric_EngModel(verbosity=verbosity, **invars)
    parametric_generator.dvdp(parameters)
    eng_model = parametric_generator.create_instance()
    eng_model.generate()
    eng_model.to_vtk("eng_model", save_file=save_vtk, verbosity=verbosity)

    mesh = eng_model.components[0].mesh.copy()

    indexes = mesh.faces.reshape(-1, 4)
    points = mesh.points[indexes[:, 1:4]].reshape(-1, 9)
    dvdp = np.array([mesh.point_data["dvd" + k] for k in parameters.keys()])
    dvdp = dvdp[:, indexes[:, 1:4], :]
    dvdp = dvdp.reshape(len(parameters), -1, 9)
    cells = CellArray(points, dvdp, mesh)

    ###################################
    ####### HyperPro - Inlet ##########
    ###################################
    eng_model.inlet = parametric_generator.create_inlet()
    conserve_2, inlet_force = eng_model.inlet.calc_inlet()
    # eng_model.inlet._PlotInlet2D()
    # exit()

    ###################################
    ####### HyperPro - Cowl ##########
    ###################################
    eng_model.cowl = parametric_generator.create_cowl(conserve_2)
    conserve_3, cowl_force = eng_model.cowl.calc_cowl()

    ###################################
    ####### HyperPro - Engine #########
    ###################################
    eng_model.engine = parametric_generator.create_engine(conserve_3)
    outflow, engine_force = eng_model.engine.calc_engine()

    ###################################
    ####### HyperPro - Nozzle #########
    ###################################
    eng_model.nozzle = parametric_generator.create_nozzle(outflow, nozzle_surrogate)
    nozzle_force = eng_model.nozzle.calc_nozzle()

    nozzle_inputs = eng_model.nozzle.surrogate_inputs
    oob_vars = [
        k
        for k, v in nozzle_inputs.items()
        if not (nozzle_bounds[k][0] <= v <= nozzle_bounds[k][1])
    ]
    nozzle_diag = {
        **nozzle_inputs,
        "Fx_p": eng_model.nozzle.Fx_p_nom,
        "Fy_p": eng_model.nozzle.Fy_p_nom,
        "P_exit": get_number(eng_model.nozzle.P_exit),
        "in_box": len(oob_vars) == 0,
        "oob_vars": oob_vars,
    }

    ###################################
    ############# PySAGAS #############
    ###################################

    flow_solver = OPMVec()

    aero_result, flow_state, Comp_force = flow_solver.solve(
        cells=cells,
        freestream=freestream,
    )
    PS_force = FlowResults(
        freestream=freestream,
        net_force=Vector3(
            aero_result["F"][0], aero_result["F"][1], aero_result["F"][2]
        ),
        net_moment=Vector3(
            aero_result["M"][0], aero_result["M"][1], aero_result["M"][2]
        ),
    )

    ###################################
    ############ HyperVisc ############
    ###################################
    if aero_visc_flg:
        cf_model = get_friction_model(cf_model_name, **cf_params)
        Df = friction_drag(cells, freestream, cf_model)
    else:
        Df = 0.0
    Visc_force = FlowResults(
        freestream=freestream,
        net_force=Vector3(x=-Df, y=0, z=0),
        net_moment=Vector3(x=0, y=0, z=0),
    )

    ################################
    ############# Mass #############
    ################################
    eng_model.mass = parametric_generator.calculate_mass()
    eng_model.weight = eng_model.mass * g

    ###################################
    ############# m_dot_f #############
    ###################################

    flight_duration = invars["flight_duration"]
    mass_f = parametric_generator.calculate_fuel_mass(cells)
    m_dot_f_lim = mass_f / flight_duration
    m_dot_f_ratio = eng_model.engine.m_dot_f / m_dot_f_lim
    mdf_model = {
        "m_dot_f": eng_model.engine.m_dot_f,
        "m_dot_f_lim": m_dot_f_lim,
        "m_dot_f_ratio": m_dot_f_ratio,
        "flight_duration": flight_duration,
        "mass_f": mass_f,
    }
    eng_model.engine.mdf_model = mdf_model

    ############################################
    ############# Internal volume ##############
    ############################################

    eng_vol = 0.5 * np.abs(
        (parametric_generator._geom["C1"].x * parametric_generator._geom["C2"].z) -
        (parametric_generator._geom["C1"].z * parametric_generator._geom["C2"].x) +
        (parametric_generator._geom["C2"].x * parametric_generator._geom["E2"].z) -
        (parametric_generator._geom["C2"].z * parametric_generator._geom["E2"].x) +
        (parametric_generator._geom["E2"].x * parametric_generator._geom["E1"].z) -
        (parametric_generator._geom["E2"].z * parametric_generator._geom["E1"].x) +
        (parametric_generator._geom["E1"].x * parametric_generator._geom["C1"].z) -
        (parametric_generator._geom["E1"].z * parametric_generator._geom["C1"].x)
    ) * parametric_generator._geom["width"]

    internal_vol = cells.vol - eng_vol

    ###################################
    ############# Forces ##############
    ###################################

    overall_force = (
        inlet_force.net_force
        + cowl_force.net_force
        + engine_force.net_force
        + nozzle_force.net_force
        + PS_force.net_force
        + Visc_force.net_force
    )

    wind_forces = Body_2_Wind(overall_force, np.deg2rad(freestream.aoa))
    Sref = cells.vol ** (2 / 3)
    wind_coeffs = calc_force_coeffs(wind_forces, Sref=Sref, q=freestream.q)

    L_W_ratio = wind_forces.z / eng_model.weight
    T_W_ratio = wind_forces.x / eng_model.weight

    nose_dot = parametric_generator._geom["I1"].dot(parametric_generator._geom["T0"])
    nose_norm = (
        parametric_generator._geom["I1"].norm * parametric_generator._geom["T0"].norm
    )
    theta_nose = np.arccos(nose_dot / nose_norm)

    Comp_force["inlet"] = inlet_force.net_force
    Comp_force["internal_cowl"] = cowl_force.net_force
    Comp_force["engine"] = engine_force.net_force
    Comp_force["nozzle"] = nozzle_force.net_force
    Comp_force["Visc_force"] = Visc_force.net_force
    Comp_force["Aero"] = Vector3.from_coordinates(
        Comp_force["Top"] + Comp_force["Cowl"] + Comp_force["Other"]
    )
    Comp_force["Uninstalled_Prop"] = (
        Comp_force["nozzle"]
        + Comp_force["inlet"]
        + Comp_force["internal_cowl"]
        + Comp_force["engine"]
    )

    ##########################################
    ################# Range ##################
    ##########################################
    # Todo: AOA is assumed to be 0!
    """
    R = V * Isp * (L+Tz) / sqrt(D^2 + Tz^2) * ln(W0/W1)
    Isp = T / (g*m_dot_f)
    T = sqrt(Tx^2 + Tz^2)
    Assumptions: 
        1. W = L+Tz
        2. D = Tx
    """
    m_dot_f = eng_model.engine.m_dot_f
    Thrust = Comp_force["internal_cowl"] + Comp_force["engine"] + Comp_force["nozzle"]
    Aero = Comp_force["Aero"] + Comp_force["inlet"] + Comp_force["Visc_force"]
    Isp = Thrust.norm / (g * m_dot_f)
    weight_0 = eng_model.weight
    weight_1 = weight_0 - mass_f * g
    velocity = freestream.v

    range = (
        velocity * Isp
        * (-Aero.z - Thrust.z) / np.sqrt(Aero.x**2 + Thrust.z**2)
        * np.log(weight_0 / weight_1)
    )

    range_mod = {
        "Thrust": Thrust,
        "Aero": Aero,
        "Isp": Isp,
        "weight_0": weight_0,
        "weight_1": weight_1,
        "velocity": velocity,
        "m_dot_f": m_dot_f,
        "mass_f": mass_f,
        "range": range,
    }

    ##########################################
    ############# Other outputs ########### ###
    ##########################################

    P_exit_ratio = eng_model.nozzle.P_exit / freestream.P
    tail_z = parametric_generator._geom["N0"].z - parametric_generator._geom["T0"].z
    mf_dV = -eng_model.engine.m_dot_f / T_W_ratio

    ###################################
    ############# Output ##############
    ###################################

    out4mat = Prepare_out4mat(eng_model, Comp_force)

    cat_values = {
        "T_W_ratio": T_W_ratio,
        "L_W_ratio": L_W_ratio,
        "internal_vol": internal_vol,
        "theta_nose": theta_nose,
        "P_exit_ratio": P_exit_ratio,
        "tail_z": tail_z,
        "mf_dV": mf_dV,
        "range": range,
    }

    out4sens = {
        "freestream": freestream,
        "flow_state": flow_state,
        "cells": cells,
        "inlet_model": eng_model.inlet,
        "cowl_model": eng_model.cowl,
        "engine_model": eng_model.engine,
        "parameters": parameters,
        "force_coeffs": wind_coeffs,
        "Sref": Sref,
        "geom": parametric_generator._geom,
        "weight": eng_model.weight,
        "comp_force": Comp_force,
        "out4mat": out4mat,
        "nozzle_diag": nozzle_diag,
        "eng_vol": eng_vol,
        "cat_values": cat_values,
        "range_mod": range_mod,
    }

    # Every catalog quantity, computed regardless of which ones a case
    # config actually activates as objective/constraints (cheap -- see
    # problem_catalog.py).
    quantities = {name: q.value(out4sens) for name, q in CATALOG.items()}

    return wind_coeffs, quantities, out4sens


def Body_2_Wind(body: Vector3, alpha=0):
    """
    Transforms from body axis system to wind axis system - Fx, Fz -> L, D
    alpha in RAD!
    """
    wind = Vector3(
        x=-body.x * np.cos(alpha) - body.z * np.sin(alpha),
        y=body.y,
        z=body.x * np.sin(alpha) - body.z * np.cos(alpha),
    )
    return wind


def calc_force_coeffs(force: Vector3, Sref, q):
    """Calculates the force coefficients"""
    Coeff = Vector3(
        x=force.x / (Sref * q),
        y=force.y / (Sref * q),
        z=force.z / (Sref * q),
    )
    return Coeff


def Prepare_out4mat(eng_model, Comp_force):
    flowstates = {
        "freestream": eng_model.inlet.states[0],
        "s1": eng_model.inlet.states[1],
        "s2": eng_model.inlet.states[2],
        "C": eng_model.cowl.flowstate_2,
        "I": eng_model.cowl.flowstate_3,
        "E": eng_model.engine.outflow,
    }
    out4mat = {}
    for key, state in flowstates.items():
        out4mat[key] = {
            "M": state.M,
            "T": state.T,
            "P": state.P,
            "Pt": state.Pt,
            "Tt": state.Tt,
            "h": state.Cp * state.T,
            "v": state.v,
            "direction": state.direction,
            "pressure_ratio": state.P / flowstates["freestream"].P,
            "temperature_ratio": state.T / flowstates["freestream"].T,
            "pressure_recovery": state.Pt / flowstates["freestream"].Pt,
        }
    STAs = {
        "C": eng_model.cowl.conserve_2,
        "I": eng_model.cowl.conserve_3,
        "E": eng_model.engine.axial_conserve_5,
    }
    for key, STA in STAs.items():
        out4mat[key]["m_dot"] = STA.m_dot
        out4mat[key]["mom"] = STA.mom
        out4mat[key]["tE"] = STA.tE

    out4mat["E"]["m_dot_f"] = eng_model.engine.mdf_model["m_dot_f"]
    eta_th = (
        Comp_force["Uninstalled_Prop"].x
        * out4mat["freestream"]["v"]
        / (out4mat["E"]["m_dot_f"] * eng_model.engine.h_pr)
    )
    out4mat["E"]["eta_th"] = eta_th
    # print(f"eta_th: {eta_th}")
    return out4mat

