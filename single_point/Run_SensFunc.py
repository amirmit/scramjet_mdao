from HyperPro.Sensitivity import HP_Sensitivity
from HyperPro.utilities import get_number

from hypervehicle.geometry import Vector3
from hypervehicle.geometry.autodiff import get_sens
from hypervisc import get_friction_model, friction_drag_sens
from pysagas.cfd import SensDeck
from pysagas.sensitivity.calculator_vec import sensitivity_calculator_vec

from hypervehicle.geometry import FloatWithSens
from problem_catalog import CATALOG
from scipy.constants import g

import numpy as np


def Run_SensFunc(out4sens, Opt_params):

    ####### Control #########
    verbosity = 0

    out4sens = normalize_out4sens(out4sens)

    ####### Input from ObjFunc Scipy #########
    freestream = out4sens["freestream"]
    flow_state = out4sens["flow_state"]
    cells = out4sens["cells"]
    inlet_model = out4sens["inlet_model"]
    cowl_model = out4sens["cowl_model"]
    engine_model = out4sens["engine_model"]
    parameters = out4sens["parameters"]
    force_coeffs = out4sens["force_coeffs"]
    Sref = out4sens["Sref"]
    comp_force = out4sens["comp_force"]
    weight = out4sens["weight"]
    eng_vol = out4sens["eng_vol"]
    cat_values = out4sens["cat_values"]
    range_mod = out4sens["range_mod"]

    #####################################
    ###### HyperPro - Sensitivity #######
    #####################################

    prop_sens = HP_Sensitivity(
        inlet_model=inlet_model,
        cowl_model=cowl_model,
        engine_model=engine_model,
        parameter_dict=parameters,
        verbosity=verbosity,
        save_to_csv=0,
    )
    HP_sens = prop_sens.generate_sensitivities()

    ###################################
    ######## PySAGAS Sens #############
    ###################################

    PS_sens = sensitivity_calculator_vec(
        cells=cells,
        freestream=freestream,
        flow_state=flow_state,
    )
    nozzle_force = comp_force["nozzle"]
    NZ_sens = {
        "Fx": get_sens(nozzle_force.x),
        "Fy": get_sens(nozzle_force.y),
        "Fz": get_sens(nozzle_force.z),
    }

    ###################################
    ############ HyperVisc ############
    ###################################
    # Mirrors calc_vehicle()'s Df computation in Run_ObjFunc.py -- same
    # cached model instance (name+params), same aero_visc_flg gate, so
    # value and gradient never drift apart.
    if Opt_params.get("aero_visc_flg", True):
        cf_model = get_friction_model(
            Opt_params.get("cf_model", "constant"),
            **Opt_params.get("cf_params", dict(cf_value=0.001)),
        )
        Visc_sens = friction_drag_sens(cells, freestream, cf_model)
    else:
        Visc_sens = np.zeros_like(cells.dAdp_int)




    HP_force_sens, PS_force_sens, NZ_force_sens, Visc_force_sens = refactor_force_sens(
        HP_sens, PS_sens, NZ_sens, Visc_sens, parameters
    )
    alpha = np.deg2rad(engine_model.freestream.aoa)
    q = engine_model.freestream.q

    HP_wind_force_sens = {}
    PS_wind_force_sens = {}
    NZ_wind_force_sens = {}
    Visc_wind_force_sens = {}
    sens_vec = {}
    sens_coeffs = {}
    for i, param in enumerate(parameters):
        HP_wind_force_sens[param] = Body_2_Wind(HP_force_sens[param], alpha)
        PS_wind_force_sens[param] = Body_2_Wind(PS_force_sens[param], alpha)
        NZ_wind_force_sens[param] = Body_2_Wind(NZ_force_sens[param], alpha)
        Visc_wind_force_sens[param] = Body_2_Wind(Visc_force_sens[param], alpha)
        sens_vec[param] = (
            HP_wind_force_sens[param]
            + PS_wind_force_sens[param]
            + NZ_wind_force_sens[param]
            + Visc_wind_force_sens[param]
        )

        sens_coeffs[param] = (
            calc_force_coeffs(sens_vec[param], Sref, q)
            - 2 / 3 * force_coeffs / cells.vol * cells.dvoldp[i]
        )

    ctx = {}
    ctx["sens_coeffs"] = sens_coeffs
    ctx["cells"] = cells
    ctx["HP_sens"] = HP_sens
    ctx["cat_values"] = cat_values

    ##################################
    #### Derived-quantity sens ##########
    ##################################
    T_W_ratio_sens = {}
    L_W_ratio_sens = {}
    m_dot_f_ratio_sens = {}
    internal_vol_sens = {}
    mf_dV_sens = {}

    mdf_model = engine_model.mdf_model

    for i, param in enumerate(parameters):

        T_W_ratio_sens[param] = (
            sens_vec[param].x / get_number(weight)
            - cat_values["T_W_ratio"].number / get_number(weight) * get_sens(weight)[i]
        )

        L_W_ratio_sens[param] = (
            sens_vec[param].z / get_number(weight)
            - cat_values["L_W_ratio"].number / get_number(weight) * get_sens(weight)[i]
        )

        m_dot_f_ratio_sens[param] = (
            HP_sens[param].m_dot_f / mdf_model["m_dot_f_lim"]
            - get_number(mdf_model["m_dot_f_ratio"])
            / mdf_model["m_dot_f_lim"]
            * 35
            * 0.1
            / mdf_model["flight_duration"]
            * cells.dvoldp[i]
        )

        internal_vol_sens[param] = (
            cells.dvoldp[i] - get_sens(eng_vol)[i]
        )

        mf_dV_sens[param] = -(
            get_sens(engine_model.m_dot_f)[i]
            / cat_values["T_W_ratio"].number
            - T_W_ratio_sens[param]
            * engine_model.m_dot_f.number
            / cat_values["T_W_ratio"].number ** 2
        )

    range_sens = range_equation_sens(
        range_mod=range_mod,
        prop_sens=prop_sens,
        PS_force_sens=PS_force_sens,
        NZ_force_sens=NZ_force_sens,
        Visc_force_sens=Visc_force_sens,
        params=parameters,
        cells=cells,
    )


    ctx["T_W_ratio_sens"] = T_W_ratio_sens
    ctx["L_W_ratio_sens"] = L_W_ratio_sens
    ctx["m_dot_f_ratio_sens"] = m_dot_f_ratio_sens
    ctx["internal_vol_sens"] = internal_vol_sens
    ctx["mf_dV_sens"] = mf_dV_sens
    ctx["mdf_model"] = mdf_model
    ctx["range_sens"] = range_sens


    # Raw (unscaled) per-catalog-entry sensitivity -- weighting and
    # per-entry scale (from the case config's objective/constraints dicts)
    # are applied by Opt_Main.sensfunc, the same place funcs[name] values
    # get scaled, so both use the exact same config-driven scale factors.
    sens = {
        name: {param: q.sens(ctx, param, i) for i, param in enumerate(parameters)}
        for name, q in CATALOG.items()
    }

    return sens


def normalize_out4sens(out4sens):
    if isinstance(out4sens, np.ndarray):
        return out4sens[()]
    return out4sens


def refactor_force_sens(HP_sens, PS_sens, NZ_sens, Visc_sens, params):
    HP_force_sens = {}
    PS_force_sens = {}
    NZ_force_sens = {}
    Visc_force_sens = {}

    for i, param in enumerate(params.keys()):
        HP_force_sens[param] = Vector3(
            x=HP_sens[param]["Fx"],
            y=HP_sens[param]["Fy"],
            z=HP_sens[param]["Fz"],
        )
        PS_force_sens[param] = Vector3(
            x=PS_sens["dForcedp"][i][0],
            y=PS_sens["dForcedp"][i][1],
            z=PS_sens["dForcedp"][i][2],
        )

        NZ_force_sens[param] = Vector3(
            x=NZ_sens["Fx"][i],
            y=NZ_sens["Fy"][i],
            z=NZ_sens["Fz"][i],
        )

        Visc_force_sens[param] = Vector3(
            x=-Visc_sens[i],
            y=0.0,
            z=0.0,
        )
    return HP_force_sens, PS_force_sens, NZ_force_sens, Visc_force_sens


def Body_2_Wind(body: Vector3, alpha=0):
    """
    Transformes from body axis system to wind axis system - Fx, Fz -> L, D
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

def range_equation_sens(
        range_mod,
        prop_sens,
        PS_force_sens,
        NZ_force_sens,
        Visc_force_sens,
        params,
        cells,
):

    n = len(params)
    PS_sens_x = np.zeros(n)
    PS_sens_z = np.zeros(n)
    NZ_sens_x = np.zeros(n)
    NZ_sens_z = np.zeros(n)
    Visc_sens_x = np.zeros(n)
    Visc_sens_z = np.zeros(n)
    inlet_sens_x = np.zeros(n)
    inlet_sens_z = np.zeros(n)
    cowl_sens_x = np.zeros(n)
    cowl_sens_z = np.zeros(n)
    eng_sens_x = np.zeros(n)
    eng_sens_z = np.zeros(n)
    Thrust_sens_x = np.zeros(n)
    Thrust_sens_z = np.zeros(n)
    Aero_sens_x = np.zeros(n)
    Aero_sens_z = np.zeros(n)
    mass_f_sens = np.zeros(n)

    for i, param in enumerate(params.keys()):
        PS_sens_x[i] = PS_force_sens[param].x
        PS_sens_z[i] = PS_force_sens[param].z
        inlet_sens_x[i] = prop_sens.inlet_sens[param]["Fx"]
        inlet_sens_z[i] = prop_sens.inlet_sens[param]["Fz"]
        cowl_sens_x[i] = prop_sens.cowl_sens[param]["Fx"]
        cowl_sens_z[i] = prop_sens.cowl_sens[param]["Fz"]
        eng_sens_x[i] = prop_sens.eng_sens[param]["Fx"]
        eng_sens_z[i] = prop_sens.eng_sens[param]["Fz"]
        NZ_sens_x[i] = NZ_force_sens[param].x
        NZ_sens_z[i] = NZ_force_sens[param].z
        Visc_sens_x[i] = Visc_force_sens[param].x
        Visc_sens_z[i] = Visc_force_sens[param].z
        mass_f_sens[i] = 35 * 0.1 * cells.dvoldp[i]

        Thrust_sens_x[i] = cowl_sens_x[i] + eng_sens_x[i] + NZ_sens_x[i]
        Thrust_sens_z[i] = cowl_sens_z[i] + eng_sens_z[i] + NZ_sens_z[i]
        Aero_sens_x[i] = PS_sens_x[i] + inlet_sens_x[i] + Visc_sens_x[i]
        Aero_sens_z[i] = PS_sens_z[i] + inlet_sens_z[i] + Visc_sens_z[i]

    Thrust_sens = Vector3(
        x=FloatWithSens(number=range_mod["Thrust"].x.number, sens=Thrust_sens_x),
        y=0,
        z=FloatWithSens(number=range_mod["Thrust"].z.number, sens=Thrust_sens_z),
    )
    Aero_sens = Vector3(
        x=FloatWithSens(number=range_mod["Aero"].x.number, sens=Aero_sens_x),
    y = 0,
    z = FloatWithSens(number=range_mod["Aero"].z.number, sens=Aero_sens_z),
    )

    mass_f = FloatWithSens(number=range_mod["mass_f"], sens=mass_f_sens)
    weight_1 = range_mod["weight_0"] - mass_f * g

    Isp = Thrust_sens.norm / (g * range_mod["m_dot_f"])

    range_sens = (
            range_mod["velocity"] * Isp
            * (-Aero_sens.z - Thrust_sens.z) / np.sqrt(Aero_sens.x ** 2 + Thrust_sens.z ** 2)
            * np.log(range_mod["weight_0"] / weight_1)
    )

    return range_sens