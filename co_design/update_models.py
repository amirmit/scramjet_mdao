import pickle
from autodiff import generate_parameters_with_sensitivity, SensMode
# from hyperrocket.rocket import Rocket, SolidRocket
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

from codesignTools import logger, VehicleProp
from geometry_generation import fairing_diameter
from mass_model import (
    HV_mass_function,
    CoMModel,
    CoMModelCombined,
    boost_components_mass_function,
    fuel_mass_function,
    mass_function_add,
)
from hypervehicle import Vehicle
from aerodeck import generate_aerodeck, generate_aerodeck_first_stage_fairing, generate_aerodeck_first_stage
from thermal_model import nose_heat_flux, zero_heat_flux


def update_models(
    parameter_dict: dict[str, float],
    vehicle: dict[str, Vehicle],
    aero_inputs: dict[str, dict[str, np.ndarray]],
    aero_outputs: list[str],
    mass_opts: dict[str, dict[str, float]],
    HV_fuel_opts: dict[str, dict[str, float]],
    plot_aero=True,
    D_fairing=None,
    HV_type="HGV",
    HV_engine=None,
    boost=None,
) -> dict[str, Callable]:
    logger.info("Updating models")
    FWS_parameters = generate_parameters_with_sensitivity(parameter_dict, SensMode.JACOBIAN)

    """ HV Vehicle """
    logger.info("Updating HV models")
    HV_mass_fun, com_model_HV = HV_mass_function(
        vehicle["HV"], parameter_dict, mass_opts[HV_type], FWS_parameters, vehicle_cs=vehicle["HV_cs"]
    )
    mass_model = {"HV": HV_mass_fun}
    com_model = {"HV": com_model_HV}

    aero_sm_HV, stab_sm_HV = generate_aerodeck(
        vehicle=vehicle["HV"],
        aero_inputs=aero_inputs["HV"],
        aero_outputs=aero_outputs,
        parameters=parameter_dict,
        com_model=com_model_HV,
        plot_aero=plot_aero,
    )
    aero_model = {"HV": aero_sm_HV}
    stab_model = {"HV": stab_sm_HV}

    thermal_model_HV = nose_heat_flux(parameter_dict)
    thermal_model = {"HV": thermal_model_HV}

    thrust_model = {}

    """ Spaceplane """
    if HV_engine:
        logger.info("Updating HV propulsion models")
        HV_thrust = rocket_thrust_model(HV_engine)
        HV_fuel_mass_fun = fuel_mass_function(
            vehicle["HV"], parameter_dict, mass_opts[HV_type], HV_fuel_opts[HV_type]
        )
        thrust_model["HV"] = HV_thrust
        mass_model["HV_fuel"] = HV_fuel_mass_fun

    """ Boost """
    if boost:
        logger.info("Updating boost models")
        if isinstance(boost, Rocket):
            thrust_boost, dry_mass_fun, prop_mass_fun, com_model_rocket = (
                generate_liquid_rocket(parameter_dict, boost)
            )
        elif isinstance(boost, SolidRocket):
            thrust_boost, dry_mass_fun, prop_mass_fun, com_model_rocket = (
                generate_solid_rocket(parameter_dict, boost)
            )
        else:
            raise ValueError(f"{boost} is not of {type(Rocket)} or {type(SolidRocket)}")

        thrust_model["boost"] = thrust_boost
        mass_model["boost_prop"] = prop_mass_fun


        if D_fairing:
            boost_components_mass_fun, com_model_boost_comps = (
                boost_components_mass_function(
                    vehicle["boost"],
                    vehicle["rocket_body"],
                    parameter_dict,
                    mass_opts["boost"],
                )
            )
            boost_dry_mass_fun = mass_function_add(
                [dry_mass_fun, boost_components_mass_fun], parameter_dict
            )
            mass_model["boost_dry"] = boost_dry_mass_fun

            HV_cm_offset = np.array([-0.5 * FWS_parameters["length"], 0, 0])
            rocket_cm_offset = np.array([-1.5 * FWS_parameters["length"], 0, 0])
            boost_comps_cm_offset = np.array([0, 0, 0])
            com_model_boost = CoMModelCombined(
                [com_model_HV, com_model_rocket, com_model_boost_comps],
                [HV_cm_offset, rocket_cm_offset, boost_comps_cm_offset],
            )

            aero_sm_boost, stab_sm_boost = generate_aerodeck_first_stage_fairing(
                vehicle=vehicle["boost"],
                aero_inputs=aero_inputs["boost"],
                aero_outputs=aero_outputs,
                parameters=parameter_dict,
                com_model=com_model_boost,
                plot_aero=plot_aero,
                D_fairing=D_fairing,
            )
            thermal_model["boost"] = zero_heat_flux()

        else:
            mass_model["boost_dry"] = dry_mass_fun
            HV_cm_offset = np.array([0, 0, 0])
            rocket_cm_offset = np.array([-FWS_parameters["length"], 0, 0])
            com_model_boost = CoMModelCombined(
                [com_model_HV, com_model_rocket],
                [HV_cm_offset, rocket_cm_offset],
            )
            D_fairing = fairing_diameter(vehicle["HV"], parameter_dict)

            aero_sm_boost, stab_sm_boost = generate_aerodeck_first_stage(
                vehicle=vehicle,
                aero_inputs=aero_inputs["boost"],
                aero_outputs=aero_outputs,
                parameters=parameter_dict,
                com_model=com_model_boost,
                plot_aero=plot_aero,
                D_fairing=D_fairing,
            )

            thermal_model["boost"] = nose_heat_flux(parameter_dict)

        com_model["boost"] = com_model_boost
        stab_model["boost"] = stab_sm_boost
        aero_model["boost"] = aero_sm_boost

    # with open("/home/jack_williamson/spaceplanes-2026/first_stage_aero/first_stage_aero.pkl", "rb") as handle:
    #     first_stage_aero = pickle.load(handle)
    #
    # plot_first_stage_aero(first_stage_aero, parameter_dict, aero_inputs)

    logger.info("Updating models...done")

    model_dict = {
        "aero": aero_model,
        "mass": mass_model,
        "thrust": thrust_model,
        "com": com_model,
        "stab": stab_model,
        "thermal": thermal_model,
    }
    with open("models.pkl", "wb") as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_dict


def vehicle_constraints(
    parameter_dict: dict[str, float],
    vehicle: list[Vehicle],
    model_dict: dict[str, Callable],
    aero_inputs: dict[str, np.ndarray],
    mass_opts: dict[str, float],
    HV_engine=None,
    boost=None,
    HV_type="HGV",
) -> dict[str, VehicleProp]:
    """ HV Vehicle """
    stab_sm_HV = model_dict["stab"]["HV"]
    mass_HV_function = model_dict["mass"]["HV"]
    com_model_HV = model_dict["com"]["HV"]

    # Check stability at 0 AoA and the sampled Mach numbers
    # Stability margin (dCM/dAoA / dCL/dAoA) should be negative
    stab_points_HV = np.column_stack(
        [np.zeros_like(aero_inputs["HV"]["mach"]), aero_inputs["HV"]["mach"]]
    )
    stab_margin_HV = stab_sm_HV.eval(stab_points_HV)
    idx_max_stab_2nd = np.argmax(stab_margin_HV)
    stab_max_HV = stab_margin_HV[idx_max_stab_2nd]
    stab_grad_HV = stab_sm_HV.eval_jac(stab_points_HV[idx_max_stab_2nd]).squeeze()
    stab_margin_grad_HV = stab_sm_HV.eval_jac(stab_points_HV).squeeze()

    param_zeros = np.zeros(len(parameter_dict))

    volume = vehicle["HV"].mesh_data.calculate_volume(thickness=mass_opts[HV_type]["t"])
    volume_sens = volume.sens if hasattr(volume, "sens") else param_zeros.copy()
    vol = VehicleProp("vol", float(volume), volume_sens)

    length, width, height, aspect_ratio = vehicle["HV"].mesh_data.calculate_dimensions()

    width_sens = width.sens if hasattr(width, "sens") else param_zeros.copy()
    height_sens = height.sens if hasattr(height, "sens") else param_zeros.copy()
    aspect_ratio_sens = (
        aspect_ratio.sens if hasattr(aspect_ratio, "sens") else param_zeros.copy()
    )

    width = VehicleProp("width", float(width), width_sens)
    height = VehicleProp("height", float(height), height_sens)
    ld_rat = VehicleProp("LD", float(aspect_ratio), aspect_ratio_sens)

    par_values = list(parameter_dict.values())
    mass_HV_val = mass_HV_function(*par_values)
    mass_HV_sens = np.array(
        mass_HV_function.jacobian()(*par_values, mass_HV_val)
    ).ravel()
    mass_HV = VehicleProp("mass_HV", float(mass_HV_val), mass_HV_sens)

    stab_HV = VehicleProp("stab_HV", stab_max_HV, stab_grad_HV)

    # Centre of mass: only x component
    com_HV = VehicleProp("com_HV", com_model_HV.eval(0)[0], com_model_HV.eval_jac()[0])

    veh_props = {
            "vol": vol,
            "LD": ld_rat,
            "stab_HV": stab_HV,
            "mass_HV": mass_HV,
            "width": width,
            "height": height,
            "com_HV": com_HV,
            }

    if HV_engine:
        mass_HV_fuel_function = model_dict["mass"]["HV_fuel"]
        mass_fuel_val = mass_HV_fuel_function(*par_values)
        mass_fuel_sens = np.array(
            mass_HV_fuel_function.jacobian()(*par_values, mass_fuel_val)
        ).ravel()
        mass_HV_fuel = VehicleProp("mass_HV_fuel", float(mass_fuel_val), mass_fuel_sens)
        veh_props["mass_HV_fuel"] = mass_HV_fuel

    if boost:
        stab_sm_boost = model_dict["stab"]["boost"]
        com_model_boost = model_dict["com"]["boost"]
        mass_prop_function = model_dict["mass"]["boost_prop"]
        mass_boost_function = model_dict["mass"]["boost_dry"]

        # Check stability at 0 AoA and the sampled Mach numbers
        """Using HV Mach numbers"""
        # Stability margin (dCM/dAoA / dCL/dAoA) should be negative
        stab_points_boost = np.column_stack(
            [np.zeros_like(aero_inputs["HV"]["mach"]), aero_inputs["HV"]["mach"]]
        )
        stab_margin_boost = stab_sm_boost.eval(stab_points_HV)
        idx_max_stab_boost = np.argmax(stab_margin_boost)
        stab_max_boost = stab_margin_boost[idx_max_stab_boost]
        stab_grad_boost = stab_sm_boost.eval_jac(stab_points_boost[idx_max_stab_boost]).squeeze()
        stab_margin_grad_boost = stab_sm_boost.eval_jac(stab_points_boost).squeeze()

        mass_boost_val = mass_boost_function(*par_values)
        mass_boost_sens = np.array(
            mass_boost_function.jacobian()(*par_values, mass_boost_val)
        ).ravel()
        mass_boost = VehicleProp(
            "mass_boost_dry", float(mass_boost_val), mass_boost_sens
        )

        mass_prop_val = mass_prop_function(*par_values)
        mass_prop_sens = np.array(
            mass_prop_function.jacobian()(*par_values, mass_prop_val)
        ).ravel()
        mass_prop = VehicleProp("mass_boost_prop", float(mass_prop_val), mass_prop_sens)

        stab_boost = VehicleProp("stab_boost", stab_max_boost, stab_grad_boost)

        # Centre of mass: only x component
        com_boost = VehicleProp("com_boost", com_model_boost.eval(0)[0], com_model_boost.eval_jac()[0])

        veh_props["mass_boost_dry"] = mass_boost
        veh_props["mass_boost_prop"] = mass_prop
        veh_props["com_boost"] = com_boost
        veh_props["stab_boost"] = stab_boost

    return veh_props
#
# def plot_first_stage_aero(aero_fun, parameters, aero_inputs):
#     keys = ["L_rocket", "D_rocket", "fin_area"]
#     params_1st = {k: parameters[k] for k in keys}
#
#     Mach_points = np.linspace(aero_inputs["mach"][0], aero_inputs["mach"][-1], 100)
#     # Create CD & CL
#     # cd, cl = aero_fun(Mach_points, parameters["L_rocket"], parameters["D_rocket"], parameters["fin_area"])
#     cd, cl = aero_fun(Mach_points, 4, 1.7, 0.5)
#     cl = np.array(cl)
#     cd = np.array(cd)
#
#     # CL = np.array(CL).reshape(np.shape(M_grid.T)).T
#     # CD = np.array(CD).reshape(np.shape(M_grid.T)).T
#
#     # Plot CL
#     fig1 = plt.figure()
#     ax = fig1.add_subplot()
#     ax.plot(Mach_points, cl, label="Interpolant")
#     # ax.plot_surface(M_grid, np.rad2deg(aoa_grid), CL, label="Input Data")
#     ax.set_xlabel("Mach")
#     ax.set_ylabel("CL")
#     # ax.legend()
#     ax.set_title("CL")
#
#     # Plot CL
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot()
#     ax2.plot(Mach_points, cd, label="Interpolant")
#     # ax2.plot_surface(M_grid, np.rad2deg(aoa_grid), CD, label="Input Data")
#     ax2.set_xlabel("Mach")
#     ax2.set_ylabel("CD")
#     # ax2.legend()
#     ax2.set_title("CD")
#
#     # fig1.savefig(fname + "_CL.png", dpi=900)
#     fig2.savefig("first_stage_CD.png", dpi=900)
