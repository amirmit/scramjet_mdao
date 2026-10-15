import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import casadi as ca

from multistage_manager import (
    HV_engine_database,
    boost_database,
    get_boost,
    get_HV_engine,
    check_fairing,
    check_HV_type,
    MultistageManager,
)
from codesignTools import logger
from solve_traj import solve_trajectory
from geometry_generation import (
    generate_glider,
    generate_first_stage,
    generate_first_stage_fairing,
)
from update_models import update_models, vehicle_constraints
from missions import (
    MaxRangeGroundLaunch,
    SprintGroundLaunch,
    MaxRangeObstructionGroundLaunch,
    TargetGroundLaunch,
    TargetObstructionGroundLaunch,
)
from spaceplane_missions import BoostSpaceplaneFlyback, SpaceplaneFlyback

# plt.ion()
veh_props_HV = ["vol", "LD", "stab_HV", "mass_HV", "width", "height", "com_HV"]
veh_props_sp = ["mass_HV_fuel"]
veh_props_boost = ["mass_boost_dry", "mass_boost_prop", "com_boost", "stab_boost"]

def make_models_and_solve_traj(
    parameter_dict,
    problem,
    solve_traj=True,
    HV_type="HGV",
    boost=None,
    HV_engine=None,
    fairing=False,
):

    check_HV_type(HV_type)
    boost = get_boost(boost)
    HV_engine = get_HV_engine(HV_engine)
    check_fairing(fairing)

    """ 1. Make vehicle """
    logger.info("Generating vehicle")
    if boost:
        if fairing:
            vehicle, D_fairing = generate_first_stage_fairing(
                parameters=parameter_dict, save=True
            )
        else:
            vehicle = generate_first_stage(parameters=parameter_dict, save=True)
            D_fairing = None
    else:
        vehicle = {"HV": generate_glider(parameters=parameter_dict, save=True)}
        D_fairing = None

    """ 2. Physics simulations and surrogate modelling """
    aero_inputs, aero_outputs, mass_opts, HV_fuel_opts = (
        MultistageManager.constant_characteristics(
            parameter_dict,
            HV_engine=HV_engine,
        )
    )

    model_dict = update_models(
        parameter_dict,
        vehicle,
        aero_inputs,
        aero_outputs,
        mass_opts,
        D_fairing=D_fairing,
        HV_type=HV_type,
        boost=boost,
        HV_engine=HV_engine,
        HV_fuel_opts=HV_fuel_opts,
    )
    """ 3. Vehicle constraints """
    prop_dict = vehicle_constraints(
        parameter_dict,
        vehicle,
        model_dict,
        aero_inputs,
        mass_opts,
        HV_engine=HV_engine,
        HV_type=HV_type,
        boost=boost,
    )

    """ 4. Solve optimal trajectory """
    if solve_traj:
        result = solve_trajectory(parameter_dict, it_no=0, problem=problem)

        return result, prop_dict
    else:
        return prop_dict, model_dict


def model_sens_checks(model_dict, parameters, M, aoa):
    # First way of getting the jacobian
    aero_glider = model_dict["aero"]["glider"]
    aero_glider_jac = aero_glider.jacobian()
    Cd, Cl = aero_glider(M, aoa, *parameters.values())
    aero_sens_1 = np.array(aero_glider_jac(M, aoa, *parameters.values(), Cd, Cl))

    # Second way of getting the jacobian
    x = aero_glider.mx_in()
    y1, y2 = aero_glider(*x)
    y = ca.vertcat(y1, y2)
    aero_jac_expr = ca.jacobian(y, ca.vertcat(*x))
    aero_jac_fun = ca.Function("aero_jac_fun", x, [aero_jac_expr])
    aero_sens = aero_jac_fun(M, aoa, *parameters.values()).full()

    # Check sens differences
    print(
        f"Cd_sens diff: {(aero_sens_1[: int(len(aero_sens_1) / 2)] - aero_sens[0, :])}\n"
    )
    print(
        f"Cl_sens diff: {(aero_sens_1[int(len(aero_sens_1) / 2) :] - aero_sens[1, :])}\n"
    )
    print(f"aero_sens_1: {aero_sens_1}\n")
    print(f"aero_sens: {aero_sens}\n")

    return Cd, Cl, aero_sens


def from_codesign(
    problem,
    codesign_name,
    sim_name,
    iter_no=1,
    HV_type="HGV",
    boost=None,
    HV_engine=None,
    fairing=False,
):
    # Co-Design
    dir_name = os.path.join(codesign_name)
    df = pd.read_csv(dir_name + "/sqp_res.csv")
    param_init_df = pd.read_csv(dir_name + "/parameters.csv")

    if iter_no < 0:
        iter_no = df.shape[0] + iter_no
    J = df["J"].iloc[iter_no]

    parameters = {}
    sensitivities = {}
    veh_props = veh_props_HV
    if HV_engine:
        veh_props += veh_props_sp
    if boost:
        veh_props += veh_props_boost

    for i in param_init_df["names"]:
        parameters[i] = df[i].iloc[iter_no]
        sensitivities[f"dJ/d{i}"] = df[f"dJ/d{i}"].iloc[iter_no]
        for props in veh_props:
            sensitivities[f"d{props}/d{i}"] = df[f"d{props}/d{i}"].iloc[iter_no]

    working_dir = sim_name

    base_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)

    sensitivities_fd = {}
    sens_diff = {}
    J_pert = {"Original": J}

    fd_perc = 1e-3
    for i in parameters.keys():
    # for i in ["length", "L_rocket", "D_rocket"]:
        print(f"Perturbing parameter: {i}")

        """ Perturb Parameters """
        parameters_iter = parameters.copy()
        parameters_iter[i] = parameters_iter[i] * (1 + fd_perc)

        result, prop_dict = make_models_and_solve_traj(parameters_iter, problem, solve_traj=True, HV_type=HV_type, boost=boost, HV_engine=HV_engine, fairing=fairing)

        J_pert[f"d{i}"] = result.cost
        sensitivities_fd[f"dJ/d{i}"] = (result.cost - J) / (parameters[i] * fd_perc)
        sens_diff[f"dJ/d{i}"] = (
            sensitivities[f"dJ/d{i}"] - sensitivities_fd[f"dJ/d{i}"]
        ) / sensitivities[f"dJ/d{i}"]
        for j in prop_dict.keys():
            sensitivities_fd[f"d{j}/d{i}"] = (
                prop_dict[j].val - df[j].iloc[iter_no]
            ) / (parameters[i] * fd_perc)
            sens_diff[f"d{j}/d{i}"] = (
                sensitivities[f"d{j}/d{i}"] - sensitivities_fd[f"d{j}/d{i}"]
            ) / sensitivities[f"d{j}/d{i}"]

        plt.close("all")

    sens_pd = pd.DataFrame(sensitivities.items(), columns=["Name", "AD Sensitivity"])
    sens_fd_pd = pd.DataFrame(sensitivities_fd.items(), columns=["Name", "FD Sensitivity"])
    sens_diff_pd = pd.DataFrame(sens_diff.items(), columns=["Name", "Sensitivity % Diff"])
    J_pert_pd = pd.DataFrame(J_pert.items(), columns=["Name", "Perturbed J"])

    sens_pd.to_csv("sens.csv")
    sens_fd_pd.to_csv("sens_fd.csv")
    sens_diff_pd.to_csv("sens_diff.csv")
    J_pert_pd.to_csv("J_pert.csv")

    sens_pd["FD Sensitivity"] = sensitivities_fd["FD Sensitivity"]
    sens_pd["Sensitivity % Diff"] = sens_diff["Sensitivity % Diff"]
    sens_pd.to_csv("sensitivities.csv")

    print("Large differences in:")
    for key in sens_diff.keys():
        if np.abs(sens_diff[key]) > 0.01:
            print(f"{key}: {sens_diff[key]}")

#
# def from_traj(
#     problem,
#     traj_name,
#     sim_name,
# ):
#     # Trajectory
#     dir_name = os.path.join(traj_name)
#     with open(dir_name + "/parameters.json") as json_file:
#         parameters = json.load(json_file)
#
#     working_dir = sim_name
#
#     base_dir = os.getcwd()
#     if not os.path.exists(working_dir):
#         os.mkdir(working_dir)
#     os.chdir(working_dir)
#
#     result, prop_dict_OG = make_models_and_solve_traj(parameters, problem)
#
#     sensitivities = {}
#     for i, key in enumerate(parameters.keys()):
#         sensitivities[f"dJ/d{key}"] = result.sensitivity[i]
#         for j in prop_dict_OG.keys():
#             sensitivities[f"d{j}/d{key}"] = prop_dict_OG[j].sens[i]
#
#     sensitivities_fd = {}
#     sens_diff = {}
#     J = result.cost
#     J_pert = {"Original": J}
#
#     fd_perc = 1e-3
#     # for i in parameters.keys():
#     # for i in ["length", "OW3", "D2", "IW2", "wspan5", "L_rocket"]:
#     for i in ["OW4", "OW5", "D1", "H1"]:
#         print(f"Perturbing parameter: {i}")
#
#         """ Perturb Parameters """
#         parameters_iter = parameters.copy()
#         parameters_iter[i] = parameters_iter[i] * (1 + fd_perc)
#
#         result, prop_dict = make_models_and_solve_traj(parameters_iter, problem)
#
#         J_pert[f"d{i}"] = result.cost
#         sensitivities_fd[f"dJ/d{i}"] = (result.cost - J) / (parameters[i] * fd_perc)
#         sens_diff[f"dJ/d{i}"] = (
#             sensitivities[f"dJ/d{i}"] - sensitivities_fd[f"dJ/d{i}"]
#         ) / sensitivities[f"dJ/d{i}"]
#
#         for j in prop_dict.keys():
#             sensitivities_fd[f"d{j}/d{i}"] = (
#                 prop_dict[j].val - prop_dict_OG[j].val
#             ) / (parameters[i] * fd_perc)
#             sens_diff[f"d{j}/d{i}"] = (
#                 sensitivities[f"d{j}/d{i}"] - sensitivities_fd[f"d{j}/d{i}"]
#             ) / sensitivities[f"d{j}/d{i}"]
#
#     sens_pd = pd.DataFrame(sensitivities.items(), columns=["name", "value"])
#     sens_fd_pd = pd.DataFrame(sensitivities_fd.items(), columns=["name", "value"])
#     sens_diff_pd = pd.DataFrame(sens_diff.items(), columns=["name", "value"])
#     J_pert_pd = pd.DataFrame(J_pert.items(), columns=["name", "value"])
#
#     sens_pd.to_csv("sens.csv")
#     sens_fd_pd.to_csv("sens_fd.csv")
#     sens_diff_pd.to_csv("sens_diff.csv")
#     J_pert_pd.to_csv("J_pert.csv")
#
#     print("Large differences in:")
#     for key in sens_diff.keys():
#         if np.abs(sens_diff[key]) > 0.01:
#             print(f"{key}: {sens_diff[key]}")
#

def vehicle_prop_sens(
    problem,
    traj_name,
    sim_name,
    HV_type="HGV",
    boost=None,
    HV_engine=None,
    fairing=False,
):
    M = 10
    aoa = 0

    # Trajectory
    dir_name = os.path.join(traj_name)
    with open(dir_name + "/parameters.json") as json_file:
        parameters = json.load(json_file)

    working_dir = sim_name

    base_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)

    prop_dict_OG, model_dict_OG = make_models_and_solve_traj(
        parameters, problem, solve_traj=False
    )
    CD, CL, aero_sens = model_sens_checks(model_dict_OG, parameters, M, aoa)

    sensitivities = {}
    for i, key in enumerate(parameters.keys()):
        sensitivities[f"dCD/d{key}"] = aero_sens[0, i + 2]
        sensitivities[f"dCL/d{key}"] = aero_sens[1, i + 2]
        for j in prop_dict_OG.keys():
            sensitivities[f"d{j}/d{key}"] = prop_dict_OG[j].sens[i]

    sensitivities_fd = {}
    sens_diff = {}

    fd_perc = 1e-2
    # for i in parameters.keys():
    # for i in ["length", "OW3", "D2", "IW2", "wspan5", "L_rocket"]:
    for i in ["length"]:
        print(f"Perturbing parameter: {i}")

        """ Perturb Parameters """
        parameters_iter = parameters.copy()
        parameters_iter[i] = parameters_iter[i] * (1 + fd_perc)

        prop_dict, model_dict = make_models_and_solve_traj(
            parameters_iter, problem, solve_traj=False
        )
        CD_pert, CL_pert = model_dict["aero"]["glider"](
            M, aoa, *parameters_iter.values()
        )

        sensitivities_fd[f"dCD/d{i}"] = (CD_pert - CD) / (parameters[i] * fd_perc)
        sensitivities_fd[f"dCL/d{i}"] = (CL_pert - CL) / (parameters[i] * fd_perc)
        sens_diff[f"dCD/d{i}"] = (
            sensitivities[f"dCD/d{i}"] - sensitivities_fd[f"dCD/d{i}"]
        ) / sensitivities[f"dCD/d{i}"]
        sens_diff[f"dCL/d{i}"] = (
            sensitivities[f"dCL/d{i}"] - sensitivities_fd[f"dCL/d{i}"]
        ) / sensitivities[f"dCL/d{i}"]

        for j in prop_dict.keys():
            sensitivities_fd[f"d{j}/d{i}"] = (
                prop_dict[j].val - prop_dict_OG[j].val
            ) / (parameters[i] * fd_perc)
            sens_diff[f"d{j}/d{i}"] = (
                sensitivities[f"d{j}/d{i}"] - sensitivities_fd[f"d{j}/d{i}"]
            ) / sensitivities[f"d{j}/d{i}"]

    sens_pd = pd.DataFrame(sensitivities.items(), columns=["name", "value"])
    sens_fd_pd = pd.DataFrame(sensitivities_fd.items(), columns=["name", "value"])
    sens_diff_pd = pd.DataFrame(sens_diff.items(), columns=["name", "value"])

    sens_pd.to_csv("sens.csv")
    sens_fd_pd.to_csv("sens_fd.csv")
    sens_diff_pd.to_csv("sens_diff.csv")

    print("Large differences in:")
    for key in sens_diff.keys():
        if np.abs(sens_diff[key]) > 0.01:
            print(f"{key}: {sens_diff[key]}")


if __name__ == "__main__":
    problem = SpaceplaneFlyback()
    iter_no = 61

    """ HV Type """
    # HV_type = "HGV"
    HV_type = "Spaceplane"

    """ Boost """
    boost = None
    # boost = "Falcon1e"
    # boost = "Orion_50S_XL"

    """ HV Engine """
    # HV_engine = None
    HV_engine = "Rutherford"

    """ Fairing """
    fairing = False
    # fairing = True

    codesign_name = "spaceplane_flyback_A"
    traj_name = "Debugging"
    sim_name = f"spaceplane_flyback_A_{iter_no}-sensitivities"

    from_codesign(problem, codesign_name, sim_name, iter_no, HV_type=HV_type, boost=boost, HV_engine=HV_engine, fairing=fairing)
    # from_traj(problem, traj_name, sim_name)
    # vehicle_prop_sens(problem, traj_name, sim_name)
