import os
from codesignTools import logger
from solve_traj import solve_trajectory
from geometry_generation import (
    generate_glider,
    generate_first_stage,
    generate_first_stage_fairing,
)
from update_models import update_models, vehicle_constraints
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import shutil

from missions import (
    FirstStageEnergyGroundLaunch,
    GroundLaunchw0,
    MaxRangeGroundLaunch,
    MaxRangeObstructionGroundLaunch,
    TargetGroundLaunch,
    TargetObstructionGroundLaunch,
    SprintGroundLaunch,
    Target2GroundLaunch,
    Sprint2GroundLaunch,
)
from spaceplane_missions import (
    BoostSpaceplane,
    BoostSpaceplaneFlyback,
    SpaceplaneFlyback,
)
from glide_missions import (
    TargetObstructionGlider,
    MaxRangeObstructionGlider,
    SprintGlider,
    Sprint2Glider,
    Target2Glider,
)
from multistage_manager import (
    HV_engine_database,
    boost_database,
    get_boost,
    get_HV_engine,
    check_fairing,
    check_HV_type,
    MultistageManager,
)

boost_param_names = ["L_rocket", "D_rocket", "fin_area"]


def ensure_empty_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# plt.ion()
def traj_from_params(
    problem,
    codesign_name,
    sim_name,
    iter_no=1,
    HV_type="HGV",
    boost=None,
    HV_engine=None,
    fairing=False,
    boost_params_codesign=None,
    boost_iter_no=-1,
):
    working_dir = sim_name
    base_dir = os.getcwd()
    ensure_empty_dir(working_dir)
    os.chdir(working_dir)

    dir_name = os.path.join(codesign_name)
    df = pd.read_csv(dir_name + f"sqp_res.csv")
    param_init_df = pd.read_csv(dir_name + "/parameters.csv")

    if boost_params_codesign:
        boost_dir_name = os.path.join(boost_params_codesign)
        boost_df = pd.read_csv(boost_dir_name + f"sqp_res.csv")
        boost_param_init_df = pd.read_csv(boost_dir_name + "/parameters.csv")
        if boost_iter_no < 0:
            boost_iter_no = boost_df.shape[0] + boost_iter_no

    if iter_no < 0:
        iter_no = df.shape[0] + iter_no

    parameter_dict = {}
    for i in param_init_df.names:
        parameter_dict[i] = df[i].iloc[iter_no]

    if boost_params_codesign:
        for i in boost_param_names:
            parameter_dict[i] = boost_df[i].iloc[boost_iter_no]
            param_init_df.loc[param_init_df["names"] == i, ["llim", "ulim"]] = (
                boost_param_init_df.loc[
                    boost_param_init_df["names"] == i, ["llim", "ulim"]
                ]
            )

    param_init_df = param_init_df.rename(columns={"init": "param_values"})
    param_init_df.param_values = parameter_dict.values()
    param_init_df.to_csv("parameters.csv")

    check_HV_type(HV_type)
    boost = get_boost(boost)
    HV_engine = get_HV_engine(HV_engine)
    check_fairing(fairing)

    if "Vehicle-" in sim_name:
        vehicle_name = sim_name.split("Vehicle-", 1)[1]
    else:
        vehicle_name = None

    """ 1. Make vehicle """
    logger.info("Generating vehicle")
    if boost:
        if fairing:
            vehicle, D_fairing = generate_first_stage_fairing(
                parameters=parameter_dict,
                save=True,
                name=vehicle_name,
            )
        else:
            vehicle = generate_first_stage(
                parameters=parameter_dict, save=True, name=vehicle_name
            )
            D_fairing = None
    else:
        vehicle =generate_glider(parameters=parameter_dict, save=True, name=vehicle_name)
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

    # # Save parameters
    # json_object = json.dumps(parameter_dict)
    # # Writing to sample.json
    # with open("parameters.json", "w") as outfile:
    #     outfile.write(json_object)
    # df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
    # df.to_csv(fname)

    # solve trajectory using a schedule for number of collocation points
    solve_trajectory(
        parameters=parameter_dict,
        it_no=iter_no,
        problem=problem,
        model_dict=model_dict,
    )


def traj_from_models(
    problem,
    codesign_name,
    sim_name,
    iter_no=1,
):
    dir_name = os.path.join(codesign_name)
    df = pd.read_csv(dir_name + f"sqp_res.csv")
    param_init_df = pd.read_csv(dir_name + "/parameters.csv")

    if iter_no < 0:
        iter_no = df.shape[0] + iter_no

    parameters = {}
    for i in param_init_df.names:
        parameters[i] = df[i].iloc[iter_no]

    model_path = os.path.join(dir_name, f"iter_{iter_no}/")
    working_dir = sim_name

    base_dir = os.getcwd()
    ensure_empty_dir(working_dir)
    os.chdir(working_dir)

    # Serializing json
    json_object = json.dumps(parameters)
    # Writing to sample.json
    with open("parameters.json", "w") as outfile:
        outfile.write(json_object)

    # Load model dict
    with open(model_path + "models.pkl", "rb") as handle:
        model_dict = pickle.load(handle)

    # solve trajectory using a schedule for number of collocation points
    solve_trajectory(
        parameters=parameters,
        it_no=iter_no,
        problem=problem,
        model_dict=model_dict,
    )


if __name__ == "__main__":
    problem = Target2GroundLaunch()
    # problem = Sprint2GroundLaunch()
    iter_no = 0

    """ HV Type """
    HV_type = "HGV"
    # HV_type = "Spaceplane"

    """ Boost """
    # boost = None
    boost = "Falcon1e"
    # boost = "Orion_50S_XL"

    """ HV Engine """
    HV_engine = None
    # HV_engine = "Rutherford"

    """ Fairing """
    fairing = False
    # fairing = True

    # codesign_name = "/home/jack_williamson/spaceplanes-2026/first_stage_energy_ground_launch_B/"
    codesign_name = (
        "/home/jack_williamson/spaceplanes-2026/target_2_obstruction_ground_launch_K/"
    )
    sim_name = "Mission-Target 2 Boost-Glide_Vehicle-Target 2 Boost-Glide_K Iter_0"
    # sim_name = "Debugging"
    # boost_codesign_name = (
    #     "/home/jack_williamson/spaceplanes-2026/target_2_obstruction_ground_launch_L/"
    # )
    # boost_codesign_name = (
    #     "/home/jack_williamson/spaceplanes-2026/first_stage_energy_ground_launch_D/"
    # )

    # traj_from_models(problem, codesign_name, sim_name, iter_no)
    traj_from_params(problem, codesign_name, sim_name, iter_no, HV_type=HV_type, boost=boost, HV_engine=HV_engine, fairing=fairing)
    # traj_from_params(
    #     problem,
    #     codesign_name,
    #     sim_name,
    #     iter_no,
    #     HV_type=HV_type,
    #     boost=boost,
    #     HV_engine=HV_engine,
    #     fairing=fairing,
    #     boost_params_codesign=boost_codesign_name,
    # )
    input("DONE")
