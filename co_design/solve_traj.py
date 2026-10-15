import pickle
import numpy as np
import matplotlib.pyplot as plt

from codesignTools.casadi_problem import CasadiProblem
from dynamics import trim_3DOF_boost_glide, trim_3DOF_boost_spaceplane_flyback, trim_3DOF_glide, trim_3DOF_spaceplane_flyback
from initialisation import initialisation, boost_spaceplane_flyback_w0, initialisation_boost_only, initialisation_glider, spaceplane_flyback_w0
from hyperdynamics.utils import Result, sens2json
from hyperdynamics.transcription import LgrCollocation
from hyperdynamics.run_nlp import trajectory_optimisation
from plot_traj import plot_from_result, result2csv


def solve_trajectory(
    parameters: dict[str, float],
    it_no: int,
    problem: CasadiProblem,
    model_dict=None,
    w0=None,
    lam_x0=None,
    lam_g0=None,
) -> Result:
    if model_dict is None:
        with open("models.pkl", "rb") as handle:
            model_dict = pickle.load(handle)

    aero_model = model_dict["aero"]
    mass_model = model_dict["mass"]
    thrust_model = model_dict["thrust"]
    thermal_model = model_dict["thermal"]

    if problem.name == "boost_spaceplane_flyback":
        system, casadi_params = trim_3DOF_boost_spaceplane_flyback(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
        )
        mission = problem(system)
    elif problem.name == "spaceplane_flyback":
        system, casadi_params = trim_3DOF_spaceplane_flyback(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
        )
        mission = problem(system)
    elif problem.name == "target_2_glider":
        system, casadi_params = trim_3DOF_glide(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
            thermal_model,
            stages=3,
        )
        mission = problem(system)
        nk_schedule = [[20]*(mission.stages), [*[20]*(mission.stages-1), 30]]  # Glide
    elif problem.name[-6:] == "glider":
        system, casadi_params = trim_3DOF_glide(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
            thermal_model,
        )
        mission = problem(system)
        nk_schedule = [[20]*(mission.stages), [*[20]*(mission.stages-1), 30]]  # Glide
    elif problem.name == "target_2_ground_launch":
        system, casadi_params = trim_3DOF_boost_glide(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
            thermal_model,
            boost_stages=2,
            HGV_stages=3,
        )
        mission = problem(system)
        nk_schedule = [[1, *[20]*(mission.stages-1)], [1, *[20]*(mission.stages-2), 30]]  # Boost-Glide
    elif problem.name == "first_stage_energy_ground_launch":
        system, casadi_params = trim_3DOF_boost_glide(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
            thermal_model,
            boost_stages=2,
            HGV_stages=0,
        )
        mission = problem(system)
        nk_schedule = [[1, *[20]*(mission.stages-1)], [1, *[20]*(mission.stages-1)]]  # Boost
    else:
        system, casadi_params = trim_3DOF_boost_glide(
            parameters,
            aero_model,
            mass_model,
            thrust_model,
            thermal_model,
        )
        mission = problem(system)
        nk_schedule = [[1, *[20]*(mission.stages-1)], [1, *[20]*(mission.stages-2), 30]]  # Boost-Glide

    results = []
    for i, n_k in enumerate(nk_schedule):
        if i == 0:
            transcription = LgrCollocation(n_k=n_k, d=2, du=2, stages=mission.stages)
            if problem.name == "boost_spaceplane_flyback":
                w0 = boost_spaceplane_flyback_w0(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_stages=[50, 150],
                    t_init=200,
                )
            elif problem.name == "spaceplane_flyback":
                w0 = spaceplane_flyback_w0(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_stages=[100],
                    t_init=200,
                )
            elif problem.name == "target_2_glider":
                w0 = initialisation_glider(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_init=200,
                    stages=3,
                )
            elif problem.name[-6:] == "glider":
                w0 = initialisation_glider(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_init=200,
                )
            elif problem.name == "target_2_ground_launch":
                w0 = initialisation(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_stages=[5, 40, 80, 100, 120],
                    t_init=150,
                    boost_stages=2,
                    HGV_stages=3,
                )
            elif problem.name == "first_stage_energy_ground_launch":
                w0 = initialisation_boost_only(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_stages=[5],
                    t_init=40,
                    boost_stages=2,
                )
            else:
                w0 = initialisation(
                    transcription=transcription,
                    mission=mission,
                    system=system,
                    t_stages=[5, 40],
                    t_init=150,
                    boost_stages=2,
                )
        else:
            transcription.n_k = n_k
            transcription.d = 3
            transcription.du = 3
            w0 = transcription.pack_w(
                x=results[-1].plotting_attr["x_int"],
                u=results[-1].plotting_attr["u_int"],
                t=results[-1].plotting_attr["t_int"],
                mission=mission,
            )

        current_result = trajectory_optimisation(
            system=system,
            mission=mission,
            transcription=transcription,
            w0=w0,
        )
        results.append(current_result)

        # Save result to csv
        result2csv(
            system=system,
            result=current_result,
            fname=f"trajectory-{i}.csv",
        )
        plot_from_result(
            results[-1],
            system,
            title="Glide Vehicle",
            fname=f"trajectory-{i}.png",
            save=True,
            show=False,
            fname_xy=f"x-y_Position-{i}.png",
            fname_z=f"altitude_vs_{i}",
            f_add_name=f"{i}",
        )
        sens2json(parameters, results[-1], fname=f"sensitivities-{i}.json")

    j = 0

    while results[-1].success == False or abs((results[-1].cost - results[-2].cost)/results[-1].cost) >0.001:
        i += 1
        w0 = transcription.pack_w(
                x=results[-1].plotting_attr["x_int"],
                u=results[-1].plotting_attr["u_int"],
                t=results[-1].plotting_attr["t_int"],
                mission=mission,
            )

        current_result = trajectory_optimisation(
            system=system,
            mission=mission,
            transcription=transcription,
            w0=w0,
        )
        results.append(current_result)

        # Save result to csv
        result2csv(
            system=system,
            result=current_result,
            fname=f"trajectory-{i}.csv",
        )
        plot_from_result(
            results[-1],
            system,
            title="Glide Vehicle",
            fname=f"trajectory-{i}.png",
            save=True,
            show=False,
            fname_xy=f"x-y_Position-{i}.png",
            fname_z=f"altitude_vs_{i}",
            f_add_name=f"{i}",
        )
        sens2json(parameters, results[-1], fname=f"sensitivities-{i}.json")

        j += 1
        if j > 5:
            break

    # Save result to csv
    result2csv(
        system=system,
        result=results[-1],
        fname=f"trajectory-final.csv",
    )
    plot_from_result(
        results[-1],
        system,
        title="Glide Vehicle",
        fname=f"trajectory-final.png",
        save=True,
        show=False,
        # fname_xy=f"x-y_Position-final.png",
        # fname_z=f"altitude_vs_final",
        f_add_name="final"
    )
    results[-1].sensitivity[-1] = results[-1].sensitivity[-1] / 10
    sens2json(parameters, results[-1], fname=f"sensitivities-final.json")

    plt.close("all")

    return results[-1]
