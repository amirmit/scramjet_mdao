import pandas as pd
import numpy as np

from hypervehicle import Vehicle
from autodiff import SensMode, generate_parameters_with_sensitivity
from hyperpanel import PanelSolver, PanelScheme, logger as panel_logger
from hyperdynamics.aero_interpolate import aero_interpolate_from_df, plot_aero
from hypergrad import SurfaceIntegrator
from smtools import Observations, PolynomialModel, RatioOfGradients, logger as sm_logger
from plot_aero_deck import plot_aero_deck
# from hyperrocket.rocket_aero import rocket_aero_coef

def generate_aerodeck(
    vehicle: Vehicle,
    aero_inputs: dict[str, list[float]],
    aero_outputs,
    parameters: dict[str, float],
    a_ref: float = 1,
    com_model=None,
    plot_aero: bool = True,
) -> Observations:
    """Run a sweep of aoa and mach numbers to generate an aerodeck and
    sensitivity deck.
    """

    if com_model is None:
        cog = np.array([-2 * parameters["length"] / 4, 0, 0])
        cog_sens = None
    else:
        cog = com_model.eval([])
        cog_sens = com_model.eval_jac([])

    # Instantiate flow solver
    flow_solver = PanelSolver(
        solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.OBLIQUE_SHOCK],
        # solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.TANGENT_CONE],
        angle_thresholds=[0],
        fallback_pos=PanelScheme.NORMAL_SHOCK,
    )
    panel_logger.info(f"Created panel solver: {flow_solver}")
    flow_solver.set_geometry(normals=vehicle.mesh_data.normal.astype(float))

    n_parameters = len(parameters)
    parameter_value_list = list(parameters.values())
    parameter_names = list(parameters.keys())
    if n_parameters != len(vehicle.exported_values):
        raise ValueError(
            "Parameter dict lengths should match between vehicle and optimiser!"
        )
    surf_int = SurfaceIntegrator(
        normal=vehicle.mesh_data.normal,
        centroid=vehicle.mesh_data.centroid,
        area=vehicle.mesh_data.area,
        com=cog,
        com_sens=cog_sens,
        nparams=n_parameters,
    )

    panel_logger.info("Sampling points")
    aero_deck = Observations(
        n_inputs=2,
        n_params=n_parameters,
        n_outputs=3,
        input_names=["aoa", "mach"],
        output_names=aero_outputs,
        param_names=parameter_names,
    )

    for i, aoa in enumerate(aero_inputs["aoa"]):
        # print(f"Mach is {mach}")
        panel_logger.info(f"\tAoA is {np.rad2deg(aoa)}deg")
        for j, mach in enumerate(aero_inputs["mach"]):
            # New method
            if mach <= 1:
                forces = [-0.8, 0, 0]
                moments = [0, 0, 0]
                forces_sens = np.zeros((3, n_parameters))
                moments_sens = np.zeros((3, n_parameters))
            else:
                results = flow_solver.compute_flow_state(mach, aoa)
                rot_matrix = flow_solver.get_rotation_matrix(aoa)
                cp_sens = surf_int.calculate_pressure_sensitivity(
                    mach,
                    1.4,
                    results["cp"],
                    results["mach_local"],
                    results["unit_vel_vector"],
                )

                forces, forces_sens, moments, moments_sens = surf_int.integrate_solution(
                    results["cp"], cp_sens, rot_matrix
                )

            coeff_sens_array = np.empty((3, n_parameters), dtype=np.float64)
            coeff_sens_array[0, :] = -forces_sens[0, :]
            coeff_sens_array[1, :] = -forces_sens[2, :]
            coeff_sens_array[2, :] = moments_sens[1, :]

            aero_deck.add_observation(
                x=[aoa, mach],
                Z=[-forces[0], -forces[2], moments[1]],
                p=parameter_value_list,
                dZdp=coeff_sens_array,
            )

    panel_logger.info("Sampling points...done")

    # Save results
    aero_deck.save(f"aero_deck_{vehicle.name}.json")
    aero_deck_df: pd.DataFrame = aero_deck.to_dataframe()
    aero_deck_df.to_csv(f"aero_deck_{vehicle.name}.csv")

    sm_logger.info("Generating polynomial model")
    aero_sm = PolynomialModel(
        degree=5,
        input_names=list(aero_inputs.keys()),
        output_names=aero_outputs,
        parameter_names=parameter_names,
    )
    weights = 1.0 / np.abs(aero_deck_df["CD"].to_numpy())
    weights = weights / np.linalg.norm(weights) * weights.size
    aero_sm.train_model_opt(aero_deck_df, pweights=weights)

    sm_logger.info("Generating stability margin model")
    stab_sm = RatioOfGradients(
        aero_sm,
        gradient_var="aoa",
        Z_numerator="CM",
        Z_denominator="CL",
        sensitivity_names=parameter_names,
    )

    if plot_aero:
        plot_aero_deck(aero_inputs, aero_sm, aero_deck_df)

    sm_logger.info("Generating casadi interpolation function")
    aero_interpolant = aero_interpolate_from_df(
        parameters,
        aero_deck_df,
        aero_inputs["mach"],
        aero_inputs["aoa"],
        include_sens=True,
        fname=f"{vehicle.name}",
    )

    return aero_interpolant, stab_sm

def generate_aerodeck_first_stage(
    vehicle: dict[str, Vehicle],
    aero_inputs: dict[str, list[float]],
    aero_outputs,
    parameters: dict[str, float],
    D_fairing: float,
    a_ref: float = 1,
    com_model=None,
    plot_aero: bool = True,
    fairing=False,
) -> Observations:
    """Run a sweep of aoa and mach numbers to generate an aerodeck and
    sensitivity deck.
    """
    FWS_parameters = generate_parameters_with_sensitivity(parameters, SensMode.JACOBIAN)
    if fairing:
        phi = np.deg2rad(15)
        L_nose = D_fairing / 2 / np.tan(phi)
        L = FWS_parameters["L_rocket"] + 1.5 * FWS_parameters["length"] - L_nose
    else:
        L_nose = FWS_parameters["length"]
        L = FWS_parameters["L_rocket"]

    if com_model is None:
        cog = np.array([-2 * parameters["length"] / 4, 0, 0])
        cog_sens = None
    else:
        cog = com_model.eval([])
        cog_sens = com_model.eval_jac([])

    # Instantiate flow solver
    flow_solver_rocket = PanelSolver(
        solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.OBLIQUE_SHOCK],
        # solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.TANGENT_CONE],
        angle_thresholds=[0],
        fallback_pos=PanelScheme.NORMAL_SHOCK,
    )
    panel_logger.info(f"Created panel solver: {flow_solver_rocket}")
    flow_solver_rocket.set_geometry(normals=vehicle["boost"].mesh_data.normal.astype(float))

    flow_solver_HV = PanelSolver(
        solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.OBLIQUE_SHOCK],
        # solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.TANGENT_CONE],
        angle_thresholds=[0],
        fallback_pos=PanelScheme.NORMAL_SHOCK,
    )
    panel_logger.info(f"Created panel solver: {flow_solver_HV}")
    flow_solver_HV.set_geometry(normals=vehicle["HV"].mesh_data.normal.astype(float))

    n_parameters = len(parameters)
    parameter_value_list = list(parameters.values())
    parameter_names = list(parameters.keys())
    if n_parameters != len(vehicle["HV"].exported_values):
        raise ValueError(
            "Parameter dict lengths should match between vehicle and optimiser!"
        )
    surf_int_rocket = SurfaceIntegrator(
        normal=vehicle["boost"].mesh_data.normal,
        centroid=vehicle["boost"].mesh_data.centroid,
        area=vehicle["boost"].mesh_data.area,
        com=cog,
        com_sens=cog_sens,
        nparams=n_parameters,
    )

    surf_int_HV = SurfaceIntegrator(
        normal=vehicle["HV"].mesh_data.normal,
        centroid=vehicle["HV"].mesh_data.centroid,
        area=vehicle["HV"].mesh_data.area,
        com=cog,
        com_sens=cog_sens,
        nparams=n_parameters,
    )

    panel_logger.info("Sampling points")
    aero_deck = Observations(
        n_inputs=2,
        n_params=n_parameters,
        n_outputs=3,
        input_names=["aoa", "mach"],
        output_names=aero_outputs,
        param_names=parameter_names,
    )

    for i, aoa in enumerate(aero_inputs["aoa"]):
        # print(f"Mach is {mach}")
        panel_logger.info(f"\tAoA is {np.rad2deg(aoa)}deg")
        for j, mach in enumerate(aero_inputs["mach"]):
            # New method
            if mach <= 1:
                CD, CL = rocket_aero_coef(aoa, mach, D_fairing, L_nose, L)
                CD = CD*1.8 #For smoothing to hyperpanel results
                forces = [float(-CD), 0, float(-CL)]
                moments = [0, 0, 0]
                forces_sens = np.array([-CD.sens, np.zeros(n_parameters), -CL.sens])
                moments_sens = np.zeros((3, n_parameters))
            else:
                results_rocket = flow_solver_rocket.compute_flow_state(mach, aoa)
                rot_matrix = flow_solver_rocket.get_rotation_matrix(aoa)
                cp_sens_rocket = surf_int_rocket.calculate_pressure_sensitivity(
                    mach,
                    1.4,
                    results_rocket["cp"],
                    results_rocket["mach_local"],
                    results_rocket["unit_vel_vector"],
                )

                forces_rocket, forces_sens_rocket, moments_rocket, moments_sens_rocket = surf_int_rocket.integrate_solution(
                    results_rocket["cp"], cp_sens_rocket, rot_matrix
                )

                results_HV = flow_solver_HV.compute_flow_state(mach, aoa)
                cp_sens_HGV = surf_int_HV.calculate_pressure_sensitivity(
                    mach,
                    1.4,
                    results_HV["cp"],
                    results_HV["mach_local"],
                    results_HV["unit_vel_vector"],
                )

                forces_HV, forces_sens_HV, moments_HV, moments_sens_HV = surf_int_HV.integrate_solution(
                    results_HV["cp"], cp_sens_HGV, rot_matrix
                )

                forces = forces_rocket + forces_HV
                forces_sens = forces_sens_rocket + forces_sens_HV
                moments = moments_rocket + moments_HV
                moments_sens = moments_sens_rocket + moments_sens_HV

            coeff_sens_array = np.empty((3, n_parameters), dtype=np.float64)
            coeff_sens_array[0, :] = -forces_sens[0, :]
            coeff_sens_array[1, :] = -forces_sens[2, :]
            coeff_sens_array[2, :] = moments_sens[1, :]

            aero_deck.add_observation(
                x=[aoa, mach],
                Z=[-forces[0], -forces[2], moments[1]],
                p=parameter_value_list,
                dZdp=coeff_sens_array,
            )

    panel_logger.info("Sampling points...done")

    # Save results
    aero_deck.save(f"aero_deck_{vehicle['boost'].name}.json")
    aero_deck_df: pd.DataFrame = aero_deck.to_dataframe()
    aero_deck_df.to_csv(f"aero_deck_{vehicle['boost'].name}.csv")

    sm_logger.info("Generating polynomial model")
    aero_sm = PolynomialModel(
        degree=5,
        input_names=list(aero_inputs.keys()),
        output_names=aero_outputs,
        parameter_names=parameter_names,
    )
    weights = 1.0 / np.abs(aero_deck_df["CD"].to_numpy())
    weights = weights / np.linalg.norm(weights) * weights.size
    aero_sm.train_model_opt(aero_deck_df, pweights=weights)

    sm_logger.info("Generating stability margin model")
    stab_sm = RatioOfGradients(
        aero_sm,
        gradient_var="aoa",
        Z_numerator="CM",
        Z_denominator="CL",
        sensitivity_names=parameter_names,
    )

    if plot_aero:
        plot_aero_deck(aero_inputs, aero_sm, aero_deck_df)

    sm_logger.info("Generating casadi interpolation function")
    aero_interpolant = aero_interpolate_from_df(
        parameters,
        aero_deck_df,
        aero_inputs["mach"],
        aero_inputs["aoa"],
        include_sens=True,
        fname=f"{vehicle['boost'].name}",
    )

    return aero_interpolant, stab_sm



def generate_aerodeck_first_stage_fairing(
    vehicle: Vehicle,
    aero_inputs: dict[str, list[float]],
    aero_outputs,
    parameters: dict[str, float],
    D_fairing: float,
    a_ref: float = 1,
    com_model=None,
    plot_aero: bool = True,
    fairing=False,
) -> Observations:
    """Run a sweep of aoa and mach numbers to generate an aerodeck and
    sensitivity deck.
    """
    FWS_parameters = generate_parameters_with_sensitivity(parameters, SensMode.JACOBIAN)
    if fairing:
        phi = np.deg2rad(15)
        L_nose = D_fairing / 2 / np.tan(phi)
        L = FWS_parameters["L_rocket"] + 1.5 * FWS_parameters["length"] - L_nose
    else:
        L_nose = FWS_parameters["length"]
        L = FWS_parameters["L_rocket"]

    if com_model is None:
        cog = np.array([-2 * parameters["length"] / 4, 0, 0])
        cog_sens = None
    else:
        cog = com_model.eval([])
        cog_sens = com_model.eval_jac([])

    # Instantiate flow solver
    flow_solver = PanelSolver(
        solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.OBLIQUE_SHOCK],
        # solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.TANGENT_CONE],
        angle_thresholds=[0],
        fallback_pos=PanelScheme.NORMAL_SHOCK,
    )
    panel_logger.info(f"Created panel solver: {flow_solver}")
    flow_solver.set_geometry(normals=vehicle.mesh_data.normal.astype(float))

    n_parameters = len(parameters)
    parameter_value_list = list(parameters.values())
    parameter_names = list(parameters.keys())
    if n_parameters != len(vehicle.exported_values):
        raise ValueError(
            "Parameter dict lengths should match between vehicle and optimiser!"
        )
    surf_int = SurfaceIntegrator(
        normal=vehicle.mesh_data.normal,
        centroid=vehicle.mesh_data.centroid,
        area=vehicle.mesh_data.area,
        com=cog,
        com_sens=cog_sens,
        nparams=n_parameters,
    )

    panel_logger.info("Sampling points")
    aero_deck = Observations(
        n_inputs=2,
        n_params=n_parameters,
        n_outputs=3,
        input_names=["aoa", "mach"],
        output_names=aero_outputs,
        param_names=parameter_names,
    )

    for i, aoa in enumerate(aero_inputs["aoa"]):
        # print(f"Mach is {mach}")
        panel_logger.info(f"\tAoA is {np.rad2deg(aoa)}deg")
        for j, mach in enumerate(aero_inputs["mach"]):
            # New method
            if mach <= 1:
                CD, CL = rocket_aero_coef(aoa, mach, D_fairing, L_nose, L)
                CD = CD*1.8 #For smoothing to hyperpanel results
                forces = [float(-CD), 0, float(-CL)]
                moments = [0, 0, 0]
                forces_sens = np.array([-CD.sens, np.zeros(n_parameters), -CL.sens])
                moments_sens = np.zeros((3, n_parameters))
            else:
                results = flow_solver.compute_flow_state(mach, aoa)
                rot_matrix = flow_solver.get_rotation_matrix(aoa)
                cp_sens = surf_int.calculate_pressure_sensitivity(
                    mach,
                    1.4,
                    results["cp"],
                    results["mach_local"],
                    results["unit_vel_vector"],
                )

                forces, forces_sens, moments, moments_sens = surf_int.integrate_solution(
                    results["cp"], cp_sens, rot_matrix
                )

            coeff_sens_array = np.empty((3, n_parameters), dtype=np.float64)
            coeff_sens_array[0, :] = -forces_sens[0, :]
            coeff_sens_array[1, :] = -forces_sens[2, :]
            coeff_sens_array[2, :] = moments_sens[1, :]

            aero_deck.add_observation(
                x=[aoa, mach],
                Z=[-forces[0], -forces[2], moments[1]],
                p=parameter_value_list,
                dZdp=coeff_sens_array,
            )

    panel_logger.info("Sampling points...done")

    # Save results
    aero_deck.save(f"aero_deck_{vehicle.name}.json")
    aero_deck_df: pd.DataFrame = aero_deck.to_dataframe()
    aero_deck_df.to_csv(f"aero_deck_{vehicle.name}.csv")

    sm_logger.info("Generating polynomial model")
    aero_sm = PolynomialModel(
        degree=5,
        input_names=list(aero_inputs.keys()),
        output_names=aero_outputs,
        parameter_names=parameter_names,
    )
    weights = 1.0 / np.abs(aero_deck_df["CD"].to_numpy())
    weights = weights / np.linalg.norm(weights) * weights.size
    aero_sm.train_model_opt(aero_deck_df, pweights=weights)

    sm_logger.info("Generating stability margin model")
    stab_sm = RatioOfGradients(
        aero_sm,
        gradient_var="aoa",
        Z_numerator="CM",
        Z_denominator="CL",
        sensitivity_names=parameter_names,
    )

    if plot_aero:
        plot_aero_deck(aero_inputs, aero_sm, aero_deck_df)

    sm_logger.info("Generating casadi interpolation function")
    aero_interpolant = aero_interpolate_from_df(
        parameters,
        aero_deck_df,
        aero_inputs["mach"],
        aero_inputs["aoa"],
        include_sens=True,
        fname=f"{vehicle.name}",
    )

    return aero_interpolant, stab_sm


