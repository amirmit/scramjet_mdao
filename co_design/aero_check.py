from casadi import mac
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from hypervehicle import Vehicle
from autodiff import SensMode, generate_parameters_with_sensitivity
from hyperpanel import PanelSolver, PanelScheme, logger as panel_logger
from hyperdynamics.aero_interpolate import aero_interpolate_from_df, plot_aero
from hypergrad import SurfaceIntegrator
from smtools import Observations, PolynomialModel, RatioOfGradients, logger as sm_logger
from multistage_manager import MultistageManager
from plot_aero_deck import plot_aero_deck
from hyperrocket.rocket_aero import rocket_aero_coef
from hyperrocket.rocket import Falcon1e, Orion_50S_XL, Rutherford
from geometry_generation import generate_first_stage, generate_first_stage_fairing, fairing_diameter


def generate_aerodeck_first_stage_fairing(
    vehicle: Vehicle,
    aero_inputs: dict[str, list[float]],
    aero_outputs,
    parameters: dict[str, float],
    D_fairing: float,
    a_ref: float = 1,
    com_model=None,
    plot_aero: bool = True,
    scheme: PanelScheme = PanelScheme.OBLIQUE_SHOCK,
    mach_cutoff=1,
    label="HGV",
    scale=1,
) -> Observations:
    """Run a sweep of aoa and mach numbers to generate an aerodeck and
    sensitivity deck.
    """
    FWS_parameters = generate_parameters_with_sensitivity(parameters, SensMode.JACOBIAN)
    phi = np.deg2rad(15)
    L_nose = D_fairing / 2 / np.tan(phi)
    L = FWS_parameters["L_rocket"] + 1.5 * FWS_parameters["length"] - L_nose

    if com_model is None:
        cog = np.array([-2 * parameters["length"] / 4, 0, 0])
        cog_sens = None
    else:
        cog = com_model.eval([])
        cog_sens = com_model.eval_jac([])

    # Instantiate flow solver
    flow_solver = PanelSolver(
        solver_methods=[PanelScheme.PRANDTL_MEYER_INTERP, scheme],
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
            if mach <= mach_cutoff:
                CD, CL = rocket_aero_coef(aoa, mach, D_fairing, L_nose, L)
                CD = CD * scale  # For smoothing to hyperpanel results
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

                forces, forces_sens, moments, moments_sens = (
                    surf_int.integrate_solution(results["cp"], cp_sens, rot_matrix)
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
    aero_deck_df.to_csv(f"aero_deck_{label}.csv")

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

    return aero_interpolant, stab_sm, aero_deck_df


def main(codesign_name, mach_cutoffs, schemes, working_dir, labels, scales, iter_no=0):
    base_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)

    dir_name = os.path.join(base_dir, codesign_name)
    df = pd.read_csv(dir_name + f"sqp_res.csv")
    param_init_df = pd.read_csv(dir_name + "/parameters.csv")

    if iter_no < 0:
        iter_no = df.shape[0] + iter_no

    parameter_dict = {}
    for i in param_init_df.names:
        parameter_dict[i] = df[i].iloc[iter_no]

    vehicle = generate_first_stage(
        parameters=parameter_dict, save=False
    )
    aero_inputs, aero_outputs, mass_opts, HV_fuel_opts = (
        MultistageManager.constant_characteristics(
            parameter_dict,
            HV_engine=None,
        )
    )

    D_fairing = fairing_diameter(vehicle["HV"], parameter_dict)

    boost_aoa = np.array([0])
    boost_mach = np.linspace(0.1, 15, 50)
    boost_aoa_grid, boost_mach_grid = np.meshgrid(boost_aoa, boost_mach)
    boost_aoa_sample = aero_inputs["boost"]["aoa"]
    boost_mach_sample = aero_inputs["boost"]["mach"]
    mach_grid_sample, aoa_grid_sample = np.meshgrid(boost_mach_sample, boost_aoa_sample)

    # CD = np.empty((len(mach_cutoffs)+1, boost_mach.size))
    # CL = np.empty((len(mach_cutoffs)+1, boost_mach.size))
    CD = boost_mach
    CL = boost_mach
    aero_sample_df = pd.DataFrame(
        np.vstack((aoa_grid_sample.flatten(), mach_grid_sample.flatten())).T,
        columns=["aoa", "mach"],
    )
    fig_CD, ax_CD = plt.subplots()
    fig_CL, ax_CL = plt.subplots()

    for mach_cutoff, scheme, label, scale in zip(mach_cutoffs, schemes, labels, scales):
        aero_sm_boost, stab_sm_boost, aero_deck_df = (
            generate_aerodeck_first_stage_fairing(
                vehicle=vehicle["boost"],
                aero_inputs=aero_inputs["boost"],
                aero_outputs=aero_outputs,
                parameters=parameter_dict,
                plot_aero=True,
                D_fairing=D_fairing,
                mach_cutoff=mach_cutoff,
                scheme=scheme,
                label=label,
                scale=scale,
            )
        )

        CD_b, CL_b = aero_sm_boost(
            boost_mach_grid, boost_aoa_grid, *parameter_dict.values()
        )
        # CD.append(CD_b.full())
        # CL.append(CL_b.full())
        CD = np.vstack((CD, CD_b.full().T))
        CL = np.vstack((CL, CL_b.full().T))
        aero_sample_df[f"{label}_CD"] = aero_deck_df["CD"]
        aero_sample_0_aoa = aero_sample_df[aero_sample_df["aoa"] == 0]

        (line1,) = ax_CD.plot(boost_mach, CD_b, label=label)
        ax_CD.plot(
            aero_sample_0_aoa["mach"],
            aero_sample_0_aoa[f"{label}_CD"],
            "o",
            color=line1.get_color(),
        )
        ax_CL.plot(boost_mach, CL_b, label=label)

    labels = ["Mach"] + labels
    CD_df = pd.DataFrame(CD.T, index=boost_mach, columns=labels)
    CD_df.to_csv(f"Surrogate_CD.csv")
    aero_sample_df.to_csv("sampled_aero.csv")

    ax_CD.set_xlabel("Mach")
    ax_CD.set_ylabel("CD")
    ax_CD.set_title("CD")
    ax_CD.legend()
    ax_CL.set_xlabel("Mach")
    ax_CL.set_ylabel("CL")
    ax_CL.set_title("CL")
    ax_CL.legend()
    plt.show()


if __name__ == "__main__":
    codesign_name = "max_range_obstruction_ground_launch_C/"
    mach_cutoffs = [30, 1, 1, 1, 3, 3, 3]
    schemes = [
        PanelScheme.OBLIQUE_SHOCK,
        PanelScheme.OBLIQUE_SHOCK,
        PanelScheme.TANGENT_CONE,
        PanelScheme.MODIFIED_NEWTONIAN,
        PanelScheme.OBLIQUE_SHOCK,
        PanelScheme.OBLIQUE_SHOCK,
        PanelScheme.OBLIQUE_SHOCK,
    ]
    labels = [
        "Component Buildup",
        "Oblique Shock",
        "Tangent Cone",
        "Modified Newtonian",
        "Oblique Shock, scale=1.8, M_cutoff=3",
        "Oblique Shock, scale=1.5, M_cutoff=3",
        "Oblique Shock, scale=2.5, M_cutoff=3",
    ]
    scales = [1, 1, 1, 1, 1.8, 1.5, 2.5]
    working_dir = "Boost_Aero_Comparison"

    main(
        codesign_name=codesign_name,
        mach_cutoffs=mach_cutoffs,
        schemes=schemes,
        labels=labels,
        working_dir=working_dir,
        scales=scales,
    )
