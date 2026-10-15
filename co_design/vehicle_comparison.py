import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from plotting import plotting

vehicle_mass = [849, 758, 851, 796]


def param_comparison(sim_names: list[str], dir: str):
    working_dir = dir
    base_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    # os.chdir(working_dir)

    theta = []
    theta_norm = []
    theta_llim = []
    theta_ulim = []
    for i, sim in enumerate(sim_names):
        theta_df = pd.read_csv(f"{sim}/parameters.csv")
        theta.append(theta_df.param_values)
        theta_llim.append(theta_df.llim)
        theta_ulim.append(theta_df.ulim)
        theta_norm.append(normalise_theta(theta[i], theta_llim[i], theta_ulim[i]))

    diff_array = np.empty((len(theta[0]) + len(sim_names), len(sim_names)))
    for i, norm_A in enumerate(theta_norm):
        for j, norm_B in enumerate(theta_norm):
            diff_array[j, i] = np.linalg.norm(norm_A - norm_B) / np.sqrt(len(theta[0]))

        diff_array[j + 1 :, i] = norm_A

    col_labels = sim_names
    row_labels = [*sim_names, *theta_df.names]
    diff_df = pd.DataFrame(diff_array, index=row_labels, columns=col_labels)
    diff_df.to_csv(f"{dir}/parameter_comparison.csv")

    check_limits(theta_llim, theta_ulim)

    return theta


def check_limits(theta_llim: list[np.ndarray], theta_ulim: list[np.ndarray]):
    for i in range(len(theta_llim) - 1):
        if theta_llim[i].all() != theta_llim[i + 1].all():
            print("CAREFUL!! Lower limits of parameters sets are not equal!")
        if theta_ulim[i].all() != theta_ulim[i + 1].all():
            print("CAREFUL!! Upper limits of parameters sets are not equal!")


def normalise_theta(theta, llim, ulim):
    theta_norm = np.empty_like(theta)
    for i, theta_i in enumerate(theta):
        if ulim[i] == llim[i]:
            theta_norm[i] = 1
        else:
            theta_norm[i] = (theta_i - llim[i]) / (ulim[i] - llim[i])

    return theta_norm


def compare_aero(sim_names: list[str], theta: list[np.ndarray], dir: str):
    boost = True

    HV_aoa = np.linspace(np.deg2rad(-5), np.deg2rad(10), 40)
    HV_mach = np.linspace(1.5, 15, 50)
    boost_aoa = np.linspace(np.deg2rad(-10), np.deg2rad(10), 50)
    boost_mach = np.linspace(0.1, 15, 50)
    HV_aoa_grid, HV_mach_grid = np.meshgrid(HV_aoa, HV_mach)
    boost_aoa_grid, boost_mach_grid = np.meshgrid(boost_aoa, boost_mach)

    HV_CD = []
    HV_CL = []
    boost_CD = []
    boost_CL = []
    LD = []
    HV_LD = [np.rad2deg(HV_aoa_grid.flatten()), HV_mach_grid.flatten()]
    boost_LD = [np.rad2deg(boost_aoa_grid.flatten()), boost_mach_grid.flatten()]
    HV_LD_max = []
    HV_LD_min = []
    boost_LD_max = []
    boost_LD_min = []
    avg = []
    fig_HV_CL = plt.figure()
    ax_HV_CL = fig_HV_CL.add_subplot(projection="3d")
    fig_HV_CD = plt.figure()
    ax_HV_CD = fig_HV_CD.add_subplot(projection="3d")
    fig_boost_CL = plt.figure()
    ax_boost_CL = fig_boost_CL.add_subplot(projection="3d")
    fig_boost_CD = plt.figure()
    ax_boost_CD = fig_boost_CD.add_subplot(projection="3d")
    fig_HV_LD = plt.figure()
    ax_HV_LD = fig_HV_LD.add_subplot(projection="3d")

    for i, sim in enumerate(sim_names):
        # Load model dict
        with open(f"{sim}/models.pkl", "rb") as handle:
            model_dict = pickle.load(handle)

        traj_name = f"{sim}/trajectory-final.csv"
        traj_data = pd.read_csv(traj_name)
        stages = int(traj_data.columns.values[-1][-1])

        CD, CL = model_dict["aero"]["HV"](HV_mach_grid, HV_aoa_grid, *theta[i])
        CD = CD.full()
        CL = CL.full()
        HV_CD.append(CD)
        HV_CL.append(CL)

        ax_HV_CL.plot_surface(
            HV_mach_grid, np.rad2deg(HV_aoa_grid), HV_CL[i], label=sim
        )
        ax_HV_CD.plot_surface(
            HV_mach_grid, np.rad2deg(HV_aoa_grid), HV_CD[i], label=sim
        )

        LD_HV = CL / CD
        HV_LD.append(LD_HV.flatten())
        HV_max_LD = np.max(LD_HV)
        HV_max_LD_idx = np.unravel_index(np.argmax(LD_HV), LD_HV.shape)
        HV_min_LD = np.min(LD_HV)
        HV_min_LD_idx = np.unravel_index(np.argmin(LD_HV), LD_HV.shape)
        HV_min_D = np.min(CD)
        HV_min_D_idx = np.unravel_index(np.argmin(CD), CD.shape)
        ax_HV_LD.plot_surface(HV_mach_grid, np.rad2deg(HV_aoa_grid), LD_HV, label=sim)

        if boost:
            CD_b, CL_b = model_dict["aero"]["boost"](
                boost_mach_grid, boost_aoa_grid, *theta[i]
            )
            CD_b = CD_b.full()
            CL_b = CL_b.full()
            boost_CD.append(CD_b)
            boost_CL.append(CL_b)

            ax_boost_CD.plot_surface(
                boost_mach_grid, np.rad2deg(boost_aoa_grid), boost_CD[i], label=sim
            )
            ax_boost_CL.plot_surface(
                boost_mach_grid, np.rad2deg(boost_aoa_grid), boost_CL[i], label=sim
            )

            LD_boost = CL_b / CD_b
            boost_LD.append(LD_boost.flatten())
            boost_max_LD = np.max(LD_boost)
            boost_max_LD_idx = np.unravel_index(np.argmax(LD_boost), LD_boost.shape)
            boost_min_LD = np.min(LD_boost)
            boost_min_LD_idx = np.unravel_index(np.argmin(LD_boost), LD_boost.shape)
            boost_min_D = np.min(CD_b)
            boost_min_D_idx = np.unravel_index(np.argmin(CD_b), CD_b.shape)
            LD.append(
                [
                    HV_max_LD,
                    HV_mach_grid[HV_max_LD_idx],
                    np.rad2deg(HV_aoa_grid[HV_max_LD_idx]),
                    HV_min_LD,
                    HV_mach_grid[HV_min_LD_idx],
                    np.rad2deg(HV_aoa_grid[HV_min_LD_idx]),
                    boost_max_LD,
                    boost_mach_grid[boost_max_LD_idx],
                    np.rad2deg(boost_aoa_grid[boost_max_LD_idx]),
                    boost_min_LD,
                    boost_mach_grid[boost_min_LD_idx],
                    np.rad2deg(boost_aoa_grid[boost_min_LD_idx]),
                    HV_min_D,
                    HV_mach_grid[HV_min_D_idx],
                    np.rad2deg(HV_aoa_grid[HV_min_D_idx]),
                    boost_min_D,
                    boost_mach_grid[boost_min_D_idx],
                    np.rad2deg(boost_aoa_grid[boost_min_D_idx]),
                ]
            )
        else:
            LD.append(
                [
                    HV_max_LD,
                    HV_mach_grid[HV_max_LD_idx],
                    np.rad2deg(HV_aoa_grid[HV_max_LD_idx]),
                    HV_min_LD,
                    HV_mach_grid[HV_min_LD_idx],
                    np.rad2deg(HV_aoa_grid[HV_min_LD_idx]),
                    HV_min_D,
                    HV_mach_grid[HV_min_D_idx],
                    np.rad2deg(HV_aoa_grid[HV_min_D_idx]),
                ]
            )

        # Lift, drag and L/D along trajectory
        traj_disp = np.array([0])
        traj_ground_disp = np.array([0])
        M_avg_total = 0
        for stage in range(stages + 1):
            time = traj_data[f"Time_{stage}"].values
            traj_mach = traj_data[f"mach_{stage}"].values
            traj_aoa = traj_data[f"alpha_{stage}"].values
            traj_q = traj_data[f"q_{stage}"].values
            traj_qalpha = traj_q * np.rad2deg(traj_aoa)
            if stage <= 1 and boost:
                traj_CD, traj_CL = model_dict["aero"]["boost"](
                    traj_mach, traj_aoa, *theta[i]
                )
            else:
                traj_CD, traj_CL = model_dict["aero"]["HV"](
                    traj_mach, traj_aoa, *theta[i]
                )
            traj_CD = traj_CD.full().squeeze()
            traj_CL = traj_CL.full().squeeze()
            traj_LD = traj_CL / traj_CD
            traj_L = traj_q * traj_CL
            traj_D = traj_q * traj_CD
            if stage <= 1 and boost:
                traj_N = traj_L / (
                    vehicle_mass[i] + traj_data[f"m_prop_{stage}"].values + 1221
                )
                traj_aD = traj_D / (
                    vehicle_mass[i] + traj_data[f"m_prop_{stage}"].values + 1221
                )
            else:
                traj_N = traj_L / (vehicle_mass[i])
                traj_aD = traj_D / vehicle_mass[i]

            # Running Length
            valid = ~np.isnan(traj_disp)
            traj_disp_end = traj_disp[valid][-1]
            traj_ground_disp_end = traj_ground_disp[valid][-1]
            traj_disp = np.cumsum(
                np.hstack(
                    (
                        traj_disp_end,
                        np.sqrt(
                            np.diff(traj_data[f"s_x_{stage}"].values) ** 2
                            + np.diff(traj_data[f"s_y_{stage}"].values) ** 2
                            + np.diff(traj_data[f"s_z_{stage}"].values) ** 2
                        ),
                    )
                )
            )
            traj_ground_disp = np.cumsum(
                np.hstack(
                    (
                        traj_ground_disp_end,
                        np.sqrt(
                            np.diff(traj_data[f"s_x_{stage}"].values) ** 2
                            + np.diff(traj_data[f"s_y_{stage}"].values) ** 2
                        ),
                    )
                )
            )
            valid = ~np.isnan(traj_mach)
            M_avg = np.trapezoid(traj_mach[valid], time[valid]) / (time[valid][-1] - time[valid][0])
            M_avg_total += M_avg * (time[valid][-1] - time[valid][0])

            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 1, f"CD_{stage}", traj_CD
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 2, f"CL_{stage}", traj_CL
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 3, f"LD_{stage}", traj_LD
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 4, f"D_{stage}", traj_D
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 5, f"L_{stage}", traj_L
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 6, f"N_{stage}", traj_N
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 7, f"aD_{stage}", traj_aD
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 8, f"q-alpha_{stage}", traj_qalpha
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 9,
                f"disp_{stage}",
                traj_disp,
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 10,
                f"ground_disp_{stage}",
                traj_ground_disp,
            )
            traj_data.insert(
                traj_data.columns.get_loc(f"rho_{stage}") + 11, f"M_avg_{stage}", M_avg
            )

        traj_data.insert(
            0, f"M_avg_total", M_avg_total / time[-1]
        )
        traj_data.to_csv(f"{sim}/trajectory-aero.csv", index=False)

    HV_CL_RMS = np.empty((len(sim_names), len(sim_names)))
    HV_CD_RMS = np.empty((len(sim_names), len(sim_names)))
    for i, sim_A in enumerate(sim_names):
        for j, sim_B in enumerate(sim_names):
            HV_CL_RMS[j, i] = np.linalg.norm(HV_CL[i] - HV_CL[j]) / np.sqrt(
                HV_aoa_grid.size
            )
            HV_CD_RMS[j, i] = np.linalg.norm(HV_CD[i] - HV_CD[j]) / np.sqrt(
                HV_aoa_grid.size
            )

    col_labels = sim_names
    row_labels = sim_names
    HV_CL_RMS_df = pd.DataFrame(HV_CL_RMS, index=row_labels, columns=col_labels)
    HV_CL_RMS_df.to_csv(f"{dir}/HV_CL_RMS.csv")
    HV_CD_RMS_df = pd.DataFrame(HV_CD_RMS, index=row_labels, columns=col_labels)
    HV_CD_RMS_df.to_csv(f"{dir}/HV_CD_RMS.csv")

    if boost:
        row_labels = [
            "HV_LD_Max",
            "HV_LD_max_mach",
            "HV_LD_max_alpha",
            "HV_LD_min",
            "HV_LD_min_mach",
            "HV_LD_min_mach",
            "Boost_LD_Max",
            "Boost_LD_max_mach",
            "Boost_LD_max_alpha",
            "Boost_LD_min",
            "Boost_LD_min_mach",
            "Boost_LD_min_mach",
            "HV_D_min",
            "HV_D_min_mach",
            "HV_D_min_mach",
            "Boost_D_min",
            "Boost_D_min_mach",
            "Boost_D_min_mach",
        ]
    else:
        row_labels = [
            "HV_LD_Max",
            "HV_LD_max_mach",
            "HV_LD_max_alpha",
            "HV_LD_min",
            "HV_LD_min_mach",
            "HV_LD_min_mach",
            "HV_D_min",
            "HV_D_min_mach",
            "HV_D_min_mach",
        ]

    LD_df = pd.DataFrame(np.array(LD).T, index=row_labels, columns=col_labels)
    LD_df.to_csv(f"{dir}/LD.csv")

    col_labels = ["Aoa (deg)", "Mach", *sim_names]
    HV_LD_df = pd.DataFrame(np.array(HV_LD).T, columns=col_labels)
    HV_LD_df.to_csv(f"{dir}/HV_LD.csv")

    ax_HV_CL.set_xlabel("Mach")
    ax_HV_CD.set_xlabel("Mach")
    ax_HV_LD.set_xlabel("Mach")

    ax_HV_CL.set_ylabel("AoA [deg]")
    ax_HV_CD.set_ylabel("AoA [deg]")
    ax_HV_LD.set_ylabel("AoA [deg]")

    ax_HV_CL.legend()
    ax_HV_CD.legend()
    ax_HV_LD.legend()

    ax_HV_CL.set_title("HV CL")
    ax_HV_CD.set_title("HV CD")
    ax_boost_CL.set_title("Boost CL")
    ax_boost_CD.set_title("Boost CD")
    ax_HV_LD.set_title("HV LD")

    fig_HV_CL.savefig(f"{dir}/HV_CL.png", dpi=900)
    fig_HV_CD.savefig(f"{dir}/HV_CD.png", dpi=900)
    fig_HV_LD.savefig(f"{dir}/HV_LD.png", dpi=900)

    if boost:
        boost_LD_df = pd.DataFrame(np.array(boost_LD).T, columns=col_labels)
        boost_LD_df.to_csv(f"{dir}/Boost_LD.csv")
        ax_boost_CL.set_xlabel("Mach")
        ax_boost_CD.set_xlabel("Mach")
        ax_boost_CL.set_ylabel("AoA [deg]")
        ax_boost_CD.set_ylabel("AoA [deg]")
        ax_boost_CL.legend()
        ax_boost_CD.legend()
        fig_boost_CL.savefig(f"{dir}/Boost_CL.png", dpi=900)
        fig_boost_CD.savefig(f"{dir}/Boost_CD.png", dpi=900)

    # plt.show()


def main(mission, vehicles, comp_name):
    sim_names = []
    for vehicle in vehicles:
        sim_names.append(f"Mission-{mission}_Vehicle-{vehicle}")

    theta = param_comparison(sim_names=sim_names, dir=comp_name)
    compare_aero(sim_names=sim_names, theta=theta, dir=comp_name)
    plotting(mission=mission, vehicles=vehicles, dir=comp_name)


if __name__ == "__main__":
    # fname = "Target2_Mission_Sprint2_vs_Target2_Boost-Glide_Comparison"
    fname = "Target2_BG-K_initial_vs_final_Comparison"
    # fname = "Target2_BG-K_vs_BG-L_Comparison"
    # fname = "Target2_BG-K_vs_BG-K_with_Boost_Opt_D"
    mission = "Target 2 Boost-Glide"
    vehicles = [
        "Target 2 Boost-Glide_K",
        # "Target 2 Glide_I with Boost_K",
        # "Target 2 Boost-Glide_L",
        # "Target 2 Boost-Glide_L with Boost_Opt_D",
        "Target 2 Boost-Glide_K Iter_0",
    ]
    # fname = "Sprint2_Boost-Glide_I_vs_Boost-Glide_J_vs_Glide_F_All_with_boost_J_Comparison"
    # mission = "Sprint 2 Boost-Glide"
    # vehicles = [
    #         "Sprint 2 Boost-Glide_I with Boost_J",
    #         "Sprint 2 Glide_F with Boost_J",
    #         "Sprint 2 Boost-Glide_J",
    # ]

    main(mission, vehicles, fname)
