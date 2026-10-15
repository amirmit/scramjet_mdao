import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
from scipy.interpolate import interp1d
import glob
from PIL import Image

labels = {
    "s_x": r"Downrange, $s_x$ (km)",
    "s_y": r"Crossrange, $s_y$ (km)",
    "s_z": r"Altitude, $s_z$ (km)",
    "v": r"Velocity, $v$ (m/s)",
    "chi": r"$Heading Angle, \chi$ ($\degree$)",
    "gamma": r"Flight Path Angle, $\gamma$ ($\degree$)",
    "m_prop": r"1st Stage Propellant Mass (kg)",
    "m_fuel": r"2nd Stage Propellant Mass (kg)",
    "throttle": r"Throttle (%)",
    "phi": r"Roll Angle, $\phi$ ($\degree$)",
    "T_vec": r"Thrust Vectoring Angle ($\degree$)",
    "alpha": r"Angle of Attack, $\alpha$ ($\degree$)",
    "q": r"Dynamic Pressure, q (kPa)",
    "mach": "Mach Number, M",
    "CD": "CD",
    "CL": "CL",
    "LD": "L/D",
    "D": "Drag (N)",
    "L": "Lift (N)",
    "N": "Load Factor",
    "aD": "Aerodynamic Deceleration (m/s$^2$)",
    "heat_flux": "$Q$ (kW/m$^2$)",
    "q-alpha": r"$q\alpha$ (kPa$\degree$)",
}

scale = {
    "s_x": 1e-3,
    "s_y": 1e-3,
    "s_z": 1e-3,
    "v": 1,
    "chi": np.rad2deg(1),
    "gamma": np.rad2deg(1),
    "m_prop": 1,
    "m_fuel": 1,
    "throttle": 100,
    "phi": np.rad2deg(1),
    "T_vec": 1,
    "alpha": np.rad2deg(1),
    "q": 1e-3,
    "mach": 1,
    "CD": 1,
    "CL": 1,
    "LD": 1,
    "D": 1,
    "L": 1,
    "N": 1,
    "aD": 1,
    "heat_flux": 1e-3,
    "q-alpha": 1e-3,
}

boost_glide_names = [
    "s_x",
    "s_y",
    "s_z",
    "v",
    "chi",
    "gamma",
    "m_prop",
    "throttle",
    "phi",
    "T_vec",
    "alpha",
    "q",
    "mach",
    "CD",
    "CL",
    "LD",
    "D",
    "L",
    "N",
    "aD",
    "heat_flux",
    "q-alpha",
]
boost_boost_names = [
    "s_x",
    "s_y",
    "s_z",
    "v",
    "chi",
    "gamma",
    "m_prop",
    "throttle",
    "phi",
    "T_vec",
    "alpha",
    "q",
    "mach",
    "m_fuel",
]
# comp_colour = ["b", "r", "#92985b", "k", "green"]
comp_colour = ["b", "#92985b", "k", "green"]
# comp_colour = ["g", "orange"]
stage_linestyle = ["solid", "dashed", "dotted", "dashdot", ":"]

target_obs = [[500, 0, 150], [1000, 100, 150], [700, 250, 100], [350, -100, 100]]
range_obs = [[700, 225, 300], [1200, -225, 300], [1700, 225, 300]]
xy_offset = [
    # [(7.5, 55), (6.5, 34), (16, 11), (15, 15)],
    # [(47.5, 55), (-40, 34), (15, 35), (15, 15)],
    [(17.5, 42), (-35, 34), (16, 11), (15, 15)],
    [(-30, 52), (-30, 32), (-30, 12), (-30, 15)],
    [(-45, -40), (-45, -60), (-45, -80), (-45, -100)],
    [(-70, 22), (-70, 2), (-70, -18), (-70, -18)],
]
xz_offset = [(-100, 15), (30, 15), (-10, -40), (-20, 15)]
anno_offset = [
    # [(-40, 55)],
    # [(0, 55)],
    [(-30, 42)],
    [(-75, 52)],
    [(-90, -40)],
    [(-115, 22)],
]


def plot_vs_time_all_stages(
    data: pd.DataFrame,
    stages: int,
    data_name: str,
    mission: str,
    vehicle: str,
    dir: str,
):
    fig, ax = plt.subplots()
    for i in range(stages + 1):
        name = f"{data_name}_{i}"
        plot_data = (
            data[name] if name in data.columns else np.zeros(data[f"Time_{i}"].size)
        )
        ax.plot(
            data[f"Time_{i}"],
            plot_data * scale[data_name],
            linestyle=stage_linestyle[i],
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(labels[data_name])

    fig.savefig(
        f"{dir}Mission-{mission}_Vehicle-{vehicle}_{data_name}_vs_time.png", dpi=900
    )


def plot_3D_trajectory_animation(
    data_set: dict[str, pd.DataFrame],
    vehicles: list[str],
    dir: str,
):
    # with geometries
    fig = plt.figure(figsize=(16, 7.5))
    gs = GridSpec(2, 3, width_ratios=[2, 2, 2], height_ratios=[2, 0.4], figure=fig)
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    ax.set_box_aspect([1250, 700, 1000])
    ax_cd = fig.add_subplot(gs[0, 0])
    ax_init = fig.add_subplot(gs[0, 2])
    ax_traj = fig.add_subplot((gs[1, 1]))
    # Without geometires
    # fig = plt.figure(figsize=(16, 7.5))
    # gs = GridSpec(2, 1, width_ratios=[2], height_ratios=[2, 0.4], figure=fig)
    # ax = fig.add_subplot(gs[0, 0], projection="3d")
    # ax.set_box_aspect([1250, 700, 1000])
    # ax_traj = fig.add_subplot(gs[1, 0])

    data = []
    t_max = 0
    ani_obs = []
    labels = [
        [
            "HGV co-designed with fixed booster: Boost-stage",
            "HGV co-designed with fixed booster: Glide-stage",
        ],
        [
            "HGV co-designed as a glide-only: Boost-stage",
            "HGV co-designed as a glide-only: Glide-stage",
        ],
    ]
    # names = ["HGV co-designed with fixed booster", "HGV co-designed as a glide-only"]
    names = ["HGV co-designed with fixed booster", "Full boost-glide system co-designed"]
    for i, vehicle in enumerate(vehicles):
        for stage in range(data_set["stages"] + 1):
            valid = ~np.isnan(data_set[vehicle][f"s_x_{stage}"])
            if stage == 0:
                s_x_b = data_set[vehicle][f"s_x_{stage}"][valid] * scale["s_x"]
                s_y_b = data_set[vehicle][f"s_y_{stage}"][valid] * scale["s_y"]
                s_z_b = data_set[vehicle][f"s_z_{stage}"][valid] * scale["s_z"]
                t_b = data_set[vehicle][f"Time_{stage}"][valid]
                mach_b = data_set[vehicle][f"mach_{stage}"][valid]
            elif stage == 1:
                s_x_b = np.hstack(
                    (s_x_b, data_set[vehicle][f"s_x_{stage}"][valid] * scale["s_x"])
                )
                s_y_b = np.hstack(
                    (s_y_b, data_set[vehicle][f"s_y_{stage}"][valid] * scale["s_y"])
                )
                s_z_b = np.hstack(
                    (s_z_b, data_set[vehicle][f"s_z_{stage}"][valid] * scale["s_z"])
                )
                t_b = np.hstack((t_b, data_set[vehicle][f"Time_{stage}"][valid]))
                mach_b = np.hstack((mach_b, data_set[vehicle][f"mach_{stage}"][valid]))
                ts = t_b[-1]
            elif stage == 2:
                s_x_g = data_set[vehicle][f"s_x_{stage}"][valid] * scale["s_x"]
                s_y_g = data_set[vehicle][f"s_y_{stage}"][valid] * scale["s_y"]
                s_z_g = data_set[vehicle][f"s_z_{stage}"][valid] * scale["s_z"]
                t_g = data_set[vehicle][f"Time_{stage}"][valid]
                mach_g = data_set[vehicle][f"mach_{stage}"][valid]
            else:
                s_x_g = np.hstack(
                    (s_x_g, data_set[vehicle][f"s_x_{stage}"][valid] * scale["s_x"])
                )
                s_y_g = np.hstack(
                    (s_y_g, data_set[vehicle][f"s_y_{stage}"][valid] * scale["s_y"])
                )
                s_z_g = np.hstack(
                    (s_z_g, data_set[vehicle][f"s_z_{stage}"][valid] * scale["s_z"])
                )
                t_g = np.hstack((t_g, data_set[vehicle][f"Time_{stage}"][valid]))
                mach_g = np.hstack((mach_g, data_set[vehicle][f"mach_{stage}"][valid]))

        data.append(
            {
                "boost": {
                    "s_x": interp1d(t_b.copy(), s_x_b.copy()),
                    "s_y": interp1d(t_b.copy(), s_y_b.copy()),
                    "s_z": interp1d(t_b.copy(), s_z_b.copy()),
                    "t": t_b.copy(),
                    "mach": interp1d(t_b.copy(), mach_b.copy()),
                },
                "glide": {
                    "s_x": interp1d(t_g.copy(), s_x_g.copy()),
                    "s_y": interp1d(t_g.copy(), s_y_g.copy()),
                    "s_z": interp1d(t_g.copy(), s_z_g.copy()),
                    "t": t_g.copy(),
                    "mach": interp1d(t_g.copy(), mach_g.copy()),
                },
                "ts": ts.copy(),
                "name": vehicle,
            }
        )

        t_max = max(t_max, t_g[-1])

        # Animated objects
        ani_obs.append(
            {
                "line_1": ax.plot(
                    [], [], [], lw=2, color=comp_colour[i], linestyle="dashed"
                ),  # , label=labels[i][0]),
                "line_2": ax.plot(
                    [], [], [], lw=2, color=comp_colour[i], linestyle="solid"
                ),  # , label=labels[i][1]),
                "point": ax.plot([], [], [], "o", color=comp_colour[i], markersize=10),
                "line_1_g": ax.plot(
                    [],
                    [],
                    [],
                    lw=1,
                    color=comp_colour[i],
                    linestyle="dashed",
                    alpha=0.5,
                ),
                "line_2_g": ax.plot(
                    [], [], [], lw=1, color=comp_colour[i], linestyle="solid", alpha=0.5
                ),
                "line_traj_1": ax_traj.plot(
                    [], [], lw=2, color=comp_colour[i], linestyle="dashed"
                ),  # , label=labels[i][0]),
                "line_traj_2": ax_traj.plot(
                    [], [], lw=2, color=comp_colour[i], linestyle="solid"
                ),  # , label=labels[i][1]),
                "point_traj": ax_traj.plot(
                    [], [], "o", color=comp_colour[i], markersize=10
                ),
            }
        )
    # line_1_b, = ax.plot([], [], [], lw=2, color=comp_colour[0], linestyle="dashed")
    # line_1_g, = ax.plot([], [], [], lw=2, color=comp_colour[0], linestyle=stage_linestyle[1])
    # point_1, = ax.plot([], [], [], 'o', color=comp_colour[0], markersize=12)
    # line_2_b, = ax.plot([], [], [], lw=12, color=comp_colour[1], linestyle=stage_linestyle[0])
    # line_2_g, = ax.plot([], [], [], lw=12, color=comp_colour[1], linestyle=stage_linestyle[1])
    # point_2, = ax.plot([], [], [], 'o', color=comp_colour[1], markersize=12)

    time_text = ax.text2D(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=12,
    )

    # Set limits
    ax.set_xlim(np.min(-50), np.max(1200))
    ax.set_ylim(np.min(-350), np.max(350))
    ax.set_zlim(np.min(0), np.max(60))

    ax.set_xlabel(r"$s_x$ (km)", fontsize=12)
    ax.set_ylabel(r"$s_y$ (km)", fontsize=12)
    ax.set_zlabel(r"$s_z$ (km)", fontsize=12)
    ax.tick_params(labelsize=10)

    ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.plot([], [], [], color="k", linestyle="dashed", label="Boost-Stage")
    ax.plot([], [], [], color="k", linestyle="solid", label="Glide-Stage")

    # Mach number plotting
    ax_traj.set_xlim(0, 600)
    ax_traj.set_ylim(0, 18)
    ax_traj.set_xlabel("Time (s)", fontsize=12)
    ax_traj.set_ylabel("Mach", fontsize=12)
    ax_traj.grid(True)
    ax_traj.tick_params(labelsize=10)

    n = 450
    t_global = np.linspace(0, t_max, n)
    n_v = np.arange(0, n, 10)

    # Vehicle geometry
    # cd_tiff_files = sorted(glob.glob(f"{dir}/BG-K/*.tiff"))
    # init_tiff_files = sorted(glob.glob(f"{dir}/G-I/*.tiff"))
    # init_tiff_files = sorted(glob.glob(f"{dir}/BG-L/*.tiff"))
    # cd_frames = [np.array(Image.open(f)) for f in cd_tiff_files]
    # init_frames = [np.array(Image.open(f)) for f in init_tiff_files]

    # cd_im = ax_cd.imshow(cd_frames[0])
    # init_im = ax_init.imshow(init_frames[0])
    ax_cd.axis("off")
    ax_init.axis("off")
    # ax_cd.set_title("HGV co-designed with fixed booster")
    # ax_init.set_title("Co-designed boost-glide system")

    def update(frame):

        # frame_img = frame % len(cd_frames)

        if frame >= len(t_global):
            # t = t_global[-1]
            frame = len(t_global) - 1

        t = t_global[frame]
        txt = f"t = {t:.0f} s"

        returns = []
        # print(t)
        for i in range(len(vehicles)):
            # print(i)
            # print(data[i]["ts"])
            if t <= data[i]["ts"]:
                ani_obs[i]["line_1"][0].set_data(
                    data[i]["boost"]["s_x"](t_global[:frame]),
                    data[i]["boost"]["s_y"](t_global[:frame]),
                )
                ani_obs[i]["line_1"][0].set_3d_properties(
                    data[i]["boost"]["s_z"](t_global[:frame]),
                )
                ani_obs[i]["point"][0].set_data(
                    [data[i]["boost"]["s_x"](t_global[frame])],
                    [data[i]["boost"]["s_y"](t_global[frame])],
                )
                ani_obs[i]["point"][0].set_3d_properties(
                    [data[i]["boost"]["s_z"](t_global[frame])],
                )
                ani_obs[i]["line_1_g"][0].set_data(
                    data[i]["boost"]["s_x"](t_global[:frame]),
                    data[i]["boost"]["s_y"](t_global[:frame]),
                )
                ani_obs[i]["line_1_g"][0].set_3d_properties(
                    np.zeros_like(t_global[:frame]),
                )
                ani_obs[i]["line_traj_1"][0].set_data(
                    t_global[:frame], data[i]["boost"]["mach"](t_global[:frame])
                )
                ani_obs[i]["point_traj"][0].set_data(
                    [t_global[frame]], [data[i]["boost"]["mach"](t_global[frame])]
                )
                if frame in n_v:
                    ax.plot(
                        [data[i]["boost"]["s_x"](t), data[i]["boost"]["s_x"](t)],
                        [data[i]["boost"]["s_y"](t), data[i]["boost"]["s_y"](t)],
                        [0, data[i]["boost"]["s_z"](t)],
                        color=comp_colour[i],
                        alpha=0.2,
                    )
            elif data[i]["ts"] <= t <= data[i]["glide"]["t"][-1]:
                idx = np.argmin(np.abs(t_global - data[i]["ts"]))
                ani_obs[i]["line_2"][0].set_data(
                    data[i]["glide"]["s_x"](t_global[idx + 1 : frame]),
                    data[i]["glide"]["s_y"](t_global[idx + 1 : frame]),
                )
                ani_obs[i]["line_2"][0].set_3d_properties(
                    data[i]["glide"]["s_z"](t_global[idx + 1 : frame]),
                )
                ani_obs[i]["point"][0].set_data(
                    [data[i]["glide"]["s_x"](t_global[frame])],
                    [data[i]["glide"]["s_y"](t_global[frame])],
                )
                ani_obs[i]["point"][0].set_3d_properties(
                    [data[i]["glide"]["s_z"](t_global[frame])],
                )
                ani_obs[i]["line_2_g"][0].set_data(
                    data[i]["glide"]["s_x"](t_global[idx + 1 : frame]),
                    data[i]["glide"]["s_y"](t_global[idx + 1 : frame]),
                )
                ani_obs[i]["line_2_g"][0].set_3d_properties(
                    np.zeros_like(t_global[idx + 1 : frame]),
                )
                ani_obs[i]["line_traj_2"][0].set_data(
                    t_global[idx + 1:frame], data[i]["glide"]["mach"](t_global[idx + 1:frame])
                )
                ani_obs[i]["point_traj"][0].set_data(
                    [t_global[frame]], [data[i]["glide"]["mach"](t_global[frame])]
                )
                if frame in n_v:
                    ax.plot(
                        [data[i]["glide"]["s_x"](t), data[i]["glide"]["s_x"](t)],
                        [data[i]["glide"]["s_y"](t), data[i]["glide"]["s_y"](t)],
                        [0, data[i]["glide"]["s_z"](t)],
                        color=comp_colour[i],
                        alpha=0.2,
                    )
            # if t >= data[i]["glide"]["t"][-1]:
            #     txt += f"\n{names[i]} t_final={data[i]['glide']['t'][-1]:.0f} s"
            # else:
            #     txt += "\n"
            returns.append(ani_obs[i]["line_1"][0])
            returns.append(ani_obs[i]["line_2"][0])
            returns.append(ani_obs[i]["line_1_g"][0])
            returns.append(ani_obs[i]["line_2_g"][0])
        # ax.view_init(elev=30, azim=frame * 0.5)
        time_text.set_text(txt)
        # returns = returns.append(time_text)

        # Update geomtrey images
        # cd_im.set_data(cd_frames[frame_img])
        # init_im.set_data(init_frames[frame_img])

        return (*returns,)

    ax.legend(fontsize=12, loc="lower center", ncols=2, bbox_to_anchor=(0.5,-0.1))
    ax.set_proj_type("ortho")
    plt.tight_layout()
    # fig.savefig(f"{dir}/Test_fig.png")
    ani = FuncAnimation(fig, update, frames=len(t_global) + 1500)
    ani.save(
        f"{dir}/BG-K_vs_BG-L_animation_traj_only.gif",
        writer="pillow",
        fps=15,
    )

    fig.savefig(f"{dir}/3D_trajectory_prezzo.png", dpi=900)


def plot_3D_trajectory(
    data: pd.DataFrame,
    stages: int,
    mission: str,
    vehicle: str,
    dir: str,
):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    steps = [20, 10, 10]
    for i in range(stages + 1):
        x = data[f"s_x_{i}"] * scale["s_x"]
        y = data[f"s_y_{i}"] * scale["s_y"]
        z = data[f"s_z_{i}"] * scale["s_z"]
        ax.plot(x, y, z, color=comp_colour[i])

        step = steps[i]
        for xi, yi, zi in zip(x[::step], y[::step], z[::step]):
            ax.plot([xi, xi], [yi, yi], [zi, 0], color="gray", linewidth=0.5)

        ax.plot(x, y, 0, color=comp_colour[i], linestyle="--", linewidth=0.5)

    if mission == "Target":
        z_min = 0
        z_max = 50
        x_min = -50
        x_max = 1250
        y_min = -200
        y_max = 250
        for obs in target_obs:
            plot_cylinder(ax, obs[0], obs[1], obs[2], z_min=z_min, z_max=z_max)

    elif mission == "Obstruction":
        z_min = 0
        z_max = 55
        x_min = -50
        x_max = 5250
        y_min = -200
        y_max = 200
        for obs in range_obs:
            plot_cylinder(ax, obs[0], obs[1], obs[2], z_min=0, z_max=55)

    # ax.set_zlim([z_min, z_max])
    # ax.set_xlim([x_min, x_max])
    # ax.set_ylim([y_min, y_max])
    # ax.set_box_aspect([x_max-x_min, y_max-y_min, 1000])

    ax.set_xlabel(labels["s_x"])
    ax.set_ylabel(labels["s_y"])
    ax.set_zlabel(labels["s_z"])

    fig.savefig(f"{dir}Mission-{mission}_Vehicle-{vehicle}_3D_trajectory.png", dpi=900)

    def update(frame):
        ax.view_init(elev=30, azim=frame)  # rotate camera
        return fig

    ani = FuncAnimation(fig, update, frames=360, interval=50)
    ani.save(
        f"{dir}Mission-{mission}_Vehicle-{vehicle}_3D_trajectory.gif",
        writer="pillow",
        fps=15,
    )

    # plt.show()


def plot_cylinder(ax, x0, y0, r, z_min, z_max, alpha=0.2, colour="r"):
    theta = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(z_min, z_max, 50)
    theta, z = np.meshgrid(theta, z)

    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    ax.plot_surface(x, y, z, alpha=alpha, color=colour)


def comparison(
    data_set: dict[str, pd.DataFrame],
    stages: int,
    data_name: str,
    mission: str,
    vehicles: list[str],
    dir: str,
):
    fig, ax = plt.subplots(figsize=(10, 3))
    for j, vehicle in enumerate(vehicles):
        data = data_set[vehicle]
        for i in range(stages + 1):
            name = f"{data_name}_{i}"
            plot_data = (
                data[name] if name in data.columns else np.zeros(data[f"Time_{i}"].size)
            )
            if i == 0:
                ax.plot(
                    data[f"Time_{i}"],
                    plot_data * scale[data_name],
                    color=comp_colour[j],
                    linestyle=stage_linestyle[i],
                    label=f"{vehicle}",
                )
            else:
                ax.plot(
                    data[f"Time_{i}"],
                    plot_data * scale[data_name],
                    color=comp_colour[j],
                    linestyle=stage_linestyle[i],
                )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(labels[data_name])
    ax.legend()
    plt.tight_layout()

    fig.savefig(f"{dir}Comparison_{data_name}_vs_time.png", dpi=900)


def load_data(mission: str, vehicles: list[str]):
    data_set = {}
    for vehicle in vehicles:
        dir = f"Mission-{mission}_Vehicle-{vehicle}/"
        name = f"{dir}trajectory-aero.csv"
        data = pd.read_csv(name)
        stages = int(data.columns.values[-1][-1])
        data_set[f"{vehicle}"] = data.copy()
        data_set[f"{vehicle}_dir"] = dir
    data_set["stages"] = stages
    return data_set


def vertical_lines(stage, x, y, z, ax, vehicle_num):
    steps = [2, 10, 10, 10, 10]
    step = steps[stage]
    for xi, yi, zi in zip(x[::step], y[::step], z[::step]):
        ax.plot(
            [xi, xi],
            [yi, yi],
            [zi, 0],
            color=comp_colour[vehicle_num],
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )


def paper_plotting(
    data_set: dict[str, pd.DataFrame],
    vehicles: list[str],
    dir: str,
):
    mission = "Target"  # "Target", "Sprint"

    fig, ax = plt.subplots(9, 2, sharex=True, figsize=(15, 15))
    ax = ax.flatten()
    fig_xy, ax_xy = plt.subplots(figsize=(20, 7))
    fig_xy_A, ax_xy_A = plt.subplots(figsize=(20, 7))
    fig_tz, ax_tz = plt.subplots(figsize=(20, 7))
    fig_aero, ax_aero = plt.subplots(3, 2, sharex=True, figsize=(15, 7))
    ax_aero = ax_aero.flatten()
    fig_xyz = plt.figure(figsize=(15, 15))
    ax_xyz = fig_xyz.add_subplot(projection="3d")

    lw = 2
    fs = 14

    traj_data_names = [
        # "s_x",
        # "s_y",
        "s_z",
        # "v",
        "mach",
        "chi",
        "gamma",
        "alpha",
        "phi",
        "q",
        "throttle",
        "heat_flux",
        "m_prop",
        "q-alpha",
        "LD",
        "CL",
        "CD",
        "L",
        "D",
        "N",
        "aD",
    ]
    data_labels = {
        "s_x": r"$s_x$ (km)",
        "s_y": r"$s_y$ (km)",
        "s_z": r"$s_z$ (km)",
        "v": r"$v$ (m/s)",
        "mach": r"M",
        "chi": r"$\psi$ ($\degree$)",
        "gamma": r"$\gamma$ ($\degree$)",
        "phi": r"$\phi$ ($\degree$)",
        "alpha": r"$\alpha$ ($\degree$)",
        "q": r"$q$ (kPa)",
        "heat_flux": r"$Q$ (kW/m$^2$)",
        "m_prop": r"$m_p$ (kg)",
        "throttle": r"$\delta$ (%)",
        "LD": r"$L/D$",
        "CL": r"$C_L$",
        "CD": r"$C_D$",
        "L": r"$L$ (N)",
        "D": r"$D$ (N)",
        "N": r"$a_L$ (m/s$^2$)",
        "aD": r"$a_D$ (m/s$^2$)",
        "q-alpha": r"$q\alpha$ (kPa$\degree$)",
    }
    limits = {
        "s_x": [None, None],
        "s_y": [None, None],
        "s_z": [[0, 60], [30, 60]],
        "v": [None, None],
        "mach": [[0, 25], [2, 25]],
        "chi": [None, None],
        "gamma": [None, None],
        "phi": [[-90, 90], [-90, 90]],
        "alpha": [[-10, 10], [-5, 15]],
        "q": [None, [0, 100]],
        "heat_flux": [[None, 6000], [None, 6000]],
        "m_prop": [None, None],
        "throttle": [[70, 100], None],
        "CL": [None, None],
        "CD": [None, None],
        "L": [None, None],
        "D": [None, None],
        "N": [None, None],
        "LD": [None, None],
        "aD": [None, None],
        "q-alpha": [[-168, 168], [None, None]],
    }
    aero_data_names = [
        "CL",
        "CD",
        "L",
        "D",
        "N",
        "LD",
    ]
    aero_data_labels = {
        "LD": r"$L/D$",
        "CL": r"$C_L$",
        "CD": r"$C_D$",
        "L": r"$L$ (N)",
        "D": r"$D$ (N)",
        "N": r"$N$",
    }

    if mission == "Target":
        vehicle_names = [
            r"$\overline{\text{B}}_6$-G$_6$",
            r"$\overline{\text{B}}_6$-G$_\text{O}$",
            "B2-G3",
        ]
        # vehicle_names = [r"B$_\text{C}$-G$_6$", r"B$_\text{C}$-G$_\text{O}$", r"B$_\text{C}$-G$_\text{C}$"]
        # vehicle_names = [r"$\overline{\text{B}}_6$-G$_6$", r"B$_\text{C}$-G$_\text{C}$"]
    elif mission == "Sprint":
        vehicle_names = ["S-BG1", "S-G", "S-BG2"]
    else:
        raise ValueError(f"Mission name {mission} not known.")

    for j, vehicle in enumerate(vehicles):
        data = data_set[vehicle]
        for k in range(data_set["stages"] + 1):
            valid = ~np.isnan(data[f"s_x_{k}"])
            x_end = data[f"s_x_{k}"][valid].iloc[-1] / 1000
            y_end = data[f"s_y_{k}"][valid].iloc[-1] / 1000
            z_end = data[f"s_z_{k}"][valid].iloc[-1] / 1000
            t_end = data[f"Time_{k}"][valid].iloc[-1]
            if k == 0:
                ax_xy.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="dashed",
                    label=f"{vehicle_names[j]} - Boost Stage",
                    lw=lw * 2,
                )
                ax_tz.plot(
                    data[f"Time_{k}"],
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="dashed",
                    label=f"{vehicle_names[j]} - Boost Stage",
                    lw=lw * 2,
                )
                ax_xyz.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="dashed",
                    label=f"{vehicle_names[j]} - Boost Stage",
                    lw=lw * 2,
                )
                vertical_lines(
                    k,
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    ax_xyz,
                    j,
                )
            elif k == 1:
                ax_xy.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="dashed",
                    lw=lw * 2,
                )
                ax_xyz.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="dashed",
                    lw=lw * 2,
                )
                vertical_lines(
                    k,
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    ax_xyz,
                    j,
                )
                ax_tz.plot(
                    data[f"Time_{k}"],
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="dashed",
                    lw=lw * 2,
                )
                ax_xy.plot(
                    x_end,
                    y_end,
                    "o",
                    markersize=8,
                    color=comp_colour[j],
                    lw=lw * 2,
                )
                ax_tz.plot(
                    t_end,
                    z_end,
                    "o",
                    markersize=8,
                    color=comp_colour[j],
                    lw=lw * 2,
                )
                ax_xyz.plot(
                    x_end,
                    y_end,
                    z_end,
                    "o",
                    markersize=8,
                    color=comp_colour[j],
                    lw=lw * 2,
                )
                if j == 0:
                    ax_xy.annotate(
                        r"$t_s =$",
                        xy=(x_end, y_end),
                        xytext=anno_offset[k - 1][j],
                        color="k",
                        fontsize=fs * 1.5,
                        textcoords="offset points",
                    )
                ax_xy.annotate(
                    rf"{t_end:.0f} s",
                    xy=(x_end, y_end),
                    xytext=xy_offset[k - 1][j],
                    color=comp_colour[j],
                    fontsize=fs * 1.5,
                    textcoords="offset points",
                    # arrowprops=dict(arrowstyle="->", color="k"),
                )
                ax_tz.annotate(
                    r"$t_s$: " + f"{t_end:.0f} s",
                    xy=(t_end, z_end),
                    xytext=xz_offset[j],
                    color=comp_colour[j],
                    fontsize=fs,
                    textcoords="offset points",
                    # arrowprops=dict(arrowstyle="->", color=comp_colour[j]),
                )
            elif k == 2:
                ax_xy.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="solid",
                    label=f"{vehicle_names[j]} - Glide Stage",
                    lw=lw * 2,
                )
                ax_tz.plot(
                    data[f"Time_{k}"],
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="solid",
                    label=f"{vehicle_names[j]} - Glide Stage",
                    lw=lw * 2,
                )
                ax_xyz.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="solid",
                    label=f"{vehicle_names[j]} - Glide Stage",
                    lw=lw * 2,
                )
                vertical_lines(
                    k,
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    ax_xyz,
                    j,
                )
                if k == data_set["stages"]:
                    ax_xy.plot(
                        x_end,
                        y_end,
                        "o",
                        markersize=8,
                        color=comp_colour[j],
                        lw=lw * 2,
                    )
                    ax_tz.plot(
                        t_end,
                        z_end,
                        "o",
                        markersize=8,
                        color=comp_colour[j],
                        lw=lw * 2,
                    )
                    ax_xyz.plot(
                        x_end,
                        y_end,
                        z_end,
                        "o",
                        markersize=8,
                        color=comp_colour[j],
                        lw=lw * 2,
                    )
                    if j == 0:
                        ax_xy.annotate(
                            r"$t_f =$",
                            xy=(x_end, y_end),
                            xytext=anno_offset[k - 1][j],
                            color="k",
                            fontsize=fs * 1.5,
                            textcoords="offset points",
                        )
                    ax_xy.annotate(
                        rf"{t_end:.0f} s",
                        xy=(x_end, y_end),
                        xytext=xy_offset[k - 1][j],
                        color=comp_colour[j],
                        fontsize=fs * 1.5,
                        textcoords="offset points",
                        # arrowprops=dict(arrowstyle="->", color="k"),
                    )
                else:
                    if j == 0:
                        ax_xy.annotate(
                            rf"$t_{k - 1} =$",
                            xy=(x_end, y_end),
                            xytext=anno_offset[k - 1][j],
                            color="k",
                            fontsize=fs * 1.5,
                            textcoords="offset points",
                        )
                    ax_xy.annotate(
                        rf"{t_end:.0f} s",
                        xy=(x_end, y_end),
                        xytext=xy_offset[k - 1][j],
                        color=comp_colour[j],
                        fontsize=fs * 1.5,
                        textcoords="offset points",
                        # arrowprops=dict(arrowstyle="->", color="k"),
                    )
            else:
                ax_xy.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="solid",
                    lw=lw * 2,
                )
                ax_tz.plot(
                    data[f"Time_{k}"],
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="solid",
                    lw=lw * 2,
                )
                ax_xyz.plot(
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    color=comp_colour[j],
                    linestyle="solid",
                    lw=lw * 2,
                )
                vertical_lines(
                    k,
                    data[f"s_x_{k}"] / 1000,
                    data[f"s_y_{k}"] / 1000,
                    data[f"s_z_{k}"] / 1000,
                    ax_xyz,
                    j,
                )
                if k == data_set["stages"]:
                    x_end = data[f"s_x_{k}"].iloc[-1] / 1000
                    y_end = data[f"s_y_{k}"].iloc[-1] / 1000
                    z_end = data[f"s_z_{k}"].iloc[-1] / 1000
                    t_end = data[f"Time_{k}"].iloc[-1]
                    ax_xy.plot(
                        x_end,
                        y_end,
                        "o",
                        markersize=8,
                        color=comp_colour[j],
                        lw=lw * 2,
                    )
                    ax_tz.plot(
                        t_end,
                        z_end,
                        "o",
                        markersize=8,
                        color=comp_colour[j],
                        lw=lw * 2,
                    )
                    ax_xyz.plot(
                        x_end,
                        y_end,
                        z_end,
                        "o",
                        markersize=8,
                        color=comp_colour[j],
                        lw=lw * 2,
                    )
                    if j == 0:
                        ax_xy.annotate(
                            rf"$t_f =$",
                            xy=(x_end, y_end),
                            xytext=anno_offset[k - 1][j],
                            color="k",
                            fontsize=fs * 1.5,
                            textcoords="offset points",
                        )
                    ax_xy.annotate(
                        rf"{t_end:.0f} s",
                        xy=(x_end, y_end),
                        xytext=xy_offset[k - 1][j],
                        color=comp_colour[j],
                        fontsize=fs * 1.5,
                        textcoords="offset points",
                        # arrowprops=dict(arrowstyle="->", color="k"),
                    )
                else:
                    if j == 0:
                        ax_xy.annotate(
                            rf"$t_{k - 1} =$",
                            xy=(x_end, y_end),
                            xytext=anno_offset[k - 1][j],
                            color="k",
                            fontsize=fs * 1.5,
                            textcoords="offset points",
                        )
                    ax_xy.annotate(
                        rf"{t_end:.0f} s",
                        xy=(x_end, y_end),
                        xytext=xy_offset[k - 1][j],
                        color=comp_colour[j],
                        fontsize=fs * 1.5,
                        textcoords="offset points",
                        # arrowprops=dict(arrowstyle="->", color="k"),
                    )
            for i, name in enumerate(traj_data_names):
                if k == 0:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="dashed",
                        label=f"{vehicle_names[j]} - Boost Stage",
                        lw=lw,
                    )
                elif k == 1:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="dashed",
                        lw=lw,
                    )
                    ax[i].plot(
                        data[f"Time_{k}"][valid].iloc[-1],
                        data[f"{name}_{k}"][valid].iloc[-1] * scale[name],
                        "o",
                        markersize=4,
                        color=comp_colour[j],
                        lw=lw,
                    )
                elif name in ["m_prop"] and k == 2:
                    ax[i].plot(
                        data[f"Time_{k}"][0],
                        data[f"{name}_{k}"][0] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        label=f"{vehicle_names[j]} - Glide Stage",
                        lw=lw,
                    )
                elif name in ["m_prop", "throttle"]:
                    continue
                elif k == 2:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        label=f"{vehicle_names[j]} - Glide Stage",
                        lw=lw,
                    )
                else:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        lw=lw,
                    )
            for i, name in enumerate(aero_data_names):
                if k == 0:
                    ax_aero[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="dashed",
                        label=f"{vehicle_names[j]} - Boost Stage",
                        lw=lw,
                    )
                elif k == 1:
                    ax_aero[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="dashed",
                        lw=lw,
                    )
                    ax_aero[i].plot(
                        data[f"Time_{k}"][valid].iloc[-1],
                        data[f"{name}_{k}"][valid].iloc[-1] * scale[name],
                        "o",
                        markersize=4,
                        color=comp_colour[j],
                        lw=lw,
                    )
                elif k == 2:
                    ax_aero[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        label=f"{vehicle_names[j]} - Glide Stage",
                        lw=lw,
                    )
                else:
                    ax_aero[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        lw=lw,
                    )

    ts = data_set[vehicles[0]]["Time_2"][0]
    xs = data_set[vehicles[0]]["s_x_1"][valid].iloc[-1] / 1000
    for axi, name in zip(ax, traj_data_names):
        axi.set_ylabel(data_labels[name], fontsize=fs)
        axi.set_axisbelow(True)
        axi.grid(True)
        xmin, xmax = axi.get_xlim()
        axi.set_xlim(xmin, xmax)
        ymin, ymax = axi.get_ylim()
        axi.set_ylim(ymin, ymax)
        axi.tick_params(axis="both", labelsize=fs)
        for i in range(2):
            if limits[name][i]:
                if i == 0:
                    axi.hlines(
                        limits[name][i][0],
                        xmin,
                        ts,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )
                    axi.hlines(
                        limits[name][i][1],
                        xmin,
                        ts,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )
                else:
                    axi.hlines(
                        limits[name][i][0],
                        ts,
                        xmax,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )
                    axi.hlines(
                        limits[name][i][1],
                        ts,
                        xmax,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )

    xmin, xmax = ax_tz.get_xlim()
    ax_tz.set_xlim(xmin, xmax)
    ax_tz.hlines(60, xmin, xmax, color="k", linestyle="dashed", lw=lw * 2, zorder=1)
    ax_tz.hlines(30, ts, xmax, color="k", linestyle="dashed", lw=lw * 2, zorder=1)
    ax_tz.tick_params(axis="both", labelsize=fs * 2)

    for axi, name in zip(ax_aero, aero_data_names):
        axi.set_ylabel(aero_data_labels[name], fontsize=fs)
        axi.set_axisbelow(True)
        axi.grid(True)
        # xmin, xmax = axi.get_xlim()
        # axi.set_xlim(xmin, xmax)
        # ymin, ymax = axi.get_ylim()
        # axi.set_ylim(ymin, ymax)
        axi.tick_params(axis="both", labelsize=fs)

    # for j, vehicle in enumerate(vehicles):
    #     for t in [ts, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    #         ax_xy.annotate(
    #             f"{t:.0f} s",
    #             xy=(x, y),
    #             # xytext=xy_offset[j],
    #             color=comp_colour[j],
    #             fontsize=fs * 2,
    #             textcoords="offset points",
    #             # arrowprops=dict(arrowstyle="->", color=comp_colour[j]),
    #         )

    start = {
        "linestyle": "None",
        "marker": ".",
        "color": "k",
        "markersize": lw * 15,
        "lw": lw * 20,
    }
    ax_xy.plot(
        0,
        0,
        **start,
    )  # , label="Start")
    ax_xy_A.plot(
        0,
        0,
        label="Start",
        **start,
    )
    ax_xy_A.annotate(
        "Start",
        xy=(0, 0),
        xytext=(10, 10),
        color="k",
        fontsize=fs * 2.5,
        textcoords="offset points",
            )
    if mission == "Target":
        waypoint1 = {
            "linestyle": "None",
            "marker": "X",
            # "color": "purple",
            "color": "orange",
            "markersize": lw * 15,
            "lw": lw * 5,
        }
        waypoint2 = {
            "linestyle": "None",
            "marker": "X",
            # "color": "purple",
            "color": "orange",
            "markersize": lw * 15,
            "lw": lw * 5,
        }
        target = {
            "linestyle": "None",
            "marker": "X",
            "color": "k",
            "markersize": lw * 15,
            "lw": lw * 5,
        }
        ax_xy.plot(500, 0, **waypoint1)
        ax_xy_A.plot(500, 0, **waypoint1, label="Waypoint 1")
        ax_xy.plot(1000, 300, **waypoint2)
        ax_xy_A.plot(1000, 300, **waypoint2, label="Waypoint 2")
        ax_xy.plot(1000, 0, **target)
        ax_xy_A.plot(1000, 0, **target, label="Target")
        ax_xy_A.annotate(
            "Waypoint 1",
            xy=(500, 0),
            xytext=(-90, 30),
            color="k",
            fontsize=fs * 2.5,
            textcoords="offset points",
                )
        ax_xy_A.annotate(
            "Waypoint 2",
            xy=(1000, 300),
            xytext=(-150, -50),
            color="k",
            fontsize=fs * 2.5,
            textcoords="offset points",
                )
        ax_xy_A.annotate(
            "Target",
            xy=(1000, 0),
            xytext=(-60, 30),
            color="k",
            fontsize=fs * 2.5,
            textcoords="offset points",
                )
    ax_xy.tick_params(axis="both", labelsize=fs * 2)
    ax_xy_A.tick_params(axis="both", labelsize=fs * 2)
    ax_xyz.tick_params(axis="both", labelsize=fs * 2)

    ax[-1].set_xlabel("Time (s)", fontsize=fs)
    ax[-2].set_xlabel("Time (s)", fontsize=fs)
    ax_xy.set_ylabel(r"$s_y$ (km)", fontsize=fs * 2)
    ax_xy.set_xlabel(r"$s_x$ (km)", fontsize=fs * 2)
    ax_xy_A.set_ylabel(r"$s_y$ (km)", fontsize=fs * 2)
    ax_xy_A.set_xlabel(r"$s_x$ (km)", fontsize=fs * 2)
    ax_tz.set_ylabel(r"$s_z$ (km)", fontsize=fs * 2)
    ax_tz.set_xlabel("Time (s)", fontsize=fs * 2)
    ax[0].legend(
        fontsize=fs, frameon=True, bbox_to_anchor=(1.6, 1.8), ncols=2, loc="upper right"
    )
    # ax[0].legend(
    #     fontsize=fs, frameon=True, bbox_to_anchor=(1.9, 1.8), ncols=3, loc="upper right"
    # )
    ax[7].set_ylim(65, 105)
    ax[-3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[-4].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax_xy.legend(fontsize=fs*2, frameon=True)
    # ax_xy_A.legend(fontsize=fs * 2, frameon=True)
    # ax_xz.legend(fontsize=fs*2, frameon=True, loc="lower right")
    ax_xy.set_aspect("equal")
    ax_xy_A.set_aspect("equal")
    ax_aero[-1].set_xlabel("Time (s)", fontsize=fs)
    ax_aero[-2].set_xlabel("Time (s)", fontsize=fs)
    ax_aero[0].legend(fontsize=fs, frameon=True, bbox_to_anchor=(0.5, 1.02), ncols=2)
    ax_xyz.set_ylabel(r"$s_y$ (km)", fontsize=fs * 2)
    ax_xyz.set_xlabel(r"$s_x$ (km)", fontsize=fs * 2)
    ax_xyz.set_zlabel(r"$s_z$ (km)", fontsize=fs * 2)
    ax_xyz.set_box_aspect([1200, 400, 500])
    ax_xyz.view_init(elev=45, azim=-135)

    ax_xy.set_axisbelow(True)
    ax_xy.grid(True)
    ax_xy_A.set_axisbelow(True)
    ax_xy_A.grid(True)
    ax_tz.set_axisbelow(True)
    ax_tz.grid(True)
    # ax_xyz.set_axisbelow(True)
    ax_xyz.grid(True)
    fig_xyz.savefig(f"{dir}{mission}_3D_plot.pdf")

    fig.align_ylabels()

    plt.tight_layout()

    fig.savefig(f"{dir}{mission}_Trajectory.pdf")
    fig_xy.savefig(f"{dir}{mission}_Ground_track.pdf")
    fig_tz.savefig(f"{dir}{mission}_Side_profile.pdf")
    fig_xy_A.savefig(f"{dir}{mission}_mission_profile.pdf")
    fig_xy_A.savefig(f"{dir}{mission}_mission_profile.png", dpi=1200)
    # fig_aero.savefig(f"{dir}{mission}_Aero_Trajectory.png", dpi=1200)

def prezzo_plotting(
    data_set: dict[str, pd.DataFrame],
    vehicles: list[str],
    dir: str,
):
    mission = "Target"  # "Target", "Sprint"

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax = ax.flatten()

    lw = 2
    fs = 14

    traj_data_names = [
        # "s_x",
        # "s_y",
        # "s_z",
        # "v",
        # "mach",
        # "chi",
        # "gamma",
        # "alpha",
        # "phi",
        "q",
        # "throttle",
        "heat_flux",
        # "m_prop",
        # "q-alpha",
        "LD",
        # "CL",
        # "CD",
        # "L",
        # "D",
        # "N",
        # "aD",
    ]
    data_labels = {
        "s_x": r"$s_x$ (km)",
        "s_y": r"$s_y$ (km)",
        "s_z": r"$s_z$ (km)",
        "v": r"$v$ (m/s)",
        "mach": r"M",
        "chi": r"$\psi$ ($\degree$)",
        "gamma": r"$\gamma$ ($\degree$)",
        "phi": r"$\phi$ ($\degree$)",
        "alpha": r"$\alpha$ ($\degree$)",
        "q": r"Dynamic pressure (kPa)",
        "heat_flux": r"Heat flux (kW/m$^2$)",
        "m_prop": r"$m_p$ (kg)",
        "throttle": r"$\delta$ (%)",
        "LD": r"$L/D$",
        "CL": r"$C_L$",
        "CD": r"$C_D$",
        "L": r"$L$ (N)",
        "D": r"$D$ (N)",
        "N": r"$a_L$ (m/s$^2$)",
        "aD": r"$a_D$ (m/s$^2$)",
        "q-alpha": r"$q\alpha$ (kPa$\degree$)",
    }
    limits = {
        "s_x": [None, None],
        "s_y": [None, None],
        "s_z": [[0, 60], [30, 60]],
        "v": [None, None],
        "mach": [[0, 25], [2, 25]],
        "chi": [None, None],
        "gamma": [None, None],
        "phi": [[-90, 90], [-90, 90]],
        "alpha": [[-10, 10], [-5, 15]],
        "q": [None, [0, 100]],
        "heat_flux": [[None, 6000], [None, 6000]],
        "m_prop": [None, None],
        "throttle": [[70, 100], None],
        "CL": [None, None],
        "CD": [None, None],
        "L": [None, None],
        "D": [None, None],
        "N": [None, None],
        "LD": [None, None],
        "aD": [None, None],
        "q-alpha": [[-168, 168], [None, None]],
    }

    if mission == "Target":
        vehicle_names = [
            r"$\overline{\text{B}}_6$-G$_6$",
            r"$\overline{\text{B}}_6$-G$_\text{O}$",
        ]
        # vehicle_names = [r"B$_\text{C}$-G$_6$", r"B$_\text{C}$-G$_\text{O}$", r"B$_\text{C}$-G$_\text{C}$"]
        # vehicle_names = [r"$\overline{\text{B}}_6$-G$_6$", r"B$_\text{C}$-G$_\text{C}$"]
    elif mission == "Sprint":
        vehicle_names = ["S-BG1", "S-G", "S-BG2"]
    else:
        raise ValueError(f"Mission name {mission} not known.")

    for j, vehicle in enumerate(vehicles):
        data = data_set[vehicle]
        for k in range(data_set["stages"] + 1):
            valid = ~np.isnan(data[f"s_x_{k}"])
            for i, name in enumerate(traj_data_names):
                if k == 0:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="dashed",
                        label=f"{vehicle_names[j]} - Boost Stage",
                        lw=lw,
                    )
                elif k == 1:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="dashed",
                        lw=lw,
                    )
                    ax[i].plot(
                        data[f"Time_{k}"][valid].iloc[-1],
                        data[f"{name}_{k}"][valid].iloc[-1] * scale[name],
                        "o",
                        markersize=4,
                        color=comp_colour[j],
                        lw=lw,
                    )
                elif name in ["m_prop"] and k == 2:
                    ax[i].plot(
                        data[f"Time_{k}"][0],
                        data[f"{name}_{k}"][0] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        label=f"{vehicle_names[j]} - Glide Stage",
                        lw=lw,
                    )
                elif name in ["m_prop", "throttle"]:
                    continue
                elif k == 2:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        label=f"{vehicle_names[j]} - Glide Stage",
                        lw=lw,
                    )
                else:
                    ax[i].plot(
                        data[f"Time_{k}"],
                        data[f"{name}_{k}"] * scale[name],
                        color=comp_colour[j],
                        linestyle="solid",
                        lw=lw,
                    )
          
    ts = data_set[vehicles[0]]["Time_2"][0]
    xs = data_set[vehicles[0]]["s_x_1"][valid].iloc[-1] / 1000
    for axi, name in zip(ax, traj_data_names):
        axi.set_ylabel(data_labels[name], fontsize=fs)
        axi.set_axisbelow(True)
        axi.grid(True)
        xmin, xmax = axi.get_xlim()
        axi.set_xlim(xmin, xmax)
        ymin, ymax = axi.get_ylim()
        axi.set_ylim(ymin, ymax)
        axi.tick_params(axis="both", labelsize=fs)
        for i in range(2):
            if limits[name][i]:
                if i == 0:
                    axi.hlines(
                        limits[name][i][0],
                        xmin,
                        ts,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )
                    axi.hlines(
                        limits[name][i][1],
                        xmin,
                        ts,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )
                else:
                    axi.hlines(
                        limits[name][i][0],
                        ts,
                        xmax,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )
                    axi.hlines(
                        limits[name][i][1],
                        ts,
                        xmax,
                        color="k",
                        linestyle="dashed",
                        lw=lw,
                        zorder=1,
                    )

    ax[-1].set_xlabel("Time (s)", fontsize=fs)
    # ax[0].legend(
    #     fontsize=fs, frameon=True, bbox_to_anchor=(1.6, 1.8), ncols=2, loc="upper right"
    # )

    fig.align_ylabels()

    # plt.tight_layout()

    fig.savefig(f"{dir}{mission}_Trajector_prezzo.png", dpi=1200)



def plotting(mission, vehicles, dir):
    data_set = load_data(mission, vehicles)

    vehicle_names = ""
    for i, vehicle in enumerate(vehicles):
        if i == 0:
            vehicle_names += vehicle
        else:
            vehicle_names += f"_&_{vehicle}"

    # for vehicle in vehicles:
    #     for data_name in boost_glide_names:
    #         plot_vs_time_all_stages(data_set[vehicle], data_set["stages"], data_name, mission, vehicle, data_set[f"{vehicle}_dir"])

    working_dir = dir
    base_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # plot_3D_tracjetory(data, stages, mission, vehicle, dir)
    # for data_name in boost_glide_names:
    # comparison(data_set, data_set["stages"], data_name, mission, vehicles, f"{working_dir}/")

    paper_plotting(data_set, vehicles, dir=f"{working_dir}/")
    # plot_3D_trajectory_animation(data_set, vehicles, dir=f"{working_dir}/")
    # prezzo_plotting(data_set, vehicles, dir=f"{working_dir}/")


if __name__ == "__main__":
    mission = "Target 2 Boost-Glide"
    vehicles = [
        "Target 2 Boost-Glide_K",
        # "Target 2 Glide_I with Boost_K",
        "Target 2 Boost-Glide_L",
    ]  # , "max_range_obstruction_glider_C"]
    plotting(
        mission,
        vehicles,
        dir="Mission_plan",
    )
