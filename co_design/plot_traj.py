import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import pandas as pd
from hyperdynamics.utils import Result, unr

eps_val=np.inf

def plot_from_result(
    result: Result,
    system,
    title: str = "",
    save: bool = True,
    show: bool = False,
    axs=None,
    fname="trajectory.png",
    fname_xy="x-y_Position.png",
    fname_z="altitude_vs.png", 
    f_add_name="",
):
    # Create parameters array
    param_arr = np.fromiter(system.parameters.values(), dtype=float)

    # Get plotting data
    plotting_attr = result.plotting_attr

    if axs is None:
        # Create axes
        fig, axs = plt.subplots(7, 1, sharex=True, figsize=(8, 6))
        axs = axs.flatten()

    lw = 1  # linewidth
    ax_x = axs[0]
    ax_z = axs[1]
    ax_v = axs[2]
    ax_ang = axs[3]
    ax_u = axs[4]
    ax_q = axs[5]
    ax_prop = axs[6]
    ax_y = plt.twinx(ax_x)
    ax_M = plt.twinx(ax_v)
    ax_qalpha = plt.twinx(ax_q)

    fig_xy, ax_xy = plt.subplots()
    fig_zt, ax_zt = plt.subplots()
    fig_zx, ax_zx = plt.subplots()
    fig_xyz = plt.figure()
    ax_xyz = fig_xyz.add_subplot(projection='3d')

    fig_xy, ax_xy = plt.subplots()
    fig_zt, ax_zt = plt.subplots()
    fig_zx, ax_zx = plt.subplots()

    for i in range(system.stages):
        x_c = plotting_attr["x_c"][i]
        u_c = plotting_attr["u_c"][i]
        tx_c = plotting_attr["tx_c"][i]
        tu_c = plotting_attr["tu_c"][i]
        x_int = plotting_attr["x_int"][i]
        u_int = plotting_attr["u_int"][i]
        t_int = plotting_attr["t_int"][i]

        q_int = system.fun["f_q"](x_int[system.x_dict["s_z"]], x_int[system.x_dict["v"]], eps_val=eps_val)
        mach_int = x_int[system.x_dict["v"]] / system.fun["f_a"](
            x_int[system.x_dict["s_z"]], eps_val=eps_val,
        )
        q_alpha_int = q_int * np.rad2deg(u_int[0])
        rho = system.fun["f_rho"](x_int[system.x_dict["s_z"]])
        heat_int = system.fun["thermal_fun"]["HV"](rho, x_int[system.x_dict["v"]], *param_arr)

        # Create casadi function
        # xdot_fn = ca.Function(
        #     "f",
        #     [system.x, system.u, system.p, system.t],
        #     [system.xdot],
        #     ["x", "u", "p", "t"],
        #     ["xdot"],
        # )
        # xdot_int = xdot_fn(x=x_int, u=u_int, p=param_arr, t=np.reshape(t_int, (1, np.size(t_int))))["xdot"]
        # xdot_int = np.array(xdot_int)

        # Limits
        # n_s = [0, *result.col_params.stages, result.col_params.n_c]

        # Stage Seperation
        # stage = np.array(
        #     [[*tx_c[result.col_params.stages], 1], [*tx_c[result.col_params.stages], 1]]
        # )
        # Plot trajectory
        (l1,) = ax_x.plot(t_int, x_int[0] / 1e3, color="b", label="x", lw=lw)
        ax_x.plot(tx_c, x_c[0] / 1e3, ".", color="b", markerfacecolor="none")
        ax_x.set_ylabel(r"$s_x [km]$")

        (l2,) = ax_y.plot(t_int, x_int[1] / 1e3, color="r", label="y", lw=lw)
        ax_y.plot(tx_c, x_c[1] / 1e3, ".", color="r", markerfacecolor="none")
        ax_y.set_ylabel(r"$s_y [km]$")
        ax_x.legend([l1, l2], ["x", "y"], fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.01, -0.1))

        (l1,) = ax_z.plot(t_int, x_int[2] / 1e3, color="b", label="z", lw=lw)
        ax_z.plot(tx_c, x_c[2] / 1e3, ".", color="b", markerfacecolor="none")
        ax_z.set_ylabel(r"$s_z [km]$")

        # Plot velocity
        (l1,) = ax_v.plot(t_int, x_int[3], color="b", label=r"$v$", lw=lw)
        ax_v.plot(tx_c, x_c[3], ".", color="b", markerfacecolor="none")
        ax_v.set_ylabel(r"$v$ [m/s]")
        (l2,) = ax_M.plot(t_int, mach_int, color="r", label="y", lw=lw)
        ax_M.set_ylabel("Mach")
        ax_v.legend([l1, l2], [r"v", "M"], fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.01, 1.1))

        # Plot flight angle(s)
        (l1,) = ax_ang.plot(
            t_int, unr(np.rad2deg(x_int[4])), color="b", label=r"$\chi$", lw=lw
        )
        (l2,) = ax_ang.plot(
            t_int, unr(np.rad2deg(x_int[5])), color="r", label=r"$\gamma$", lw=lw
        )
        ax_ang.plot(tx_c, unr(np.rad2deg(x_c[4])), ".", color="b", markerfacecolor="none")
        ax_ang.plot(tx_c, unr(np.rad2deg(x_c[5])), ".", color="r", markerfacecolor="none")
        ax_ang.set_ylabel(r"[deg]")
        ax_ang.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.01, 1.1))

        # Plot control input(s)
        ax_u.plot(t_int, np.rad2deg(u_int[0]), color="b", label=r"$\alpha$", linewidth=lw)
        ax_u.plot(tu_c, np.rad2deg(u_c[0]), ".", color="b", markerfacecolor="none")
        ax_u.plot(t_int, np.rad2deg(u_int[1]), color="g", label=r"$\phi$", linewidth=lw)
        ax_u.plot(tu_c, np.rad2deg(u_c[1]), ".", color="g", markerfacecolor="none")
        ax_u.set_ylabel(r"$[deg]$")
        ax_u.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.01, 1.1))

        (l1, ) = ax_q.plot(t_int, q_int / 1e3, color="purple", linewidth=lw, label=r"$q$")
        (l2,) = ax_qalpha.plot(t_int, q_alpha_int/1e3, color="r", linewidth=lw, label=r"$q\alpha$")
        ax_q.set_ylabel("q [kPa]")
        ax_qalpha.set_ylabel(r"$q\alpha$")
        ax_q.legend([l1, l2], [r"$q$", r"$q\alpha$"], fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.01, -0.1))

        # Plot Fuel and Prop Mass
        if i == 0 and len(u_int) == 3:
            ax_prop.plot(t_int, u_int[2], label="Throttle", linewidth=lw)
            ax_prop.plot(tu_c, u_c[2], ".", markerfacecolor="none")
        # ax_prop.plot(t_int, x_int[6], color="b", label="m_prop", lw=lw)
        # ax_prop.plot(tx_c, x_c[6], ".", color="b", markerfacecolor="none")
        # ax_prop.set_ylabel("Prop Mass [kg]")
        #
        # Plot limits
        # q_ulim = [100, 100]
        # for i in range(len(result.col_params.stages)+1):
        #     x_llim = [x * s for x, s in zip(result.col_params.x_llim[i], result.col_params.x_scale)]
        #     x_ulim = [x * s for x, s in zip(result.col_params.x_ulim[i], result.col_params.x_scale)]
        #
        #     ax_z.plot([tx_c[n_s[i]], tx_c[n_s[i+1]]], [x_ulim[2]/1e3, x_ulim[2]/1e3], "--", color="k")
        #     ax_z.plot([tx_c[n_s[i]], tx_c[n_s[i+1]]], [x_llim[2]/1e3, x_llim[2]/1e3], "--", color="k")
        #     # ax_v.plot([t_c[n_s[i]], t_c[n_s[i+1]]], [x_ulim[3], x_ulim[3]], "--", color="k")
        #     ax_v.plot([tx_c[n_s[i]], tx_c[n_s[i+1]]], [x_llim[3], x_llim[3]], "--", color="k")
        #     ax_q.plot([tx_c[n_s[i]], tx_c[n_s[i+1]]], [q_ulim[i], q_ulim[i]], "--", color="k")
        #     ax_q.plot([tx_c[n_s[i]], tx_c[n_s[i+1]]], [1, 1], "--", color="k")
        #
        # for i, ax in enumerate(axs):
        #     y = ax.get_ylim()
        #     ax.plot(stage[:, :-1], stage[:, -1] * y, "--" ,color="r",)
            

        axs[-1].set_xlabel("Time [s]")

        try:
            fig.suptitle(title)
        except:
            pass
        # plt.tight_layout()

        # Additional Plots
        fig_xy, ax_xy = plot_xy_fun(x_int[0]/1e3, x_int[1]/1e3, fig_xy=fig_xy, ax_xy=ax_xy)
        fig_zt, ax_zt, fig_zx, ax_zx = plot_altitude(x_int, t_int, fig_zt=fig_zt, ax_zt=ax_zt, fig_zx=fig_zx, ax_zx=ax_zx)
        fig_xyz, ax_xyz = plot_xyz_fun(x_int[0]/1e3, x_int[1]/1e3, x_int[2]/1e3, fig_xyz=fig_xyz, ax_xyz=ax_xyz)
                

    if save:
        fig.savefig(fname, dpi=900)
        fig_xy.savefig(f"x-y_position_{f_add_name}.png", dpi=900, bbox_inches="tight")
        fig_zt.savefig(f"altitude_vs_time_{f_add_name}.png")
        fig_zx.savefig(f"altitude_vs_x_{f_add_name}.png")
        fig_xyz.savefig(f"3d_trajectory_{f_add_name}.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_xy)
        plt.close(fig_zt)
        plt.close(fig_zx)
        plt.close(fig_xyz)
    return axs


def plot_xy_fun(x_int, y_int, fig_xy, ax_xy):
    mission = "target2" # target
    if mission == "target":
        ob1 = plt.Circle((500, 0), 150, color="r", alpha=0.5, label="Obstructions")
        ob2 = plt.Circle((1000, 100), 150, color="r", alpha=0.5)
        ob3 = plt.Circle((700, 250), 100, color="r", alpha=0.5)
        ob4 = plt.Circle((350, -100), 100, color="r", alpha=0.5)
    elif mission == "obstruction":
        ob1 = plt.Circle((700, 225), 300, color="r", alpha=0.5, label="Obstructions")
        ob2 = plt.Circle((1200, -225), 300, color="r", alpha=0.5)
        ob3 = plt.Circle((1700, 225), 300, color="r", alpha=0.5)
        ob4 = plt.Circle((2200, -225), 300, color="r", alpha=0.5)
        ob5 = plt.Circle((2700, 225), 300, color="r", alpha=0.5)
        ob6 = plt.Circle((3200, -225), 300, color="r", alpha=0.5)
        ob7 = plt.Circle((3700, 225), 300, color="r", alpha=0.5)

    x_tar = 1200
    y_tar = 0

    # ax_xy.plot(x_int, y_int, label="Trajectory")
    ax_xy.plot(0, 0, ".", color="b", markerfacecolor="none", label="Start")
    if mission == "target":
        ax_xy.add_patch(ob1)
        ax_xy.add_patch(ob2)
        ax_xy.add_patch(ob3)
        ax_xy.add_patch(ob4)
        ax_xy.plot(x_tar, y_tar, "x", color="k", label="Target")
        ax_xy.set_ylim([-100, 300])
    elif mission == "obstruction":
        ax_xy.add_patch(ob1)
        ax_xy.add_patch(ob2)
        ax_xy.add_patch(ob3)
        ax_xy.add_patch(ob4)
        ax_xy.add_patch(ob4)
        ax_xy.add_patch(ob5)
        ax_xy.add_patch(ob6)
        ax_xy.add_patch(ob7)
        ax_xy.set_ylim([-200, 200])
    elif mission == "target2":
        ax_xy.plot(500, 0, ".", color="g", label="1st way-point")
        ax_xy.plot(1000, 300, ".", color="k", label="2nd way-point")
        ax_xy.plot(1000, 0, "X", color="r", label="Target")
    # ax_xy.set_xlim([-100, 1800])
    # ax_xy.set_ylim([-0.5*x_tar*1.2, 0.5*x_tar*1.2])
    # # ax_xy.set_ylim([-200, 200])
    # ax_xy.set_title(title+"xy")
    ax_xy.set_xlabel(r"$s_x$ [km]")
    ax_xy.set_ylabel(r"$s_y$ [km]")
    # ax_xy.axis("scaled")
    ax_xy.set_aspect(1)
    ax_xy.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return fig_xy, ax_xy

def obstruction_for_plot(x_c, y_c, r):
    theta = np.linspace(0, 2*np.pi, 100)
    x_ob = x_c + r * np.cos(theta)
    y_ob = y_c + r * np.sin(theta)

    return x_ob, y_ob


def plot_altitude(x_int, t_int, fig_zt, ax_zt, fig_zx, ax_zx):
    ax_zt.plot(t_int, x_int[2]/1e3)
    ax_zt.set_aspect(5)
    ax_zt.set_ylabel(r"$s_z$ [km]")
    ax_zt.set_xlabel(r"Time [s]")

    ax_zx.plot(x_int[0]/1e3, x_int[2]/1e3)
    ax_zx.set_aspect(8)
    ax_zx.set_ylabel(r"$s_z$ [km]")
    ax_zx.set_xlabel(r"$s_x$ [km]")

    return fig_zt, ax_zt, fig_zx, ax_zx


def plot_xyz_fun(x_int, y_int, z_int, fig_xyz, ax_xyz):
    ax_xyz.plot(x_int, y_int, z_int)
    ax_xyz.set_xlabel(r"$s_x$ [km]")
    ax_xyz.set_ylabel(r"$s_y$ [km]")
    ax_xyz.set_zlabel(r"$s_z$ [km]")

    return fig_xyz, ax_xyz

def result2csv(system, result, fname=""):
    """
    Write Trajectory Results to csv
    """
    data_dict = {}
    for i in range(result.stages):
        x_int = result.plotting_attr["x_int"][i]
        u_int = result.plotting_attr["u_int"][i]
        q_int = np.array(
            system.fun["f_q"](x_int[system.x_dict["s_z"]], x_int[system.x_dict["v"]], eps_val=eps_val)
        ).flatten()
        mach_int = np.array(
            x_int[system.x_dict["v"]] / system.fun["f_a"](x_int[system.x_dict["s_z"]], eps_val=eps_val)
        ).flatten()
        rho = system.fun["f_rho"](x_int[system.x_dict["s_z"]]).flatten()
        heat_int = np.array(system.fun["thermal_fun"]["HV"](rho, x_int[system.x_dict["v"]], *system.parameters.values())).flatten()

        data_dict[f"Time_{i}"] = result.plotting_attr["t_int"][i]

        for state, idx in system.x_dict.items():
            data_dict[f"{state}_{i}"] = x_int[idx]

        for input, idx in system.u_dict[i].items():
            data_dict[f"{input}_{i}"] = u_int[idx]

        data_dict[f"q_{i}"] = q_int
        data_dict[f"mach_{i}"] = mach_int
        data_dict[f"heat_flux_{i}"] = heat_int
        data_dict[f"rho_{i}"] = rho

    if fname == "":
        fname = "trajectory.csv"
    else:
        if fname.split(".")[-1] == "csv":
            pass
        else:
            fname = fname + ".csv"

    # df = pd.DataFrame(data_dict)
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
    df.to_csv(fname)



if __name__ == "__main__":
    fig_xy, ax_xy = plt.subplots()
    fig_xy, ax_xy = plot_xy_fun(0, 0, fig_xy, ax_xy)
    fig_xy.savefig(f"Target 2 Mission.png", dpi=900, bbox_inches="tight")
