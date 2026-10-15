import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_aero_deck(aero_inputs: dict[str, list[float]], aero_sm, df: pd.DataFrame):

    mach_range = np.array(aero_inputs["mach"])
    aoa_range = np.array(aero_inputs["aoa"])

    mach_range_fine = np.linspace(
        np.min(aero_inputs["mach"]), np.max(aero_inputs["mach"]), 2*len(mach_range)-1
    )
    aoa_range_fine = np.linspace(
        np.min(aero_inputs["aoa"]), np.max(aero_inputs["aoa"]), 2*len(aoa_range)-1
    )

    # Save aerodeck figs in directory to check model is accurate (e.g. not overfitting)
    aoa_mesh_fine, mach_mesh_fine = np.meshgrid(aoa_range_fine, mach_range_fine)
    z_mesh_fine = aero_sm.eval_grid((aoa_mesh_fine, mach_mesh_fine))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
    ax.scatter(df['aoa'],
               df['mach'],
               df['CD'],
               color='red',
               label='Sampled data'
               )
    ax.plot_wireframe(aoa_mesh_fine, mach_mesh_fine,
                      z_mesh_fine[0],
                      color='black', label='Model'
                      )
    ax.legend()
    ax.set_xlabel('AoA [deg]')
    ax.set_ylabel('Mach')
    ax.set_zlabel(r'$C_D$')
    plt.savefig("Surrogate_model_drag.png")
    plt.close(fig)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
    ax.scatter(df['aoa'],
               df['mach'],
               df['CL'],
               color='red',
               label='Sampled data'
               )
    ax.plot_wireframe(aoa_mesh_fine, mach_mesh_fine,
                      z_mesh_fine[1],
                      color='black', label='Model'
                      )
    ax.legend()
    ax.set_xlabel('AoA [deg]')
    ax.set_ylabel('Mach')
    ax.set_zlabel(r'$C_L$')
    plt.savefig("Surrogate_model_lift.png")
    plt.close(fig)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
    ax.scatter(df['aoa'],
               df['mach'],
               df['CM'],
               color='red',
               label='Sampled data'
               )
    ax.plot_wireframe(aoa_mesh_fine, mach_mesh_fine,
                      z_mesh_fine[2],
                      color='black', label='Model'
                      )
    ax.legend()
    ax.set_xlabel('AoA [deg]')
    ax.set_ylabel('Mach')
    ax.set_zlabel(r'$C_M$')
    plt.savefig("Surrogate_model_moment.png")
    plt.close(fig)

    # Andy's original plots
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    ax_d = ax[0]
    ax_l = ax[1]
    ax_m = ax[2]

    for mach in mach_range:
        mach_ = np.full(len(aoa_range), mach)
        pm_inputs = np.array([aoa_range, mach_]).T

        drag_new = aero_sm.eval(pm_inputs)[:, 0]
        lift_new = aero_sm.eval(pm_inputs)[:, 1]
        m_new = aero_sm.eval(pm_inputs)[:, 2]

        (l1,) = ax_d.plot(
            aoa_range,
            drag_new,
            "-",
            linewidth=1,
            markerfacecolor="none",
            markersize=5,
            label="Ma " + str(mach),
        )
        (l2,) = ax_l.plot(
            aoa_range,
            lift_new,
            "-",
            linewidth=1,
            color=l1.get_color(),
            markerfacecolor="none",
            markersize=5,
        )

        (l3,) = ax_m.plot(
            aoa_range,
            m_new,
            "-",
            linewidth=1,
            color=l1.get_color(),
            markerfacecolor="none",
            markersize=5,
        )

    (l1,) = ax_d.plot(
        df["aoa"],
        df["CD"],
        "s",
        linewidth=1,
        markerfacecolor="none",
        markersize=5,
        color="k",
        label="Input data",
    )
    (l2,) = ax_l.plot(
        df["aoa"],
        df["CD"],
        "s",
        linewidth=1,
        markerfacecolor="none",
        markersize=5,
        color="k",
    )
    (l3,) = ax_m.plot(
        df["aoa"],
        df["CM"],
        "s",
        linewidth=1,
        markerfacecolor="none",
        markersize=5,
        color="k",
    )

    ax_d.plot(
        [aoa_range[0], aoa_range[-1]], [0, 0], "--", linewidth=0.75, color="k"
    )

    ax_d.set_ylabel(r"$C_d$")
    ax_l.set_ylabel(r"$C_l$")
    ax_m.set_ylabel(r"$C_m$")
    ax[-1].set_xlabel("AoA")
    ax[0].legend()
    plt.tight_layout()

    fig.savefig("Surrogate-model-AoA.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    ax_d = ax[0]
    ax_l = ax[1]
    ax_m = ax[2]

    for aoa in aoa_range:
        aoa_ = np.full(len(mach_range), aoa)
        pm_inputs = np.array([aoa_, mach_range]).T

        drag_new = aero_sm.eval(pm_inputs)[:, 0]
        lift_new = aero_sm.eval(pm_inputs)[:, 1]
        m_new = aero_sm.eval(pm_inputs)[:, 2]

        (l1,) = ax_d.plot(
            mach_range,
            drag_new,
            "-",
            linewidth=1,
            markerfacecolor="none",
            markersize=5,
            label="AoA " + str(aoa),
        )
        (l2,) = ax_l.plot(
            mach_range,
            lift_new,
            "-",
            linewidth=1,
            color=l1.get_color(),
            markerfacecolor="none",
            markersize=5,
        )
        (l3,) = ax_m.plot(
            mach_range,
            m_new,
            "-",
            linewidth=1,
            color=l1.get_color(),
            markerfacecolor="none",
            markersize=5,
        )

    (l1,) = ax_d.plot(
        df["mach"],
        df["CD"],
        "s",
        linewidth=1,
        markerfacecolor="none",
        markersize=5,
        color="k",
        label="Input data",
    )
    (l2,) = ax_l.plot(
        df["mach"],
        df["CL"],
        "s",
        linewidth=1,
        markerfacecolor="none",
        markersize=5,
        color="k",
    )
    (l3,) = ax_m.plot(
        df["mach"],
        df["CM"],
        "s",
        linewidth=1,
        markerfacecolor="none",
        markersize=5,
        color="k",
    )

    ax_d.set_ylabel(r"$C_d$")
    ax_l.set_ylabel(r"$C_l$")
    ax_m.set_ylabel(r"$C_m$")
    ax[-1].set_xlabel("Mach")
    ax[0].legend()
    plt.tight_layout()

    fig.savefig("Surrogate-model-mach.png", dpi=300)
    plt.close(fig)
