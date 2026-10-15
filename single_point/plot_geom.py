import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from hypervehicle.geom.path import Bezier


def plot_geom(geom, iteration_count, inlet=None, post_process=False):
    """

    :param geom: vhicle geometry class
    :param iteration_count: iteration number to be plotted
    :param inlet: vehicle inlet class for shock waves
    :param post_process: True for plotting a single iteration. False for live plotting during optimization
    :return:
    """

    N = 5

    x_body = np.empty(N + 5)
    z_body = np.empty(N + 5)

    x_cowl = np.empty(2)
    z_cowl = np.empty(2)

    x_intake = np.empty(2)
    z_intake = np.empty(2)

    x_outlet = np.empty(2)
    z_outlet = np.empty(2)

    x_shock = np.empty([2, 2])
    z_shock = np.empty([2, 2])

    #####################################################
    ##                 VEHICLE GEOMETRY                ##
    #####################################################

    # Vehicle cross-section (x-z plane).  +x toward nose (left), +z down.
    #
    #               e7            e6
    #         N0 ──────── N1 ────────────── T0
    #       e0 \                         ,/'
    #           I1                      T1    e5: Bezier C2─T2─T1─T0
    #  +x ←   e1 \                     /      T2: Bezier ctrl pt (see bez_length)
    #  +z ↓       C1 ─────────── C2 ─ T2
    #          e2 /            e4 │
    #            E1 ──────────── E2
    #                    e3
    #

    ##### Body #####
    x_body[0] = geom["N0"].x
    z_body[0] = geom["N0"].z

    x_body[1] = geom["I1"].x
    z_body[1] = geom["I1"].z

    x_body[2] = geom["C1"].x
    z_body[2] = geom["C1"].z

    Nozzle = Bezier([geom["C2"], geom["T2"], geom["T1"], geom["T0"]])

    for i in range(N + 1):
        x_body[i + 3] = Nozzle(i / N).x
        z_body[i + 3] = Nozzle(i / N).z

    x_body[N + 4] = geom["N0"].x
    z_body[N + 4] = geom["N0"].z

    ##### Cowl #####
    x_cowl[0] = geom["E1"].x
    z_cowl[0] = geom["E1"].z

    x_cowl[1] = geom["E2"].x
    z_cowl[1] = geom["E2"].z

    ##### Intake #####
    x_intake[0] = geom["C1"].x
    z_intake[0] = geom["C1"].z

    x_intake[1] = geom["E1"].x
    z_intake[1] = geom["E1"].z

    ##### Outlet #####
    x_outlet[0] = geom["C2"].x
    z_outlet[0] = geom["C2"].z

    x_outlet[1] = geom["E2"].x
    z_outlet[1] = geom["E2"].z

    if inlet is not None:
        x_shock, z_shock = calc_shocks(geom, inlet)

    # ================== PLOTTING PART ==================

    # ================= POST PROCESSING =================

    if post_process:
        # Stand-alone, single-iteration figure for post-processing
        fig, ax = plt.subplots(figsize=(13, 2))
        fig.name = "geom"

        ax.plot(-x_body, -z_body, "k", label=f"iteration {iteration_count}")
        ax.plot(-x_cowl, -z_cowl, "k")
        ax.plot(-x_intake, -z_intake, "k--")
        ax.plot(-x_outlet, -z_outlet, "k--")
        if inlet is not None:
            ax.plot(-x_shock.T, -z_shock.T, "r--")

        ax.set_aspect("equal", adjustable="box")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.85)
        ax.grid(True)
        ax.set_xlabel("(-)x m")
        ax.set_ylabel("(-)z m")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=1)

        # Optional: return fig/ax so caller can save/modify
        return fig, ax

    # ================= OPTIMIZATION =================

    # First iteration: create figure/axes and store the first geometry
    if iteration_count == 1 or not hasattr(plot_geom, "_fig"):
        plot_geom._fig, plot_geom._ax = plt.subplots(figsize=(13, 2))
        plot_geom._fig.name = "geom"
        fig, ax = plot_geom._fig, plot_geom._ax

        # Plot first iteration in one style
        ax.plot(-x_body, -z_body, color="0.7", label="iteration 1")
        ax.plot(-x_cowl, -z_cowl, color="0.7")
        ax.plot(-x_intake, -z_intake, "--", color="0.7")
        ax.plot(-x_outlet, -z_outlet, "--", color="0.7")

        if inlet is not None:
            ax.plot(-x_shock.T, -z_shock.T, "r--", alpha=0.4)

        ax.set_aspect("equal", adjustable="box")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.85)
        ax.grid(True)
        ax.set_xlabel("(-)x m")
        ax.set_ylabel("(-)z m")

        # Legend above axes, single row
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=True,
        )

        plt.draw()
        plt.pause(0.001)  # if running in a loop

        # Store first-iteration geometry if you want to reuse raw data later
        plot_geom._first = {
            "x_body": x_body.copy(),
            "z_body": z_body.copy(),
            "x_cowl": x_cowl.copy(),
            "z_cowl": z_cowl.copy(),
            "x_intake": x_intake.copy(),
            "z_intake": z_intake.copy(),
            "x_outlet": x_outlet.copy(),
            "z_outlet": z_outlet.copy(),
            "x_shock": x_shock.copy(),
            "z_shock": z_shock.copy(),
            "has_shock": inlet is not None,
        }

    else:
        # Subsequent iterations: overplot on same axes
        fig = plot_geom._fig
        ax = plot_geom._ax

        # Clear only previous "current iteration" lines, keep first iteration.
        # Simpler: clear axis and replot first + current.
        ax.clear()

        # Replot first iteration from stored data
        f = plot_geom._first
        ax.plot(-f["x_body"], -f["z_body"], color="0.7", label="iteration 1")
        ax.plot(-f["x_cowl"], -f["z_cowl"], color="0.7")
        ax.plot(-f["x_intake"], -f["z_intake"], "--", color="0.7")
        ax.plot(-f["x_outlet"], -f["z_outlet"], "--", color="0.7")
        if f["has_shock"]:
            ax.plot(-f["x_shock"].T, -f["z_shock"].T, "r--", alpha=0.4)

        # Plot current iteration in another color
        ax.plot(-x_body, -z_body, "k", label=f"iteration {iteration_count}")
        ax.plot(-x_cowl, -z_cowl, "k")
        ax.plot(-x_intake, -z_intake, "k--")
        ax.plot(-x_outlet, -z_outlet, "k--")
        if inlet is not None:
            ax.plot(-x_shock.T, -z_shock.T, "r--")

        ax.set_aspect("equal", adjustable="box")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.85)
        ax.grid(True)
        ax.set_xlabel("(-)x m")
        ax.set_ylabel("(-)z m")

        # Legend above axes, single line with two entries
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=2,
            frameon=True,
        )

        fig.canvas.draw_idle()
        plt.pause(0.001)


def calc_shocks(geom, inlet):
    x_shock = np.empty([2, 2])
    z_shock = np.empty([2, 2])

    x_shock[0][0] = geom["N0"].x
    z_shock[0][0] = geom["N0"].z

    x_shock[1][0] = geom["I1"].x
    z_shock[1][0] = geom["I1"].z

    m0 = -np.tan(inlet.betas[0])
    m1 = -np.tan(inlet._inlet_geom.theta[0] + inlet.betas[1])

    if inlet.InletMode == 1:  # Both shock spill
        x_shock[0][1], z_shock[0][1] = find_end_of_shock(
            geom, x_shock[0][0], z_shock[0][0], m0
        )
        x_shock[1][1], z_shock[1][1] = find_end_of_shock(
            geom, x_shock[1][0], z_shock[1][0], m1
        )

    elif inlet.InletMode == 2:  # One shock ingested
        x_shock[0][1], z_shock[0][1] = find_end_of_shock(
            geom, x_shock[0][0], z_shock[0][0], m0
        )
        x_shock[1][1], z_shock[1][1] = shock_intake_intersection(
            geom["C1"], geom["E1"], x_shock[1][0], z_shock[1][0], m1
        )

    elif inlet.InletMode == 3:  # Two shocks ingested
        x_shock[0][1], z_shock[0][1] = shock_intake_intersection(
            geom["C1"], geom["E1"], x_shock[0][0], z_shock[0][0], m0
        )
        x_shock[1][1], z_shock[1][1] = shock_intake_intersection(
            geom["C1"], geom["E1"], x_shock[1][0], z_shock[1][0], m1
        )

    elif inlet.InletMode == 4:  # Shock-shock interaction
        x_shock[0][1], z_shock[0][1] = shock_intake_intersection(
            geom["C1"], geom["E1"], x_shock[0][0], z_shock[0][0], m0
        )
        x_shock[1][1], z_shock[1][1] = shock_shock_intersection(
            x_shock[0][0], z_shock[0][0], m0, x_shock[1][0], z_shock[1][0], m1
        )

    return x_shock, z_shock


def find_end_of_shock(geom, x_0, z_0, m):
    x_1 = geom["C1"].x
    z_1 = z_0 + m * (x_1 - x_0)
    return x_1, z_1


def shock_intake_intersection(p1, p2, x3, z3, m1):
    """
    Find intersection of:
    Line 1: through p1=(x1,z1) and p2=(x2,z2)
    Line 2: through (x3,z3) with m1 = tan(angle)
    """
    x1, z1 = p1.x, p1.z
    x2, z2 = p2.x, p2.z

    # slope of line 1
    if np.isclose(x2 - x1, 0):
        m0 = 1 + 1e10  # vertical line
    else:
        m0 = (z2 - z1) / (x2 - x1)

    # Handle vertical line (m1 = inf)
    if m0 > 1e10:
        x_int = x1
        z_int = m1 * (x_int - x3) + z3
        return x_int, z_int

    # Standard intersection formula for two non-vertical lines
    x_int = (m0 * x1 - z1 - m1 * x3 + z3) / (m0 - m1)
    z_int = m0 * (x_int - x1) + z1

    return x_int, z_int


def shock_shock_intersection(x1, z1, m0, x3, z3, m1):
    x_int = (m0 * x1 - z1 - m1 * x3 + z3) / (m0 - m1)
    z_int = m0 * (x_int - x1) + z1
    return x_int, z_int
