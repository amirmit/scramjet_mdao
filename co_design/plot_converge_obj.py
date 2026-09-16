import matplotlib.pyplot as plt

_iter_hist = []
_cd_hist = []
_opt_hist = []
_feas_hist = []

_fig = None
_ax_cd = None
_ax_metrics = None
_line_cd = None
_line_opt = None
_line_feas = None


def _autoscale_log_to_data(ax, *value_lists):
    """Set y-limits (log scale) from the real trajectory data, with a
    proportional margin so points near the extremes (e.g. feasibility
    dropping to ~1e-6) aren't clipped right at the axis edge. Values <= 0
    can't be shown on a log scale and are ignored.

    Returns the (lo, hi) limits actually set, or None if there was no
    positive data to scale to.
    """
    vals = [v for lst in value_lists for v in lst if v is not None and v > 0]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = (hi / lo) ** 0.1 if hi > lo else 10.0
    lo_lim, hi_lim = lo / pad, hi * pad
    ax.set_ylim(lo_lim, hi_lim)
    return lo_lim, hi_lim


def _clamp_for_log(values, floor):
    """SNOPT reports exact 0.0 at some iterates (e.g. a fully-feasible
    restart point). A log axis can't display 0, so left as-is these points
    make the line shoot straight off the bottom of the panel past the
    computed y-limits. Clamp them to the axis floor instead so every plotted
    point stays inside the visible range.
    """
    return [v if (v is not None and v > 0) else floor for v in values]


def init_converge_obj_plot(obj_label="Objective"):
    global _fig, _ax_cd, _ax_metrics, _line_cd, _line_opt, _line_feas

    plt.ion()
    _fig, (_ax_cd, _ax_metrics) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    _fig.name = "convergence_obj"

    _ax_cd.set_ylabel(obj_label)
    _ax_metrics.set_ylabel("SNOPT metrics")
    _ax_metrics.set_yscale("log")
    _ax_metrics.set_xlabel("Major iteration")

    (_line_cd,) = _ax_cd.plot([], [], "-o", label=obj_label)
    (_line_opt,) = _ax_metrics.plot([], [], "-o", label="optimality")
    (_line_feas,) = _ax_metrics.plot([], [], "-s", label="feasibility")

    _ax_cd.grid(True)
    _ax_metrics.grid(True, which="both")
    _ax_cd.legend(loc="upper right")
    _ax_metrics.legend(loc="upper right")

    _fig.tight_layout()
    _fig.show()


def plot_converge_obj(nMajor, obj_value, optimality, feasibility, obj_label="Objective"):
    """
    Driven by SNOPT's snSTOP callback (see Opt_Main.py), so this fires once
    per major iteration — not once per objfunc call like the other live
    plots.

    nMajor: major iteration counter (from the snSTOP iterDict)
    obj_value: objective value at this major iterate
    optimality: SNOPT dual infeasibility (iterDict["optimality"])
    feasibility: SNOPT primal infeasibility (iterDict["feasibility"])
    obj_label: name of the active objective (case config's objective_name,
        or its single term's name) -- only used the first time the figure
        is created, since the axis label is set once at init.
    """
    global _iter_hist, _cd_hist, _opt_hist, _feas_hist
    global _fig, _ax_cd, _ax_metrics, _line_cd, _line_opt, _line_feas

    if _fig is None:
        init_converge_obj_plot(obj_label)

    _iter_hist.append(nMajor)
    _cd_hist.append(obj_value)
    _opt_hist.append(optimality)
    _feas_hist.append(feasibility)

    _line_cd.set_data(_iter_hist, _cd_hist)

    limits = _autoscale_log_to_data(_ax_metrics, _opt_hist, _feas_hist)
    floor = limits[0] if limits is not None else 1e-300
    _line_opt.set_data(_iter_hist, _clamp_for_log(_opt_hist, floor))
    _line_feas.set_data(_iter_hist, _clamp_for_log(_feas_hist, floor))

    _ax_cd.relim()
    _ax_cd.autoscale_view()

    _fig.canvas.draw()
    _fig.canvas.flush_events()
