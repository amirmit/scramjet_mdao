import matplotlib.pyplot as plt
from cycler import cycler

_iter_hist = []
_multi_hist = {}  # DV name -> list of values
_geom_hist = {}  # nozzle geom key -> list of fraction-of-box values

_fig2 = None
_ax_multi = None
_ax_geom = None
_lines_multi = {}  # DV name -> Line2D
_lines_geom = {}  # nozzle geom key -> Line2D
_line_box_lo = None
_line_box_hi = None
_line_dv_box_lo = None
_line_dv_box_hi = None

# The nozzle geometry inputs to the surrogate (subset of nozzle_diag / BOUNDS)
_GEOM_KEYS = ["h5", "nozzle_length", "nozzle_angle", "nozzle_mid_frac"]


def init_converge_dvs_plot():
    global _fig2, _ax_multi, _ax_geom, _line_box_lo, _line_box_hi
    global _line_dv_box_lo, _line_dv_box_hi

    plt.ion()
    _fig2, (_ax_multi, _ax_geom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    _fig2.name = "convergence_dvs"

    _ax_multi.set_ylabel("DVs (fraction of [flb, fub])")
    _ax_geom.set_ylabel("Nozzle geom (fraction of surrogate box)")
    _ax_geom.set_xlabel("Iteration")

    # Set color + linestyle cycler for the DV axis (many more DVs than colors)
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", ":", "-."]
    prop_cycler = cycler(linestyle=linestyles) * cycler(color=colors)
    _ax_multi.set_prop_cycle(prop_cycler)
    _ax_geom.set_prop_cycle(cycler(color=colors))

    _line_dv_box_lo = _ax_multi.plot([], [], "k--", linewidth=0.65, label="bound")[0]
    _line_dv_box_hi = _ax_multi.plot([], [], "k--", linewidth=0.65)[0]
    _ax_multi.set_ylim(-0.05, 1.05)

    _line_box_lo = _ax_geom.plot([], [], "k--", linewidth=0.65, label="box edge")[0]
    _line_box_hi = _ax_geom.plot([], [], "k--", linewidth=0.65)[0]

    _ax_multi.grid(True)
    _ax_geom.grid(True)

    _fig2.subplots_adjust(right=0.78)


def _autoscale_to_data(ax, *value_lists):
    """Set y-limits from the real trajectory data only, ignoring the box-edge
    reference lines also drawn on this axis — so a bound only becomes
    visible on screen once the trajectory actually gets close to it.
    """
    vals = [v for lst in value_lists for v in lst if v is not None]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    if hi > lo:
        pad = 0.1 * (hi - lo)
    else:
        pad = abs(lo) * 0.1 if lo != 0 else 1.0
    ax.set_ylim(lo - pad, hi + pad)


def plot_converge_dvs(xdict, nozzle_diag, iteration, bounds, dv_bounds):
    """
    xdict: dict[str, float] — current DV values (as passed to objfunc)
    nozzle_diag: out4sens["nozzle_diag"] — has the raw physical values of
        h5, nozzle_length, nozzle_angle, nozzle_mid_frac
    iteration: int
    bounds: the surrogate's trained BOUNDS dict ({var: [lo, hi]}), used to
        normalise the geometry subplot to fraction-of-box
    dv_bounds: case.DV dict ({name: {"flb":.., "fub":.., ...}}), used to
        normalise the DVs subplot so every DV is plotted on the same
        0 (flb) - 1 (fub) scale regardless of its physical units/range
    """
    global _iter_hist, _multi_hist, _geom_hist
    global _fig2, _ax_multi, _ax_geom, _lines_multi, _lines_geom
    global _line_box_lo, _line_box_hi, _line_dv_box_lo, _line_dv_box_hi

    if _fig2 is None:
        init_converge_dvs_plot()

    _iter_hist.append(iteration)

    # --- DVs subplot (each DV scaled to fraction of its [flb, fub] bounds) ---
    for name, val in xdict.items():
        flb, fub = dv_bounds[name]["flb"], dv_bounds[name]["fub"]
        frac = (val - flb) / (fub - flb)

        if name not in _multi_hist:
            _multi_hist[name] = []
        _multi_hist[name].append(frac)

        if name not in _lines_multi:
            (line,) = _ax_multi.plot([], [], label=name)
            _lines_multi[name] = line

        _lines_multi[name].set_data(
            _iter_hist[: len(_multi_hist[name])], _multi_hist[name]
        )

    _line_dv_box_lo.set_data(_iter_hist, [0.0] * len(_iter_hist))
    _line_dv_box_hi.set_data(_iter_hist, [1.0] * len(_iter_hist))

    # Autoscale the x-axis (iteration count) only -- y stays fixed at the
    # [0, 1] fraction-of-bounds range set in init_converge_dvs_plot().
    _ax_multi.relim()
    _ax_multi.autoscale_view(scaley=False)

    handles, labels = _ax_multi.get_legend_handles_labels()
    _ax_multi.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    # --- Nozzle geometry fraction-of-box subplot ---
    for key in _GEOM_KEYS:
        lo, hi = bounds[key]
        frac = (nozzle_diag[key] - lo) / (hi - lo)

        if key not in _geom_hist:
            _geom_hist[key] = []
        _geom_hist[key].append(frac)

        if key not in _lines_geom:
            (line,) = _ax_geom.plot([], [], label=key)
            _lines_geom[key] = line

        _lines_geom[key].set_data(
            _iter_hist[: len(_geom_hist[key])], _geom_hist[key]
        )

    _line_box_lo.set_data(_iter_hist, [0.0] * len(_iter_hist))
    _line_box_hi.set_data(_iter_hist, [1.0] * len(_iter_hist))

    _autoscale_to_data(_ax_geom, *[_geom_hist[k] for k in _GEOM_KEYS])

    _ax_geom.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True
    )

    _fig2.canvas.draw()
    _fig2.canvas.flush_events()
