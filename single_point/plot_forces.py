import numpy as np
import matplotlib.pyplot as plt
from utilities import Vector3

_force_hist = {}  # name -> dict with 'X' or 'Z'
_iter_hist = []
_P_exit_ratio_hist = []
_P_exit_ratio_limit_hist = []

_figF = None
_axX = None
_axZ = None
_axP = None


def init_force_fraction_plot():
    global _figF, _axX, _axZ, _axP
    plt.ion()
    _figF, (_axX, _axZ, _axP) = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    _figF.name = "forces"
    _axP.set_xlabel("Iteration")

    # Fixed right margin for the axis-anchored legends below. Using a
    # manual margin (rather than layout="constrained") avoids constrained
    # layout fighting over space with the bbox_to_anchor-positioned legends
    # — mixing the two caused the axes to shrink and the legends to drift.
    _figF.subplots_adjust(right=0.72)


def _body_to_wind(Fx, Fz, alpha_rad):
    D = -Fx * np.cos(alpha_rad) - Fz * np.sin(alpha_rad)
    L = Fx * np.sin(alpha_rad) - Fz * np.cos(alpha_rad)
    return D, L


def _autoscale_to_data(ax, *value_lists):
    """Set y-limits from the real trajectory data only, ignoring the limit
    reference line also drawn on this axis — so the limit only becomes
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


def plot_forces(
    forces_dict,
    iteration,
    alpha=0.0,
    frame_flag="Body",
    P_exit_ratio=None,
    P_exit_ratio_limit=None,
):
    """
    forces_dict: {component_name: (Fx, Fz)} in body axes
    alpha: angle of attach [rad]
    frame_flag: "Body" or "Wind"
    P_exit_ratio: nozzle exit-plane pressure ratio to freestream (the
        P_exit_ratio constraint value)
    P_exit_ratio_limit: the constraint's lower bound, marked as a reference
        line on the third subplot
    """
    global _figF, _axX, _axZ, _axP, _force_hist, _iter_hist
    global _P_exit_ratio_hist, _P_exit_ratio_limit_hist

    if _figF is None:
        init_force_fraction_plot()

    if iteration not in _iter_hist:
        _iter_hist.append(iteration)

    # prepare storage per component per iteration
    for name in forces_dict.keys():
        if name not in _force_hist:
            _force_hist[name] = {"X": [], "Z": []}

    # compute total X/Z (or D/L) for this iteration
    X_tot = 0.0
    Z_tot = 0.0

    for name, F in forces_dict.items():
        (Fx, Fz) = (F.x, F.z) if isinstance(F, Vector3) else (F[0], F[2])
        if frame_flag == "Wind":
            D, L = _body_to_wind(Fx, Fz, alpha)
            X_val, Z_val = D, L  # X→D, Z→L
        else:
            X_val, Z_val = Fx, Fz  # use body axes directly

        _force_hist[name]["X"].append(X_val)
        _force_hist[name]["Z"].append(Z_val)

        X_tot += X_val
        Z_tot += Z_val

    # compute fractions for this iteration
    # protect against divide-by-zero
    eps = 1e-12
    X_tot = X_tot if abs(X_tot) > eps else eps
    Z_tot = Z_tot if abs(Z_tot) > eps else eps

    # clear axes and replot fractions
    _axX.clear()
    _axZ.clear()

    for name, comp in _force_hist.items():
        X_arr = np.array(comp["X"])
        Z_arr = np.array(comp["Z"])

        X_frac = X_arr / X_tot
        Z_frac = Z_arr / Z_tot

        _axX.plot(_iter_hist, X_frac, label=name)
        _axZ.plot(_iter_hist, Z_frac, label=name)

    if frame_flag == "Wind":
        _axX.set_ylabel("D fraction")
        _axZ.set_ylabel("L fraction")
        _axX.set_title(f"D = {X_tot/1e3:.1f} kN")
        _axZ.set_title(f"L = {Z_tot/1e3:.1f} kN")
    else:
        _axX.set_ylabel("Fx fraction")
        _axZ.set_ylabel("Fz fraction")
        _axX.set_title(f"Fx = {X_tot/1e3:.1f} kN")
        _axZ.set_title(f"Fz = {Z_tot/1e3:.1f} kN")

    _axX.grid(True)
    _axZ.grid(True)

    # --- P_exit_ratio subplot ---
    _P_exit_ratio_hist.append(P_exit_ratio)
    _P_exit_ratio_limit_hist.append(P_exit_ratio_limit)

    _axP.clear()
    _axP.plot(
        _iter_hist, _P_exit_ratio_hist, "-o", color="tab:green", label="P_exit_ratio"
    )
    if any(v is not None for v in _P_exit_ratio_limit_hist):
        _axP.plot(
            _iter_hist, _P_exit_ratio_limit_hist, "r--", linewidth=0.65, label="limit"
        )
    _axP.set_ylabel("P_exit_ratio")
    _axP.set_xlabel("Iteration")
    _axP.grid(True)
    _autoscale_to_data(_axP, _P_exit_ratio_hist)

    # Component-fraction legend, anchored to axX's own axis (not the whole
    # figure) at its bottom-right corner — sits between axX and axZ, which
    # share the same component set. Axis-anchored (not Figure-level
    # "outside ...") so it plays nicely with the fixed margin above.
    handles, labels = _axX.get_legend_handles_labels()
    _axX.legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
    )

    # P_exit_ratio legend anchored to its own axis, so it's vertically
    # centered on that axis regardless of subplot sizing.
    _axP.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)

    _figF.canvas.draw()
    _figF.canvas.flush_events()
