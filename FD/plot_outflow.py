import matplotlib.pyplot as plt

# Stations plotted: E = engine outflow (into nozzle), C = post-inlet-model
# (cowl.flowstate_2, combustor-entrance averaged flow before cowl turning),
# I = post-cowl-model (cowl.flowstate_3, after cowl turning / isolator entry).
_STATIONS = ["E", "C", "I"]
_STATION_COLORS = {"E": "tab:blue", "C": "tab:orange", "I": "tab:green"}
_H_STATIONS = ["C", "I"]

# Storage for history
_iters = []
_md_vals = []
_P_vals = {s: [] for s in _STATIONS}
_M_vals = {s: [] for s in _STATIONS}
_T_vals = {s: [] for s in _STATIONS}
_h_vals = {s: [] for s in _H_STATIONS}
_LimM_val = []
_LimT_val = []
_BoxM_lo_val = []
_BoxM_hi_val = []
_BoxP_lo_val = []
_BoxP_hi_val = []

_fig = None
_axes = None
_line_md = None
_lines_P = {}
_lines_M = {}
_lines_T = {}
_lines_h = {}
_line_LimM = None
_line_LimT = None
_line_BoxM_lo = None
_line_BoxM_hi = None
_line_BoxP_lo = None
_line_BoxP_hi = None


def init_flow_plot():
    global _fig, _axes, _line_md, _lines_P, _lines_M, _lines_T, _lines_h
    global _line_LimM, _line_LimT
    global _line_BoxM_lo, _line_BoxM_hi, _line_BoxP_lo, _line_BoxP_hi

    plt.ion()  # interactive mode for live updating [web:8]
    _fig, _axes = plt.subplots(5, 1, sharex=True, figsize=(6, 9))  # [web:16]
    _fig.suptitle(
        "Combustor/nozzle-inflow conditions\n(E=engine outflow, C=post-inlet, I=post-cowl)"
    )
    _fig.name = "engine_outflow"

    axmd, axP, axM, axT, axH = _axes

    _line_md = axmd.plot([], [], "-o", color=_STATION_COLORS["E"], label="m_dot")[0]

    for s in _STATIONS:
        _lines_P[s] = axP.plot(
            [], [], "-o", color=_STATION_COLORS[s], label=f"P ({s})"
        )[0]
        _lines_M[s] = axM.plot(
            [], [], "-o", color=_STATION_COLORS[s], label=f"M ({s})"
        )[0]
        _lines_T[s] = axT.plot(
            [], [], "-o", color=_STATION_COLORS[s], label=f"T ({s})"
        )[0]

    for s in _H_STATIONS:
        _lines_h[s] = axH.plot(
            [], [], "-o", color=_STATION_COLORS[s], label=f"h ({s})"
        )[0]

    _line_LimM = axM.plot([], [], "r", linewidth=0.65, label="LimM")[0]
    _line_LimT = axT.plot([], [], "r", linewidth=0.65, label="LimT")[0]
    # Surrogate training-box bounds (M_out, P_out) — lo/hi pair per axis.
    # These apply to the engine outflow (station E), the surrogate's own input.
    _line_BoxM_lo = axM.plot(
        [], [], color="orange", linestyle="--", linewidth=0.65, label="surrogate box"
    )[0]
    _line_BoxM_hi = axM.plot([], [], color="orange", linestyle="--", linewidth=0.65)[0]
    _line_BoxP_lo = axP.plot(
        [], [], color="orange", linestyle="--", linewidth=0.65, label="surrogate box"
    )[0]
    _line_BoxP_hi = axP.plot([], [], color="orange", linestyle="--", linewidth=0.65)[0]

    axmd.set_ylabel("m_dot kg/s")
    axP.set_ylabel("Pressure kPa")
    axM.set_ylabel("Mach number")
    axT.set_ylabel("Temperature K")
    axH.set_ylabel("Enthalpy kJ/kg")
    axH.set_xlabel("Iteration")

    for ax in _axes:
        ax.grid(True)

    _fig.tight_layout()
    _fig.show()


def _autoscale_to_data(ax, *value_lists):
    """Set y-limits from the real trajectory data only, ignoring any
    reference/limit lines also drawn on this axis — so a limit line only
    becomes visible on screen once the trajectory actually gets close to it.
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


def plot_outflow(flowstate, m_dot, iteration, state_C=None, state_I=None, con_lim={}):
    """
    flowstate: engine outflow FlowState (station E)
    m_dot: engine exit mass flow [kg/s]
    state_C: out4mat["C"] dict (post-inlet-model, cowl.flowstate_2) or None
    state_I: out4mat["I"] dict (post-cowl-model, cowl.flowstate_3) or None
    con_lim: dict of reference-line values, see keys used below
    """
    global _iters, _md_vals, _fig, _axes
    global _P_vals, _M_vals, _T_vals, _h_vals
    global _LimM_val, _LimT_val, _BoxM_lo_val, _BoxM_hi_val, _BoxP_lo_val, _BoxP_hi_val

    if _fig is None:
        init_flow_plot()

    # Append new data
    _iters.append(iteration)
    _md_vals.append(m_dot)

    _P_vals["E"].append(flowstate.p / 1000)
    _M_vals["E"].append(flowstate.mach)
    _T_vals["E"].append(flowstate.T)

    for s, state in (("C", state_C), ("I", state_I)):
        if state is not None:
            _P_vals[s].append(state["P"] / 1000)
            _M_vals[s].append(state["M"])
            _T_vals[s].append(state["T"])
            _h_vals[s].append(state["h"] / 1000)
        else:
            _P_vals[s].append(None)
            _M_vals[s].append(None)
            _T_vals[s].append(None)
            _h_vals[s].append(None)

    _LimM_val.append(con_lim["LimM"]) if "LimM" in con_lim else _LimM_val.append(None)
    _LimT_val.append(con_lim["LimT"]) if "LimT" in con_lim else _LimT_val.append(None)
    _BoxM_lo_val.append(con_lim["BoxM_lo"]) if "BoxM_lo" in con_lim else _BoxM_lo_val.append(None)
    _BoxM_hi_val.append(con_lim["BoxM_hi"]) if "BoxM_hi" in con_lim else _BoxM_hi_val.append(None)
    _BoxP_lo_val.append(con_lim["BoxP_lo"]) if "BoxP_lo" in con_lim else _BoxP_lo_val.append(None)
    _BoxP_hi_val.append(con_lim["BoxP_hi"]) if "BoxP_hi" in con_lim else _BoxP_hi_val.append(None)

    # Update line data
    axmd, axP, axM, axT, axH = _axes

    _line_md.set_data(_iters, _md_vals)
    for s in _STATIONS:
        _lines_P[s].set_data(_iters, _P_vals[s])
        _lines_M[s].set_data(_iters, _M_vals[s])
        _lines_T[s].set_data(_iters, _T_vals[s])
    for s in _H_STATIONS:
        _lines_h[s].set_data(_iters, _h_vals[s])

    _line_LimM.set_data(_iters, _LimM_val)
    _line_LimT.set_data(_iters, _LimT_val)
    _line_BoxM_lo.set_data(_iters, _BoxM_lo_val)
    _line_BoxM_hi.set_data(_iters, _BoxM_hi_val)
    _line_BoxP_lo.set_data(_iters, _BoxP_lo_val)
    _line_BoxP_hi.set_data(_iters, _BoxP_hi_val)

    # Rescale axes. m_dot carries no reference line, so plain autoscale is
    # fine. P/M/T carry limit and/or surrogate-box reference lines, so their
    # y-limits are computed from the real trajectory only (see
    # _autoscale_to_data) — a reference line then only shows up once the
    # trajectory is actually close to it.
    axmd.relim()
    axmd.autoscale_view()
    _autoscale_to_data(axP, *_P_vals.values())
    _autoscale_to_data(axM, *_M_vals.values())
    _autoscale_to_data(axT, *_T_vals.values())
    _autoscale_to_data(axH, *_h_vals.values())

    # Update legends conditionally
    axmd.legend(handles=[_line_md], labels=[_line_md.get_label()])

    axP_handles = [_lines_P[s] for s in _STATIONS]
    if any(v is not None for v in _BoxP_lo_val):
        axP_handles.append(_line_BoxP_lo)
    axP.legend(handles=axP_handles, labels=[h.get_label() for h in axP_handles])

    axM_handles = [_lines_M[s] for s in _STATIONS]
    if any(v is not None for v in _LimM_val):
        axM_handles.append(_line_LimM)
    if any(v is not None for v in _BoxM_lo_val):
        axM_handles.append(_line_BoxM_lo)
    axM.legend(handles=axM_handles, labels=[h.get_label() for h in axM_handles])

    axT_handles = [_lines_T[s] for s in _STATIONS]
    if any(v is not None for v in _LimT_val):
        axT_handles.append(_line_LimT)
    axT.legend(handles=axT_handles, labels=[h.get_label() for h in axT_handles])

    axH_handles = [_lines_h[s] for s in _H_STATIONS]
    axH.legend(handles=axH_handles, labels=[h.get_label() for h in axH_handles])

    _fig.canvas.draw()
    _fig.canvas.flush_events()
