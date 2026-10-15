import os
import sys
import traceback

import numpy as np

from functools import partial, update_wrapper
from collections import OrderedDict
from pathlib import Path

from pyoptsparse import Optimization, OPT

from utilities import get_number
from plot_geom import plot_geom
from plot_converge import plot_converge_dvs
from plot_converge_obj import plot_converge_obj
from plot_forces import plot_forces
from plot_outflow import plot_outflow
from Run_ObjFunc import Run_ObjFunc, get_surrogate, EvaluationFailure
from Run_SensFunc import Run_SensFunc
from record import slim_out4sens
from PostOpt import SaveHistory
from OptLogger import OptLogger
from case_config import load_case_config


### for updating plots #####
import matplotlib

matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt

plt.ion()
#############################


iteration_count = 1


def objfunc(xdict, Opt_params={}, con_lim={}):
    print("here")
    global iteration_count

    print(f"xdict={xdict}")

    try:
        _, con, out4sens = Run_ObjFunc(xdict, Opt_params)
    except EvaluationFailure as e:
        # Only the expected can't-evaluate-this-design failures land here (a
        # coding error still crashes). fail=1 makes SNOPT back off to a
        # shorter step. pyoptsparse still processes `funcs` on a failed call,
        # so every objective/constraint name must be present (dummy values).
        print(f"EVALUATION FAILED -> fail=1 returned to SNOPT: {e}")
        traceback.print_exception(e.__cause__ or e)
        return {name: 0.0 for name in (*Consts, OBJ_NAME)}, 1

    #### Every catalog quantity, scaled by its case-config entry (default
    #### 1.0 for anything not named in objective/constraints) -- computed
    #### and logged regardless of what's active, per problem_catalog.py. ####
    funcs = {name: get_number(value) / SCALES.get(name, 1.0) for name, value in con.items()}

    #### Objective: weighted sum of one or more catalog quantities ####
    funcs[OBJ_NAME] = sum(
        term["weight"] * funcs[name] for name, term in Objective.items()
    )

    print(f"obj={funcs[OBJ_NAME]}")

    funcs["out4sens"] = slim_out4sens(out4sens)  # only what the old repo saved

    logger.log_obj(xdict, funcs)

    if Opt_params["plot_flg"]:

        plot_geom(out4sens["geom"], iteration_count, out4sens["inlet_model"])

        plot_converge_dvs(
            xdict,
            out4sens["nozzle_diag"],
            iteration_count,
            bounds=NOZZLE_BOUNDS,
            dv_bounds=DV,
        )

        outflow = get_number(out4sens["engine_model"].outflow)
        m_dot = get_number(out4sens["engine_model"].axial_conserve_5.m_dot)
        out4mat = get_number(out4sens["out4mat"])
        plot_outflow(
            outflow,
            m_dot,
            iteration_count,
            state_C=out4mat["C"],
            state_I=out4mat["I"],
            con_lim=con_lim,
        )

        alpha = out4sens["alpha"]
        comp_force = get_number(out4sens["comp_force"])
        plot_forces(
            comp_force,
            iteration_count,
            alpha,
            "Wind",
            P_exit_ratio=funcs["P_exit_ratio"],
            P_exit_ratio_limit=Consts.get("P_exit_ratio", {}).get("glb"),
        )

    iteration_count += 1

    fail = False

    return funcs, fail


def sensfunc(xdict, funcs, Opt_params={}):

    raw_sens = Run_SensFunc(funcs["out4sens"], Opt_params)

    #### Same per-name scale as objfunc's funcs[], applied to every
    #### catalog entry's gradient. ####
    sens = {
        name: {param: g / SCALES.get(name, 1.0) for param, g in grad.items()}
        for name, grad in raw_sens.items()
    }
    sens[OBJ_NAME] = {
        param: sum(term["weight"] * sens[name][param] for name, term in Objective.items())
        for param in xdict
    }

    logger.log_sens(xdict, sens)
    return sens

def snstop_callback(iterDict):
    """SNOPT's snSTOP hook — fires once per major iteration, with SNOPT's
    own optimality/feasibility measures already computed (iterDict), plus
    the funcs dict for the matched history call (if a match was found).
    Drives the objective/optimality/feasibility convergence plot, which
    therefore updates on major-iteration cadence rather than every objfunc
    call.
    """
    funcs = iterDict.get("funcs", {})
    obj_value = funcs.get(OBJ_NAME)
    if obj_value is None:
        return None
    plot_converge_obj(
        iterDict["nMajor"],
        obj_value,
        iterDict["optimality"],
        iterDict["feasibility"],
        obj_label=OBJ_NAME,
    )
    return None


DEFAULT_CASE_CONFIG = "opt_input.py"

if len(sys.argv) > 2:
    raise SystemExit(f"Usage: python {sys.argv[0]} [case_config.py]  (default: {DEFAULT_CASE_CONFIG})")

case_config_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CASE_CONFIG
if len(sys.argv) == 1:
    print(f"No case config given -- defaulting to {DEFAULT_CASE_CONFIG}")

case = load_case_config(case_config_path)

# Optional `headless = True` in the case config: plots are still created and saved (PostOpt
# saves every open figure) but no window opens during the run and the final plt.show() is
# skipped, so the run can go unattended / several at once. Absent or False -> unchanged.
HEADLESS = bool(getattr(case, "headless", False))
if HEADLESS:
    plt.switch_backend("Agg")

# Optional output folder from the case config: everything (SNOPT files, the .hst and the saved
# run folder output_dir/<case_name>/) is written there. A relative output_dir is relative to
# the directory the run is launched from. The config path is made absolute first because the
# process changes into the output folder; without output_dir nothing changes (cwd is used).
case_config_path = str(Path(case_config_path).resolve())
if getattr(case, "output_dir", None):
    _out = Path(case.output_dir).resolve()
    _out.mkdir(parents=True, exist_ok=True)
    os.chdir(_out)
    print(f"Output directory: {_out}")

DV = case.DV
Consts = case.constraints
Objective = case.objective  # {name: {"weight":.., "scale":..}, ...}

# DVs split by their three-state `active` flag (default True):
#   True      -> optimized by SNOPT (addVar below)
#   False     -> held fixed at its x0, injected into invars as a constant
#   "default" -> omitted from invars entirely, so Eng_Build's own hardcoded
#                default for that attribute applies
ACTIVE_DV = {name: term for name, term in DV.items() if term.get("active", True) is True}
FIXED_DV = {
    name: term["x0"] for name, term in DV.items() if term.get("active", True) is False
}
DEFAULT_DV = [name for name, term in DV.items() if term.get("active", True) == "default"]

# The SNOPT objective name: the single term's own name, or an explicit
# objective_name when the objective is a weighted combination of several
# catalog quantities (case_config.py requires objective_name in that case).
if len(Objective) == 1:
    OBJ_NAME = next(iter(Objective))
else:
    OBJ_NAME = case.objective_name

# Per-catalog-name scale (default 1.0), covering both objective terms and
# constraints -- applied uniformly to every quantity's value/gradient in
# objfunc/sensfunc, replacing the old CD_scale/Tout_scale special cases.
SCALES = {
    **{name: term.get("scale", 1.0) for name, term in Objective.items()},
    **{name: term.get("scale", 1.0) for name, term in Consts.items()},
}

# Opt_params is the single bag threaded into every Run_ObjFunc/Run_SensFunc
# call, so nozzle_surrogate/aoa_deg/objective travel with it rather than
# needing separate parameters down the objfunc -> Run_ObjFunc ->
# calc_vehicle chain. case.opt_params itself is left untouched (it's the
# loaded module's own dict).
Opt_params = dict(case.opt_params)
Opt_params["nozzle_surrogate"] = case.nozzle_surrogate
Opt_params["aoa_deg"] = getattr(case, "aoa_deg", 0.0)
Opt_params["objective"] = Objective
Opt_params["objective_name"] = OBJ_NAME
Opt_params["fixed_dvs"] = FIXED_DV
Opt_params["default_dvs"] = DEFAULT_DV
# Recorded here (not a separate run_details key) so it rides along in the
# existing Opt_params diff category for free -- a config rename/swap
# between a run and its parent shows up as a structured change automatically.
Opt_params["case_config_file"] = Path(case_config_path).name

# The selected surrogate's training-box BOUNDS -- resolved once here (not
# per-iteration) since it's used for plotting bounds, not the physics.
_, _nozzle_cfg = get_surrogate(case.nozzle_surrogate)
NOZZLE_BOUNDS = _nozzle_cfg.BOUNDS

con_lim = {  ### For plotting outflow -- Outflow_M/Outflow_T may not be
    ### active in every case config, hence the .get() guards.
    "LimM": Consts.get("Outflow_M", {}).get("glb"),
    "LimT": (
        Consts["Outflow_T"]["gub"] * SCALES.get("Outflow_T", 1.0)
        if "Outflow_T" in Consts
        else None
    ),
    "BoxM_lo": NOZZLE_BOUNDS["M_out"][0],
    "BoxM_hi": NOZZLE_BOUNDS["M_out"][1],
    "BoxP_lo": NOZZLE_BOUNDS["P_out"][0] / 1000,
    "BoxP_hi": NOZZLE_BOUNDS["P_out"][1] / 1000,
}


# Optimization problem
objFun = partial(objfunc, Opt_params=Opt_params, con_lim=con_lim)
update_wrapper(objFun, objfunc)
sensFun = partial(sensfunc, Opt_params=Opt_params)
update_wrapper(sensFun, sensfunc)
prob = Optimization("OptInlet", objFun=objFun, sens=sensFun)
# prob = Optimization("OptInlet", objFun=objfunc, sens="FD")

# Populate DVs (active ones only -- fixed/default DVs never reach SNOPT)
for param, val in ACTIVE_DV.items():
    prob.addVar(
        name=param,
        lower=val["flb"],
        upper=val["fub"],
        value=val["x0"],
        varType="c",
        scale=1 / (val["fub"] - val["flb"]),
        offset=val["flb"],
    )


# Populate Constrains
for param, val in Consts.items():
    prob.addCon(
        name=param,
        upper=val["gub"],
        lower=val["glb"],
    )

# Objective
prob.addObj(OBJ_NAME)

options = dict(case.snopt_options)
if Opt_params["plot_flg"]:  # the callback only draws the convergence plot
    options["snSTOP function handle"] = snstop_callback
print(prob)
optimizer = OPT(optName="SNOPT", options=options)

hist_file_name = f"{case.case_name}.hst"
logger = OptLogger(objective_name=OBJ_NAME)

sol = optimizer(prob, storeHistory=hist_file_name)
# sol = optimizer(prob, sens="FD", storeHistory=hist_file_name)
print(sol)


#########################
##### Save session ######
#########################
xdict = OrderedDict(
    (name, np.array(vars_[0].value)) for name, vars_ in sol.variables.items()
)
const_vars = Run_ObjFunc(xdict, Opt_params, save_flg=True)
# Normalize so every saved DV entry explicitly carries "active" -- otherwise
# a DV that never set the flag (implicitly True) would diff as
# "active: None -> True" against a run that wrote it explicitly, even
# though nothing about the run actually changed.
DV_saved = {name: {**term, "active": term.get("active", True)} for name, term in DV.items()}
run_details = {
    "DV": DV_saved,
    "Con": Consts,
    # SNOPT options can hold a callable (snSTOP function handle); swap any
    # such entries for their name so the saved record stays JSON-safe.
    "options": {k: (v.__name__ if callable(v) else v) for k, v in options.items()},
    "const_vars": const_vars,
    "Opt_params": Opt_params,
}

SaveHistory(hist_file_name, logger, run_details, case_config_path=case_config_path)


#### keep plots open #####
plt.ioff()
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
if not HEADLESS:
    plt.show()


