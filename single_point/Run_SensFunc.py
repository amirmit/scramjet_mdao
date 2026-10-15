"""Sensitivities of every catalog quantity w.r.t. the active DVs.

Every quantity Run_ObjFunc returns is a FloatWithSens whose `.sens` is
d(value)/d(DV) in `context.variable_names` order (== the active-DV xdict
order), so there is nothing to assemble here -- just pull each entry out by
DV name. Weighting/scaling is applied by Opt_Main.sensfunc, the same place
the values are scaled.
"""


import numpy as np


def normalize_out4sens(out4sens):
    """pyoptsparse applies np.real to every value in the funcs dict, which
    turns the out4sens dict into a 0-d object array -- unwrap it."""
    if isinstance(out4sens, np.ndarray):
        return out4sens[()]
    return out4sens


def Run_SensFunc(out4sens, Opt_params=None):
    """
    :param out4sens: the dict Run_ObjFunc returns (needs "quantities" and
        "context").
    :param Opt_params: unused; kept so the signature matches Opt_Main's call.
    :return: {quantity_name: {dv_name: d(quantity)/d(dv_name)}} (unscaled)
    """
    out4sens = normalize_out4sens(out4sens)
    params = out4sens["context"].variable_names

    sens = {}
    for name, q in out4sens["quantities"].items():
        if not hasattr(q, "sens"):
            # A plain float here would silently give an all-zero gradient.
            raise TypeError(
                f"quantity '{name}' has no .sens ({type(q).__name__}) -- it "
                f"was not carried through the FloatWithSens chain."
            )
        sens[name] = {param: float(q.sens[i]) for i, param in enumerate(params)}
    return sens
