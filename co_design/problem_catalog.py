"""Registry of every named scalar quantity usable as an objective or a
constraint. A case config selects which entries are active (and, for
constraints, their bounds); every entry is still computed and logged each
iteration regardless of what's active (cheap -- arithmetic over state
calc_vehicle/Run_SensFunc already produce).

value(out4sens) -> raw value (a FloatWithSens where the model carries
sensitivity, a plain float/Vector3 component otherwise) -- the same
convention calc_vehicle's old positional `constraints` list used.

sens(ctx, param, i) -> d(value)/d(param), unscaled. `ctx` is out4sens
enriched by Run_SensFunc.py with the sensitivity building blocks several
entries share (HP_sens, sens_vec, sens_coeffs) plus any quantity-specific
gradient that combines those with an autodiff-only quantity (weight,
cells.dvoldp) via a one-off formula not shared by anything else
(T_W_ratio_sens, L_W_ratio_sens, m_dot_f_ratio_sens) -- computed in
Run_SensFunc next to where the matching value is computed in calc_vehicle,
so every entry here is a pure pull. `i` is the positional index of `param`
within `parameters`, matching how dvoldp/get_sens are indexed.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from hypervehicle.geometry.autodiff import get_sens
from sphinx.addnodes import desc_type


@dataclass
class Quantity:
    value: Callable[[dict], Any]
    sens: Callable[[dict, str, int], float]
    description: str = ""


CATALOG: Dict[str, Quantity] = {
    "CD": Quantity(
        value=lambda o: o["force_coeffs"].x,
        sens=lambda ctx, p, i: ctx["sens_coeffs"][p].x,
        description="Drag coefficient (wind axis).",
    ),
    "T_W_ratio": Quantity(
        value=lambda o: o["cat_values"]["T_W_ratio"],
        sens=lambda ctx, p, i: ctx["T_W_ratio_sens"][p],
        description="Thrust/weight ratio.",
    ),
    "mf_dV": Quantity(
        value=lambda o: o["cat_values"]["mf_dV"],
        sens=lambda ctx, p, i: ctx["mf_dV_sens"][p],
        description="mf/Delta_V - for accelerators",
    ),
    "range": Quantity(
        value=lambda o: o["cat_values"]["range"],
        sens=lambda ctx, p, i: get_sens(ctx["range_sens"])[i],
        description="Bregue Range equation",
    ),
    "m_dot_f": Quantity(
        value=lambda o: o["engine_model"].mdf_model["m_dot_f"],
        sens=lambda ctx, p, i: get_sens(ctx["mdf_model"]["m_dot_f"])[i],
        description="mf/Delta_V - for accelerators",
    ),
    "Outflow_M": Quantity(
        value=lambda o: o["engine_model"].outflow.M,
        sens=lambda ctx, p, i: ctx["HP_sens"][p].M,
        description="Combustor exit Mach number.",
    ),
    "Outflow_T": Quantity(
        value=lambda o: o["engine_model"].outflow.T,
        sens=lambda ctx, p, i: ctx["HP_sens"][p].Temp,
        description="Combustor exit temperature.",
    ),
    "Outflow_P": Quantity(
        value=lambda o: o["engine_model"].outflow.P,
        sens=lambda ctx, p, i: ctx["HP_sens"][p].P,
        description="Combustor exit temperature.",
    ),
    "volume": Quantity(
        value=lambda o: o["cells"].vol,
        sens=lambda ctx, p, i: ctx["cells"].dvoldp[i],
        description="Total vehicle volume.",
    ),
    "L_W_ratio": Quantity(
        value=lambda o: o["cat_values"]["L_W_ratio"],
        sens=lambda ctx, p, i: ctx["L_W_ratio_sens"][p],
        description="Lift/weight ratio.",
    ),
    "m_dot_f_ratio": Quantity(
        value=lambda o: o["engine_model"].mdf_model["m_dot_f_ratio"],
        sens=lambda ctx, p, i: ctx["m_dot_f_ratio_sens"][p],
        description="Fuel mass flow / fuel-budget mass flow limit.",
    ),
    "theta_nose": Quantity(
        value=lambda o: o["cat_values"]["theta_nose"],
        sens=lambda ctx, p, i: get_sens(ctx["cat_values"]["theta_nose"])[i],
        description="Nose leading edge angle",
    ),
    "P_exit_ratio": Quantity(
        value=lambda o: o["cat_values"]["P_exit_ratio"],
        sens=lambda ctx, p, i: get_sens(ctx["cat_values"]["P_exit_ratio"])[i],
        description="Nozzle exit pressure / freestream pressure.",
    ),
    "tail_z": Quantity(
        value=lambda o: o["cat_values"]["tail_z"],
        sens=lambda ctx, p, i: get_sens(ctx["cat_values"]["tail_z"])[i],
        description="Nose z minus tail z (positive = tail above nose).",
    ),
    "internal_vol": Quantity(
        value=lambda o: o["cat_values"]["internal_vol"],
        sens=lambda ctx, p, i: ctx["internal_vol_sens"][p],
        description="Vehicle volume minus engine internal volume.",
    ),
}
