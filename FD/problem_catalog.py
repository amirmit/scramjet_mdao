"""Registry of every named scalar quantity Run_ObjFunc.calc_vehicle returns --
the names a case config may use as an objective term or a constraint.

Unlike HyperMDAO's problem_catalog (a value AND a sens lambda per entry, over
hand-assembled sensitivity building blocks), each quantity here is already a
FloatWithSens whose `.sens` is d(value)/d(DV) in context.variable_names
order, so the catalog is just names + descriptions, and Run_SensFunc pulls
`.sens[i]` per quantity.

Keep in sync with Run_ObjFunc.calc_vehicle (checked at the end of every
evaluation by check_quantities) and co_design/quantity_catalog.CATALOG (the
first block below is that shared catalog's entries).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    description: str
    unit: str = ""


CATALOG: dict[str, Quantity] = {
    # ---- co_design/quantity_catalog.CATALOG (shared) -----------------------
    "CD_aero": Quantity("Aero bucket (top + cowl_out + inlet) drag coefficient, wind axis."),
    "CL_aero": Quantity("Aero bucket lift coefficient, wind axis."),
    "CD_propulsion": Quantity("Propulsion bucket (cowl_in + engine + nozzle) drag coefficient, wind axis."),
    "CL_propulsion": Quantity("Propulsion bucket lift coefficient, wind axis."),
    "CD_friction": Quantity("Skin-friction drag coefficient, wind axis."),
    "CL_friction": Quantity("Skin-friction lift coefficient, wind axis (zero at aoa=0)."),
    "P_out": Quantity("Engine (combustor) outflow static pressure.", "Pa"),
    "T_out": Quantity("Engine (combustor) outflow static temperature.", "K"),
    "M_out": Quantity("Engine (combustor) outflow Mach number."),
    "P_exit": Quantity("Nozzle exit-plane static pressure.", "Pa"),
    "m_dot_f": Quantity("Fuel mass flow rate.", "kg/s"),
    # ---- single_point/Run_ObjFunc.calc_vehicle -----------------------------
    "mass": Quantity("Vehicle mass (Fleeman correlation).", "kg"),
    "weight": Quantity("Vehicle weight.", "N"),
    "fuel_mass": Quantity("Maximum fuel mass (10% of volume at 35 kg/m^3).", "kg"),
    "CD": Quantity("Total drag coefficient (aero + propulsion [+ friction]), wind axis."),
    "CL": Quantity("Total lift coefficient (aero + propulsion [+ friction]), wind axis."),
    "T_W_ratio": Quantity("Wind-axis x force / weight (HyperMDAO's thrust/weight ratio)."),
    "L_W_ratio": Quantity("Lift / weight ratio."),
    "m_dot_f_ratio": Quantity("Fuel mass flow / fuel-budget mass flow limit."),
    "mf_dV": Quantity("mf/Delta_V -- for accelerators."),
    "range": Quantity("Breguet range.", "m"),
    "Outflow_M": Quantity("Combustor exit Mach number (alias of M_out)."),
    "Outflow_P": Quantity("Combustor exit pressure (alias of P_out).", "Pa"),
    "Outflow_T": Quantity("Combustor exit temperature (alias of T_out).", "K"),
    "P_exit_ratio": Quantity("Nozzle exit pressure / freestream pressure."),
    "tail_z": Quantity("Nose z minus tail z (positive = tail above nose).", "m"),
    "internal_vol": Quantity("Vehicle volume minus engine internal volume.", "m^3"),
    "theta_nose": Quantity("Nose leading-edge angle.", "rad"),
    "volume": Quantity("Total vehicle volume.", "m^3"),
}


def check_quantities(quantities: dict) -> None:
    """Raise if a quantities dict doesn't have exactly the CATALOG's names."""
    extra = sorted(set(quantities) - set(CATALOG))
    missing = sorted(set(CATALOG) - set(quantities))
    if extra or missing:
        raise KeyError(
            "Run_ObjFunc quantities and problem_catalog.CATALOG disagree -- "
            f"not in catalog: {extra}; missing from quantities: {missing}"
        )
