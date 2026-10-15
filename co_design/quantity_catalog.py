"""Named catalog of scalar quantities extractable from
scramjet_point.evaluate_scramjet_point()'s result dict.

Mirrors single_point/problem_catalog.py's CATALOG pattern (a registry of
named, independently-pullable quantities), adapted for this stack: value and
sensitivity aren't separate callables here, because everything in this
pipeline is already FloatWithSens -- one entry's extractor returns a value
that already carries its own DV-sensitivity via `.sens`, no second sens()
function needed.

Each entry is a pure pull: `extract(result, aoa) -> FloatWithSens` (or a
plain float, for anything not FWS-tracked). `aoa` is accepted uniformly by
every entry even though only the wind-axis-rotated ones (CD_*/CL_*) use it,
so a caller (e.g. a future generic deck-builder) can call every catalog
entry the same way without knowing which ones need it.

Caution: AeroResults.wind_forces/wind_coeffs' docstring says it returns
(L, D, M) but the code actually returns (D, L, M) -- see PROJECT_STATE.md.
_wind_coeff() below indexes by the code-verified order.
"""

# SHARED WITH single_point: scramjet_mdao/single_point/ imports this file directly
# (CATALOG entries; single_point/case_config.py's VALID_QUANTITY_NAMES must stay in sync with it).
# Changing anything it uses affects BOTH the co-design pipeline and single-point
# optimisation -- check scramjet_mdao/single_point/ (Run_ObjFunc.py, case_config.py)
# and the note in scramjet_mdao/co_design/PROJECT_STATE.md before editing.

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Quantity:
    extract: Callable[[dict, float], Any]
    description: str = ""


def _wind_coeff(bucket_name: str, index: int) -> Callable[[dict, float], Any]:
    """index 0 -> CD (drag), index 1 -> CL (lift), per AeroResults.wind_coeffs'
    actual (D, L, M) return order."""

    def extractor(result: dict, aoa: float):
        return result[bucket_name].wind_coeffs(aoa=aoa)[index]

    return extractor


CATALOG: dict[str, Quantity] = {
    "CD_aero": Quantity(
        _wind_coeff("aero", 0),
        "Aero bucket (top + cowl_out + inlet) drag coefficient, wind axis. "
        "No equivalence_ratio dependence.",
    ),
    "CL_aero": Quantity(
        _wind_coeff("aero", 1),
        "Aero bucket lift coefficient, wind axis.",
    ),
    "CD_propulsion": Quantity(
        _wind_coeff("propulsion", 0),
        "Propulsion bucket (cowl_in + engine + nozzle) drag-axis coefficient, "
        "wind axis. Depends on equivalence_ratio.",
    ),
    "CL_propulsion": Quantity(
        _wind_coeff("propulsion", 1),
        "Propulsion bucket lift-axis coefficient, wind axis.",
    ),
    "CD_friction": Quantity(
        _wind_coeff("friction", 0),
        "Skin-friction drag coefficient, wind axis. Body-axis friction force "
        "is purely -x, but rotating to wind axis at nonzero aoa still gives "
        "this a nonzero value -- see CL_friction.",
    ),
    "CL_friction": Quantity(
        _wind_coeff("friction", 1),
        "Skin-friction contribution to the lift-axis coefficient, wind axis "
        "-- nonzero away from aoa=0 even though the underlying body-axis "
        "force has no z-component, purely from the wind-axis rotation.",
    ),
    "P_out": Quantity(
        lambda result, aoa: result["models"]["engine"].outflow.p,
        "Engine (combustor) outflow static pressure [Pa].",
    ),
    "T_out": Quantity(
        lambda result, aoa: result["models"]["engine"].outflow.T,
        "Engine (combustor) outflow static temperature [K].",
    ),
    "M_out": Quantity(
        lambda result, aoa: result["models"]["engine"].outflow.mach,
        "Engine (combustor) outflow Mach number.",
    ),
    "P_exit": Quantity(
        lambda result, aoa: result["models"]["nozzle"].P_exit,
        "Nozzle exit-plane (aft) static pressure [Pa].",
    ),
    "m_dot_f": Quantity(
        lambda result, aoa: result["models"]["engine"].m_dot_f,
        "Fuel mass flow rate [kg/s].",
    ),
}
