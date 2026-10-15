# SHARED WITH single_point: scramjet_mdao/single_point/ imports this file directly
# (generate_scramjet_sj2d; SJ2D_GEOM_PARAM_NAMES also defines single_point/case_config.py's VALID_DV_NAMES).
# Changing anything it uses affects BOTH the co-design pipeline and single-point
# optimisation -- check scramjet_mdao/single_point/ (Run_ObjFunc.py, case_config.py)
# and the note in scramjet_mdao/co_design/PROJECT_STATE.md before editing.

from hypervehicle import Vehicle
from autodiff import SensMode, generate_parameters_with_sensitivity
from autodiff.context import ADContext

from Scramjet2D import Scramjet2D_Comp


# Geometry attribute names accepted by Scramjet2D_Comp.set_defaults() (this
# file's Scramjet2D.py -- the one with the nozzle model, not the older copy
# in scramjet2D/SJ_run). Kept explicit rather than introspected, mirroring
# single_point/case_config.py's VALID_DV_NAMES: it's the only way
# generate_scramjet_sj2d() can split a combined geom+eng active-DV dict
# without being told the split by its caller. Any DV/override name not in
# this set is treated as engine-side and left untouched here, for the
# engine/aero step to consume later. If Scramjet2D_Comp's geometry
# parameters ever change, update this alongside it.
SJ2D_GEOM_PARAM_NAMES = frozenset(
    {
        "inlet_length_1",
        "inlet_length_2",
        "inlet_theta_1",
        "inlet_theta_2",
        "throat_height",
        "engine_tilt_ang",
        "cowl_LE",
        "engine_length",
        "isolator_length",
        "engine_h5_h3",
        "nozzle_length",
        "nozzle_angle",
        "nozzle_mid_frac",
        "bez_start_ratio",
    }
)


def generate_scramjet_sj2d(
    active_dv_dict: dict,
    overrides: dict | None = None,
    mesh_params: dict | None = None,
    save: bool = False,
    name: str | None = None,
) -> dict:
    """
    Build the SJ2D vehicle geometry, and the shared geom+eng FWS dict /
    ADContext that the (not yet implemented) engine+aero step reuses.

    `active_dv_dict` must be the full, combined geometry+engine active-DV
    dict (e.g. straight from ParameterSet.vec2dict(x)) -- it is promoted to
    FWS here, once, for both domains together, so geometry and engine
    sensitivities land on the same axis ordering (mirrors
    scramjet2D/SJ_run/Run_Scramjet2D.py's sj_run()). Never pass a filtered
    or reordered subset of this dict to generate_parameters_with_sensitivity
    anywhere else downstream -- axis assignment is purely positional
    (autodiff.generate_parameters.generate_parameters_with_sensitivity seeds
    by dict-iteration order), so a mismatch silently produces wrong/zero
    gradient columns rather than an error.

    `overrides` (constants for DVs currently held fixed, from the
    catalog's active=False entries) and `active_dv_dict` must not overlap.

    Returns
    ----------
    dict with:
        "vehicle": the generated Vehicle
        "fws": the full combined geom+eng FloatWithSens dict (unfiltered)
        "context": ADContext built from active_dv_dict's key order
        "overrides": the overrides dict passed in, unfiltered (the
            engine/aero step picks its own eng-relevant slice out of this
            and out of "fws")
    """
    overrides = overrides or {}

    FWS_param_dict = generate_parameters_with_sensitivity(active_dv_dict, SensMode.JACOBIAN)
    context = ADContext(list(active_dv_dict.keys()))

    geom_fws = {n: FWS_param_dict[n] for n in active_dv_dict if n in SJ2D_GEOM_PARAM_NAMES}
    geom_overrides = {n: v for n, v in overrides.items() if n in SJ2D_GEOM_PARAM_NAMES}
    geom_in_dict = geom_fws | geom_overrides

    scramjet_comp = Scramjet2D_Comp(
        label="sj2D", params=geom_in_dict, mesh_params=mesh_params
    )
    scramjet_comp.define_export_parameters({n: n for n in geom_fws})

    vehicle = Vehicle(
        name=name,
        components=[scramjet_comp],
        sensitivity=SensMode.JACOBIAN,
        refinement_level=1,
    )
    vehicle.apply_design_variables(
        geom_fws, SensMode.JACOBIAN, parameter_order=list(active_dv_dict.keys())
    )

    vehicle.generate(check_normal=True, delaunay_triangulation=False)
    if save:
        vehicle.save((name or "scramjet_vehicle") + ".vtk", binary=False)

    return {
        "vehicle": vehicle,
        "FWS_param_dict": FWS_param_dict,
        "context": context,
    }



if __name__ == "__main__":
    pass