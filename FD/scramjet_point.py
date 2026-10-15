"""Single flight-condition evaluation of scramjet forces.

evaluate_scramjet_point() is the core function step 3 populates the CasADi
surrogate grid with: given a fixed geometry (from geometry_generation.
generate_scramjet_sj2d) and one flight condition (alpha, mach, altitude,
equivalence_ratio), it returns every individual component's AeroResults plus
the three CasADi-facing buckets (aero, propulsion, friction). It's a plain
point evaluation with no grid/loop logic of its own, so the same function
also serves a single-point optimisation directly.
"""

# SHARED WITH single_point: scramjet_mdao/single_point/ imports this file directly
# (evaluate_scramjet_point and the structure of its result dict).
# Changing anything it uses affects BOTH the co-design pipeline and single-point
# optimisation -- check scramjet_mdao/single_point/ (Run_ObjFunc.py, case_config.py)
# and the note in scramjet_mdao/co_design/PROJECT_STATE.md before editing.

from ambiance import Atmosphere as Atm

from utilities import AlphaPhi2Flow

from Scramjet2D import Parametric_Scramjet2D, Force_Calculator, PatchTag, sum_aero_results


def evaluate_scramjet_point(
    alpha: float,
    mach: float,
    altitude: float,
    equivalence_ratio: float,
    vehicle,
    fws: dict,
    context,
    overrides: dict,
    nozzle_surrogate,
    cf_model,
    S_ref: float = 1.0,
    verbosity: int = 0,
) -> dict:
    """
    Arguments
    ----------
    alpha, mach, altitude, equivalence_ratio
        The one flight condition to evaluate at.
    vehicle, fws, context, overrides
        As returned by geometry_generation.generate_scramjet_sj2d() -- the
        geometry/mesh is held fixed here; only the four flight-condition
        arguments above vary between calls.
    nozzle_surrogate
        A loaded hyperProp.Nozzle.nozzle_surrogate_model.NozzleSurrogateModel
        (load once, e.g. in MultistageManager.__init__ -- not here).
    cf_model
        A loaded hypervisc friction model (load once, same reasoning).
    S_ref
        Reference area for force/moment coefficients on the three returned
        buckets (aero/propulsion/friction) -- e.g. vehicle.mesh_data.
        calculate_volume() ** (2/3), computed once per outer-loop design
        point since geometry doesn't change across a flight-condition sweep.
        Defaults to 1.0 (i.e. AeroResults' own default) when unset.

    Returns
    ----------
    dict with:
        "components": every individual contribution's AeroResults, keyed by
            name ("top", "cowl_out", "inlet", "cowl_in", "engine", "nozzle",
            "friction") -- for debugging/comparison against single_point.
            "cowl_in"/"engine"/"nozzle" are None if the propulsion chain
            failed (see "propulsion_error").
        "aero": top + cowl_out + inlet (no equivalence_ratio dependence).
            Always populated -- does not depend on the propulsion chain
            succeeding (see note below).
        "propulsion": cowl_in + engine + nozzle, or None if that chain
            raised (PROJECT_STATE.md decision 14 -- the cowl-turn and
            engine-conservation solves are known-fragile near thermal
            choke). Check "propulsion_error" to distinguish "failed" from
            "genuinely zero".
        "propulsion_error": None on success, else str(exception) from
            whichever propulsion-chain stage raised.
        "friction": same object as components["friction"], surfaced at the
            top level too since it's one of the three CasADi-facing pieces.
            Always populated, same reasoning as "aero".
        "freestream": the FlowState built from (alpha, mach, altitude).

    Note on partial failure: inlet is computed first and is allowed to raise
    normally (nothing downstream is salvageable without it -- inlet failures
    are not part of the known propulsion-only fragility this function
    guards against). cowl/engine/nozzle are wrapped together: if any of them
    raises, "aero"/"friction" are still computed and returned (they only
    need inlet_force + the mesh + freestream, confirmed independent of the
    propulsion chain -- see below), while "propulsion" and the
    cowl/engine/nozzle entries in "models"/"components" become None. This
    matters for the deck-generation grid sweep (PROJECT_STATE.md decision
    14): one bad propulsion node shouldn't cost the caller the aero/friction
    data that already succeeded at that same node.
    """
    atm = Atm(altitude)
    freestream = AlphaPhi2Flow(
        mach=mach, temperature=atm.temperature[0], pressure=atm.pressure[0], alpha=alpha
    )

    eng_model = Parametric_Scramjet2D(
        {**fws, **overrides, "freestream": freestream, "equivalence_ratio": equivalence_ratio},
        verbosity=verbosity,
    )
    eng_model.create_vehicle_geom()

    inlet = eng_model.create_inlet()
    conserve_2, inlet_force = inlet.calc_inlet()

    cowl = engine = nozzle = None
    cowl_in_force = engine_force = nozzle_force = outflow = None
    propulsion_error = None
    try:
        cowl = eng_model.create_cowl(conserve_2)
        conserve_3, cowl_in_force = cowl.calc_cowl()

        engine = eng_model.create_engine(conserve_3)
        outflow, engine_force = engine.calc_engine()

        nozzle = eng_model.create_nozzle(outflow, nozzle_surrogate)
        nozzle_force = nozzle.calc_nozzle()
    except Exception as e:
        propulsion_error = str(e)

    # Reused mesh from step 2 (geometry/normals/areas unchanged); only the
    # flow evaluation at this condition is new. eng_flow only ever reaches
    # NOZZLE-tagged panels (Force_Calculator.calculate_panel_flow), and
    # NOZZLE is PanelScheme.DONT_CALC (_parse_vehicle) -- never an actually
    # computed force -- so TOP/COWL (everything "aero" uses) are provably
    # unaffected by eng_flow's value. Falling back to freestream here when
    # propulsion failed (outflow is None) is therefore safe, not a fudge.
    panel_solver = Force_Calculator(
        mesh=vehicle.mesh_data,
        tags=vehicle.pvmesh.cell_data["PatchTag"],
        context=context,
    )
    panel_solver.calculate_panel_flow(
        freestream=freestream, eng_flow=outflow if outflow is not None else freestream
    )
    top_force = panel_solver.patch_aero_result(PatchTag.TOP)
    cowl_out_force = panel_solver.patch_aero_result(PatchTag.COWL)
    friction_force = panel_solver.calculate_friction_force(freestream, cf_model, eng_model._geom["length"]).update_input(S_ref=S_ref)

    ref_point = panel_solver.reference_point
    aero = sum_aero_results(
        [top_force, cowl_out_force, inlet_force], freestream, ref_point
    ).update_input(S_ref=S_ref)

    if propulsion_error is None:
        propulsion = sum_aero_results(
            [cowl_in_force, engine_force, nozzle_force], freestream, ref_point
        ).update_input(S_ref=S_ref)
    else:
        propulsion = None

    return {
        "models": {
            "inlet": inlet,
            "cowl": cowl,
            "engine": engine,
            "nozzle": nozzle,
            "eng_model": eng_model,
        },
        "components": {
            "top": top_force,
            "cowl_out": cowl_out_force,
            "inlet": inlet_force,
            "cowl_in": cowl_in_force,
            "engine": engine_force,
            "nozzle": nozzle_force,
            "friction": friction_force,
        },
        "aero": aero,
        "propulsion": propulsion,
        "propulsion_error": propulsion_error,
        "friction": friction_force,
        "freestream": freestream,
    }


if __name__ == "__main__":
    pass
