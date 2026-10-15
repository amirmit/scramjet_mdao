"""Grid-sweep deck builder.

generate_scramjet_decks() evaluates evaluate_scramjet_point() over an
alpha x mach grid (altitude and equivalence_ratio held fixed as constants
for this first pass -- see scramjet_point.py) and fits one CasADi
interpolant per quantity_catalog.CATALOG entry, mirroring aerodeck.py's
generate_aerodeck() pattern for the glider vehicle, generalized from the
original 3 hardcoded (CD, CL) buckets to an arbitrary catalog of named
scalar quantities.

Uses a local single-output interpolant builder (_casadi_interpolant_from_df)
instead of hyperdynamics.aero_interpolate_from_df, which hardcodes exactly
two outputs named "CD"/"CL" -- most catalog entries now (P_out, m_dot_f,
etc.) are single, arbitrarily-named scalars. This local builder doesn't
touch hyperDynamics; generalizing aero_interpolate_from_df itself (in its
own branch) remains a separate, optional piece of work.

Verified against how these get consumed downstream (dynamics.py /
hyperdynamics.run_nlp.trajectory_optimisation): each ca.Function here is a
plain CasADi Function of (mach, alpha, *DV symbols) -- composes into a
larger symbolic expression graph the same way regardless of how many
separate Function objects contribute to it, and the outer-loop DV
sensitivity of the optimal NLP cost falls out of nlpsol's own lam_p
(Lagrange multipliers on the parameter vector) after solving, not from
anything specific to how these interpolants are built. The one real
invariant: every interpolant's internal parameter-symbol order must match
whatever DV name/order the dynamics function later builds its own
casadi_params from -- both derive from the same fws/active_dv_dict here, so
this holds automatically as long as that dict is never filtered/reordered.

PROPULSION-ONLY GUARD (PROJECT_STATE.md architecture decision 14)
-------------------------------------------------------------------
The engine's own internal conservation solve (and, rarely, the cowl-turn
solve) is known-fragile near thermal choking (engine outflow Mach close to
1) -- confirmed via a 100-point residual sweep to fail on ~65% of randomly
sampled points, both silently (scipy reports success=True with a large
residual) and as hard exceptions (negative pressure). scramjet_point.py's
evaluate_scramjet_point() already isolates this: a propulsion-stage failure
still returns valid "aero"/"friction" data, with "propulsion_error" set and
"propulsion"/"models"["engine"/"nozzle"] left None.

This module is where that gets turned into a usable deck: CD_aero/CL_aero/
CD_friction/CL_friction (independent of the propulsion chain) are extracted
at every grid node unconditionally; the 7 propulsion-dependent CATALOG
entries (CD_propulsion, CL_propulsion, P_out, T_out, M_out, P_exit,
m_dot_f) are extracted only at nodes where propulsion succeeded AND cleared
the M_out safety margin (decided response to the choking fragility: an
M_out > 1.15 NLP constraint keeps the trajectory away from this regime, but
the deck-generation grid sweep runs before any trajectory/constraint exists
and needs its own independent guard). Nodes that fail either check are
repaired by FREEZING to the nearest good neighbor along mach, same aoa row
(mach is the only grid axis here -- equivalence_ratio isn't swept yet, see
decision 13) -- not by extrapolating a slope through sparse, possibly
widely-separated good points, which was tried first and produced visibly
wild, non-monotonic surfaces when good points were sparse (see _repair_grid's
docstring). Both the value AND its sensitivity columns are frozen together,
since patching only the value would leave a smooth-looking deck with
silently wrong gradients baked in.

This is a deliberately scoped-down first step: it makes the *existing* 2-D
(alpha, mach) grid robust to the known failure mode. The full N-D
(alpha, mach, altitude, equivalence_ratio) outer/inner-sweep restructuring
from decision 13 is a separate, larger piece of work, not done here.
"""
import warnings

import casadi as ca
from smtools import Observations

from utilities import get_number
from scramjet_point import evaluate_scramjet_point
from quantity_catalog import CATALOG

# Catalog entries that read from result["propulsion"] or
# result["models"]["engine"/"nozzle"] -- None whenever
# evaluate_scramjet_point's propulsion chain fails. Everything else in
# CATALOG (CD_aero, CL_aero, CD_friction, CL_friction) is independent of it.
PROPULSION_QUANTITIES = {
    "CD_propulsion", "CL_propulsion", "P_out", "T_out", "M_out", "P_exit", "m_dot_f",
}

# Engine outflow Mach below which a propulsion-chain result is treated as
# unreliable even if it didn't raise -- see module docstring. Matches the
# NLP-side M_out > 1.15 constraint with no extra margin subtracted; the gap
# between "converged" (M >= ~1.09 in the residual sweep) and "stalled"
# (M <= ~1.03) is wide enough that this isn't a close call in practice.
M_SAFETY_MARGIN = 1.15


def _casadi_interpolant_from_df(
    parameters: dict,
    df,
    output_name: str,
    mach_grid,
    aoa_grid,
    include_sens: bool = True,
) -> ca.Function:
    """
    Single-output generalization of hyperdynamics.aero_interpolate_from_df's
    core mechanism (bspline value interpolant + first-order Taylor
    correction in the DVs, from each grid point's own dZdp), without the
    hardcoded "CD"/"CL" names and without any plotting -- this runs inside
    an optimisation loop, not interactively (see quantity_catalog.py /
    PROJECT_STATE.md for a standalone interactive-plotting script instead).

    Returns
    ----------
    ca.Function(mach, alpha, *parameters.values()) -> value
    """
    Z_interp = ca.interpolant(
        f"{output_name}_interp", "bspline", [mach_grid, aoa_grid], df[output_name]
    )

    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]
    mach = ca.MX.sym("mach")
    alpha = ca.MX.sym("alpha")
    ca_args = ca.vertcat(mach, alpha)

    Z_p = 0
    if include_sens:
        for i, (p_name, p_val) in enumerate(parameters.items()):
            dZ = ca.interpolant(
                f"d{output_name}_dp{i}", "bspline", [mach_grid, aoa_grid],
                df[f"d{output_name}/d{p_name}"],
            )
            Z_p += (p[i] - p_val) * dZ(ca_args)

    return ca.Function(f"{output_name}_fun", [mach, alpha, *p], [Z_interp(ca_args) + Z_p])


def _repair_grid(raw_q: dict, aoa_grid, mach_grid, quantity_name: str):
    """Fill in missing (aoa, mach) -> (value, dZdp) entries in raw_q
    (mutated in place) for one CATALOG quantity, by FREEZING to the
    nearest good neighbor's value -- not extrapolating a slope through it.

    Originally did linear extrapolation from the two nearest good mach
    points per aoa row. Changed (2026-09-21) after the user spotted a real
    problem with that: near thermal choke, good points can be sparse and
    scattered (point 48's case: 6/16 good), so "two nearest good points in
    this row" can end up being a widely-separated, physically-unrelated
    pair -- the resulting line's slope has no real basis and can extrapolate
    to wild, non-monotonic values, visibly twisting the fitted surface.
    Freezing avoids this by construction: every repaired value is copied
    from an actual computed point, so it can never fall outside the range
    of what was actually observed on the grid, unlike a synthetic linear
    blend/extrapolation. The tradeoff -- flat patches instead of a smoothly
    varying repair -- is the right one specifically because the guarded
    region is a genuine singularity (thermal choke) where no nearby slope
    is trustworthy anyway (see decision 14); flat-and-bounded beats
    smooth-but-potentially-wild here.

    Prefers the nearest good mach value in the SAME aoa row (mach is the
    more direct physical driver of choke margin than aoa, same reasoning
    as the original extrapolation-axis choice); falls back to the nearest
    good node anywhere in the grid (any aoa) if the whole row is bad.
    Raises if literally every node failed -- no data to repair from at all.
    """
    all_good = [(a, m) for a in aoa_grid for m in mach_grid if (a, m) in raw_q]
    if not all_good:
        raise RuntimeError(
            f"generate_scramjet_decks: every grid node failed the propulsion "
            f"guard for '{quantity_name}' -- no data to repair from. This "
            f"geometry likely can't clear M_out > {M_SAFETY_MARGIN} anywhere "
            f"in the swept envelope."
        )

    for aoa in aoa_grid:
        row_good_machs = [m for m in mach_grid if (aoa, m) in raw_q]
        missing_machs = [m for m in mach_grid if (aoa, m) not in raw_q]
        if not missing_machs:
            continue

        if row_good_machs:
            for m in missing_machs:
                nearest_m = min(row_good_machs, key=lambda gm: abs(gm - m))
                raw_q[(aoa, m)] = raw_q[(aoa, nearest_m)]
        else:
            # Whole aoa row bad -- nearest good node anywhere in the grid.
            for m in missing_machs:
                nearest = min(all_good, key=lambda am: (am[0] - aoa) ** 2 + (am[1] - m) ** 2)
                raw_q[(aoa, m)] = raw_q[nearest]


def generate_scramjet_decks(
    vehicle,
    fws: dict,
    context,
    overrides: dict,
    nozzle_surrogate,
    cf_model,
    aoa_grid,
    mach_grid,
    altitude: float,
    equivalence_ratio: float,
    verbosity: int = 0,
    repair_log: list | None = None,
    raw_data_out: dict | None = None,
) -> dict:
    """
    Arguments
    ----------
    vehicle, fws, context, overrides
        As returned by geometry_generation.generate_scramjet_sj2d() -- held
        fixed across the whole sweep. `fws`'s keys/order are the single
        source of truth for the DV names/order used everywhere below (the
        outer-loop active-DV dict, not passed separately, to avoid a second
        copy that could drift out of sync with `context`).
    nozzle_surrogate, cf_model
        Loaded once by the caller (e.g. MultistageManager.__init__).
    aoa_grid, mach_grid
        Grid points to sweep, in rad / Mach number. aoa is the outer loop,
        mach the inner loop -- mach must be the fastest-varying axis in the
        flattened data (same convention as hyperdynamics.aero_interpolate).
    altitude, equivalence_ratio
        Fixed for this first pass (not swept) -- see evaluate_scramjet_point.

    Returns
    ----------
    dict[str, ca.Function], one entry per quantity_catalog.CATALOG name.
    Each ca.Function takes (mach, alpha, *fws.values()) and returns that
    quantity's value, ready to plug into a dynamics function the way
    dynamics.py's trim_3DOF_* use aero_fun["HV"] -- just one output instead
    of two, so callers needing e.g. both CD_aero and CL_aero call the two
    corresponding functions separately rather than unpacking one call.

    Propulsion-dependent entries (see module docstring / PROPULSION_QUANTITIES)
    are repaired by freezing to the nearest good mach-axis neighbor at any
    grid node where the propulsion chain failed or landed too close to
    thermal choke (M_out < M_SAFETY_MARGIN) -- a UserWarning is raised
    listing how many nodes needed repair, but the call still succeeds and
    returns a complete, usable deck.

    repair_log
        Optional -- pass a list to have it extended (not replaced) with one
        (aoa, mach, reason) tuple per repaired grid node, in addition to the
        UserWarning. None (default) skips this; existing callers that don't
        pass it see no change in behavior. For diagnostics/plotting (e.g.
        deck_smoketest.py/deck_gui.py coloring repaired training points
        differently) -- every propulsion-dependent CATALOG entry shares the
        same bad/good status per node (propulsion either fails as a whole at
        a node or it doesn't), so this one list covers all 7 of them.
    raw_data_out
        Optional -- pass a dict to have it populated (not replaced) with
        {catalog_name: {(aoa, mach): (value, dZdp)}} for every grid node,
        AFTER repair -- i.e. exactly the data that was fed into each
        deck's interpolant, not a re-derivation. None (default) skips this.
        Added (2026-09-21) after deck_smoketest.py/deck_gui.py were caught
        showing the pre-repair raw physics value at repaired nodes instead
        of the actual frozen value the deck uses -- an independent second
        evaluate_scramjet_point pass in those scripts can't know what got
        repaired to what, only this function can, since only it does the
        repairing. Use this instead of re-evaluating physics independently
        whenever you need to know exactly what a deck was built from.
    """
    param_names = list(fws.keys())
    p_values = [float(v) for v in fws.values()]
    n_params = len(param_names)
    # The interpolant builder needs plain floats (it does CasADi arithmetic
    # against these and later calls the compiled function with them) --
    # fws itself holds FloatWithSens, not usable here directly.
    plain_dv_dict = dict(zip(param_names, p_values))

    # Geometry/mesh is fixed across the whole sweep -- compute once.
    S_ref = vehicle.mesh_data.calculate_volume() ** (2 / 3)

    # ---- Pass 1: evaluate the grid. Propulsion-dependent quantities are
    # left out of `raw` at bad nodes (hard failure or near-choke), repaired
    # in pass 2 below; the 4 propulsion-independent quantities are always
    # extracted (evaluate_scramjet_point guarantees aero/friction survive a
    # propulsion failure -- see scramjet_point.py / decision 14).
    raw = {name: {} for name in CATALOG}
    bad_nodes = []
    for aoa in aoa_grid:
        for mach in mach_grid:
            result = evaluate_scramjet_point(
                alpha=aoa,
                mach=mach,
                altitude=altitude,
                equivalence_ratio=equivalence_ratio,
                vehicle=vehicle,
                fws=fws,
                context=context,
                overrides=overrides,
                nozzle_surrogate=nozzle_surrogate,
                cf_model=cf_model,
                S_ref=S_ref,
                verbosity=verbosity,
            )

            propulsion_bad = result["propulsion_error"] is not None
            bad_reason = result["propulsion_error"]
            if not propulsion_bad:
                m_out = get_number(result["models"]["engine"].outflow.mach)
                if m_out < M_SAFETY_MARGIN:
                    propulsion_bad = True
                    bad_reason = f"M_out={m_out:.4f} < {M_SAFETY_MARGIN} (thermal-choke margin)"
            if propulsion_bad:
                bad_nodes.append((aoa, mach, bad_reason))

            for name, quantity in CATALOG.items():
                if name in PROPULSION_QUANTITIES and propulsion_bad:
                    continue  # repaired in pass 2
                value = quantity.extract(result, aoa)
                dvalue_dp = list(value.sens) if hasattr(value, "sens") else [0.0] * n_params
                raw[name][(aoa, mach)] = (float(value), dvalue_dp)

    # ---- Pass 2: repair propulsion-dependent quantities at bad nodes.
    for name in PROPULSION_QUANTITIES:
        _repair_grid(raw[name], aoa_grid, mach_grid, name)

    if repair_log is not None:
        repair_log.extend(bad_nodes)

    if raw_data_out is not None:
        for name in CATALOG:
            raw_data_out.setdefault(name, {}).update(raw[name])

    if bad_nodes:
        n_total = len(aoa_grid) * len(mach_grid)
        warnings.warn(
            f"generate_scramjet_decks: {len(bad_nodes)}/{n_total} grid nodes "
            f"failed the propulsion guard (near thermal choke or solver "
            f"stall) and were repaired by freezing to the nearest good "
            f"mach-axis neighbor -- "
            f"see PROJECT_STATE.md decision 14. First few: "
            f"{bad_nodes[:3]}"
        )

    # ---- Build Observations from the now-complete (possibly repaired) data.
    decks = {
        name: Observations(
            n_inputs=2,
            n_outputs=1,
            n_params=n_params,
            input_names=["aoa", "mach"],
            output_names=[name],
            param_names=param_names,
        )
        for name in CATALOG
    }
    for aoa in aoa_grid:
        for mach in mach_grid:
            for name in CATALOG:
                value, dvalue_dp = raw[name][(aoa, mach)]
                decks[name].add_observation(x=[aoa, mach], Z=[value], p=p_values, dZdp=[dvalue_dp])

    return {
        name: _casadi_interpolant_from_df(
            plain_dv_dict,
            decks[name].to_dataframe(),
            name,
            mach_grid,
            aoa_grid,
            include_sens=True,
        )
        for name in CATALOG
    }


if __name__ == "__main__":
    pass
