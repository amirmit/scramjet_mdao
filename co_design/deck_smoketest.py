"""Interactive smoke-test for a fitted scramjet_deck.py CasADi deck.

Not part of the production pipeline -- a standalone visual sanity check.
Plots, for one chosen CATALOG quantity:
  1. The fitted deck's VALUE surface over a fine (mach, alpha) mesh, with
     the actual coarse training-grid points scattered on top (red dots),
     so you can see by eye whether the bspline fit tracks the real physics
     reasonably between grid nodes.
  2. The same treatment for that quantity's SENSITIVITY w.r.t. one chosen
     geometry DV (dQuantity/dp_i as a function of (mach, alpha)) against its
     own dZdp training points -- this is the per-DV parameter sensitivity
     (scramjet_deck.py's Taylor-correction term), not the grid-axis
     derivative (dQuantity/dMach or dQuantity/dalpha) that a function like
     hyperdynamics' plot_aero_sens already covers.

The sensitivity surface is recovered by finite-differencing the fitted
ca.Function itself w.r.t. the chosen DV, holding (mach, alpha) fixed --
this is exact (not just approximate), not merely convenient: by
construction (_casadi_interpolant_from_df's Taylor correction), the fitted
function is EXACTLY LINEAR in each DV, so any nonzero step size recovers
the true per-DV sensitivity interpolant with no truncation error.

Edit the CONFIGURATION block below to explore a different quantity, DV,
geometry, or grid. Needs a real interactive matplotlib backend (TkAgg) and
an actual display -- run it yourself in an interactive session, not from a
headless/background one.

USAGE
-----
    source /home/s4679920/MDAO/MDOenv/bin/activate
    cd /home/s4679920/MDAO/scramjet_mdao/co_design
    python deck_smoketest.py
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from geometry_generation import generate_scramjet_sj2d
from scramjet_deck import generate_scramjet_decks, PROPULSION_QUANTITIES
from utilities import get_number

from hyperProp.Nozzle.nozzle_surrogate_model import load_surrogate_model
from hypervisc import get_friction_model

# ======================= CONFIGURATION =======================
QUANTITY = "P_out"          # any quantity_catalog.CATALOG name
SENS_DV = "throat_height"   # any of ACTIVE_DV_DICT's keys below

d2r = np.deg2rad
# The validated "small_test" DV set (PROJECT_STATE.md) by default.
ACTIVE_DV_DICT = {
    "inlet_length_1": 1.1, "inlet_length_2": 0.3,
    "inlet_theta_1": d2r(6.0), "inlet_theta_2": d2r(5.0),
    "throat_height": 0.08, "engine_tilt_ang": d2r(0.1),
    "cowl_LE": 0.1, "engine_length": 1.0, "isolator_length": 1.0,
    "engine_h5_h3": 1.02, "nozzle_length": 1.0,
    "nozzle_angle": d2r(10.0), "nozzle_mid_frac": 0.5,
}
EQUIVALENCE_RATIO = 0.2
ALTITUDE = 30e3
AOA_GRID = np.deg2rad([0, 2, 4, 6])
MACH_GRID = [6.0, 6.5, 7.0, 7.5]
FINE_N = 60  # fine-mesh resolution per axis for the plotted surface
# ================================================================


def main():
    geom_out = generate_scramjet_sj2d(
        active_dv_dict=ACTIVE_DV_DICT, overrides={}, mesh_params=None, save=False
    )
    vehicle, fws, context = geom_out["vehicle"], geom_out["FWS_param_dict"], geom_out["context"]
    dv_names = list(fws.keys())
    dv_values = [get_number(v) for v in fws.values()]
    dv_idx = dv_names.index(SENS_DV)

    nozzle_surrogate = load_surrogate_model("v_3_1")
    cf_model = get_friction_model("constant", cf_value=0.001)

    print(f"Building deck ({QUANTITY}) over the {len(AOA_GRID)}x{len(MACH_GRID)} grid...")
    repair_log = []
    raw_data = {}
    decks = generate_scramjet_decks(
        vehicle=vehicle, fws=fws, context=context, overrides={},
        nozzle_surrogate=nozzle_surrogate, cf_model=cf_model,
        aoa_grid=AOA_GRID, mach_grid=MACH_GRID,
        altitude=ALTITUDE, equivalence_ratio=EQUIVALENCE_RATIO,
        repair_log=repair_log, raw_data_out=raw_data,
    )
    repaired_nodes = {(aoa, mach) for aoa, mach, _reason in repair_log}
    is_propulsion_q = QUANTITY in PROPULSION_QUANTITIES
    fn = decks[QUANTITY]

    # ---- Fine mesh: fitted deck's value and (via finite-difference w.r.t.
    # the chosen DV, exact per the module docstring) sensitivity surface.
    mach_fine = np.linspace(min(MACH_GRID), max(MACH_GRID), FINE_N)
    aoa_fine = np.linspace(min(AOA_GRID), max(AOA_GRID), FINE_N)
    MACH_MESH, AOA_MESH = np.meshgrid(mach_fine, aoa_fine)
    VALUE_MESH = np.zeros_like(MACH_MESH)
    SENS_MESH = np.zeros_like(MACH_MESH)

    h = 1e-3 * max(abs(dv_values[dv_idx]), 1.0)
    dv_plus = list(dv_values); dv_plus[dv_idx] += h
    dv_minus = list(dv_values); dv_minus[dv_idx] -= h
    for i in range(MACH_MESH.shape[0]):
        for j in range(MACH_MESH.shape[1]):
            m, a = float(MACH_MESH[i, j]), float(AOA_MESH[i, j])
            VALUE_MESH[i, j] = float(fn(m, a, *dv_values))
            SENS_MESH[i, j] = (float(fn(m, a, *dv_plus)) - float(fn(m, a, *dv_minus))) / (2 * h)

    # ---- Coarse training grid: read straight from raw_data_out -- exactly
    # what generate_scramjet_decks fed into this deck's interpolant, AFTER
    # repair. NOT a fresh evaluate_scramjet_point call: an independent pass
    # can't know what got frozen to what (only generate_scramjet_decks does
    # the repairing), and a prior version of this script did exactly that,
    # showing the pre-repair raw (unreliable) value at repaired nodes
    # instead of what the deck actually uses -- caught by the user noticing
    # orange points that didn't coincide with their source neighbor
    # (2026-09-21).
    direct_mach, direct_aoa, direct_val, direct_sens = [], [], [], []
    rep_mach, rep_aoa, rep_val, rep_sens = [], [], [], []
    for (aoa, mach), (value, sens) in raw_data[QUANTITY].items():
        s = sens[dv_idx]
        if is_propulsion_q and (aoa, mach) in repaired_nodes:
            rep_mach.append(mach); rep_aoa.append(aoa); rep_val.append(value); rep_sens.append(s)
        else:
            direct_mach.append(mach); direct_aoa.append(aoa); direct_val.append(value); direct_sens.append(s)

    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(MACH_MESH, AOA_MESH, VALUE_MESH, cmap="viridis", alpha=0.7, edgecolor="none")
    if direct_mach:
        ax1.scatter(direct_mach, direct_aoa, direct_val, color="red", s=50, label="training (direct)")
    if rep_mach:
        ax1.scatter(rep_mach, rep_aoa, rep_val, color="orange", s=50, label="training (repaired/frozen)")
    ax1.set_xlabel("Mach"); ax1.set_ylabel("alpha [rad]"); ax1.set_zlabel(QUANTITY)
    ax1.set_title(f"{QUANTITY}: fitted deck vs. training grid")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(MACH_MESH, AOA_MESH, SENS_MESH, cmap="plasma", alpha=0.7, edgecolor="none")
    if direct_mach:
        ax2.scatter(direct_mach, direct_aoa, direct_sens, color="red", s=50, label="training (direct)")
    if rep_mach:
        ax2.scatter(rep_mach, rep_aoa, rep_sens, color="orange", s=50, label="training (repaired/frozen)")
    ax2.set_xlabel("Mach"); ax2.set_ylabel("alpha [rad]"); ax2.set_zlabel(f"d{QUANTITY}/d{SENS_DV}")
    ax2.set_title(f"d{QUANTITY}/d{SENS_DV}: fitted deck vs. training dZdp")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
