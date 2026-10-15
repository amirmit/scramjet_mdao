"""Generate the 100 shared random points for the new-stack-vs-HyperMDAO
engine-model comparison (inlet/cowl/engine/nozzle).

Both the new-stack runner (run_newstack.py, this repo/MDOenv) and the
HyperMDAO-side runner (HyperMDAO/RunFolder_dev/scramjet2D_comparison/
run_100pt_compare.py, hyenv) read the SAME points.csv produced here, so
every point is evaluated by both stacks with identical inputs.

Sampling (per user decision, 2026-09-20):
  - 13 geometry DVs + equivalence_ratio: independent uniform draws over
    main.py's own DV bounds (co_design/main.py's geom_DV/eng_DV dicts,
    copied below -- not imported, since main.py isn't import-safe as a
    module, being a `def main(): ...` script with matplotlib.use() calls).
  - alpha (AoA): uniform [0, 6] deg -- PROJECT_STATE.md's validated grid
    range.
  - mach: fixed 6.5 (user explicitly pinned this rather than a range).
  - altitude: fixed 30000 m -- matches the nozzle_v_3_1 surrogate's own
    reference condition and PROJECT_STATE.md's validated grid.

Sampling the full DV bounds independently (not correlated) can and does
produce some geometrically/physically invalid combinations (e.g. a long
nozzle_angle with a short nozzle_length) -- expected; both runners handle
per-point failures without aborting the batch, and the comparison report
flags them.
"""
import numpy as np
import pandas as pd

SEED = 42
N_POINTS = 100

# Copied from co_design/main.py's geom_DV / eng_DV dicts (llim/ulim only).
DV_BOUNDS = {
    "inlet_length_1": (0.5, 1.2),
    "inlet_length_2": (0.001, 2.0),
    "inlet_theta_1": (np.deg2rad(5.0), np.deg2rad(10.0)),
    "inlet_theta_2": (np.deg2rad(1.0), np.deg2rad(10.0)),
    "throat_height": (0.05, 0.12),
    "engine_tilt_ang": (np.deg2rad(0.0), np.deg2rad(3.0)),
    "cowl_LE": (0.02, 0.3),
    "engine_length": (0.5, 1.5),
    "isolator_length": (0.5, 1.5),
    "engine_h5_h3": (1.0, 1.1),
    "nozzle_length": (0.5, 3.5),
    "nozzle_angle": (np.deg2rad(1.0), np.deg2rad(25.0)),
    "nozzle_mid_frac": (0.0, 0.8),
    "equivalence_ratio": (0.1, 1.0),
}

ALPHA_BOUNDS_DEG = (0.0, 6.0)
MACH = 6.5
ALTITUDE_M = 30_000.0


def generate(n_points: int = N_POINTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = {"point_id": np.arange(n_points)}
    for name, (lo, hi) in DV_BOUNDS.items():
        data[name] = rng.uniform(lo, hi, size=n_points)
    data["alpha_deg"] = rng.uniform(*ALPHA_BOUNDS_DEG, size=n_points)
    data["mach"] = np.full(n_points, MACH)
    data["altitude_m"] = np.full(n_points, ALTITUDE_M)

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate()
    out_path = "points.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} points to {out_path}")
    print(df.describe())
