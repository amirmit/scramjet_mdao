"""Merge results_newstack.csv and results_oldstack.csv (long format: one row
per point_id/component/quantity, with `value` and one `sens_<dv_name>`
column per geometry DV) and report value + sensitivity agreement between
the new stack (evaluate_scramjet_point) and HyperMDAO
(Run_ObjFunc.calc_vehicle) for inlet/cowl/engine/nozzle.

Only points that converged on BOTH sides are compared (an inner merge on
point_id/component/quantity naturally drops any point missing from either
side -- see failures_newstack.csv / failures_oldstack.csv for what didn't
converge and why).

Outputs (this directory):
  - merged_values.csv       one row per (point_id, component, quantity):
                             value_new, value_old, abs_diff, pct_diff
  - merged_sensitivities.csv one row per (point_id, component, quantity, dv):
                             sens_new, sens_old, abs_diff, pct_diff
  - printed summary: per (component, quantity) median/max value pct diff,
    per (component, quantity, dv) median/max sensitivity pct diff, and the
    worst-offender rows for each.

USAGE
-----
    source /home/s4679920/MDAO/MDOenv/bin/activate   (or hyenv -- only needs pandas/numpy)
    cd /home/s4679920/MDAO/scramjet_mdao/co_design/hypermdao_100pt_compare
    python compare_report.py
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 60)

KEY = ["point_id", "component", "quantity"]


def pct_diff(new, old, eps=1e-9):
    denom = np.abs(old)
    denom = np.where(denom < eps, np.nan, denom)
    return 100.0 * (new - old) / denom


def main():
    new = pd.read_csv("results_newstack.csv")
    old = pd.read_csv("results_oldstack.csv")

    sens_cols = sorted(c for c in new.columns if c.startswith("sens_"))
    dv_names = [c[len("sens_"):] for c in sens_cols]

    # ---- Values ----
    merged = new[KEY + ["value"]].merge(
        old[KEY + ["value"]], on=KEY, suffixes=("_new", "_old"), how="inner"
    )
    merged["abs_diff"] = merged["value_new"] - merged["value_old"]
    merged["pct_diff"] = pct_diff(merged["value_new"], merged["value_old"])
    merged.to_csv("merged_values.csv", index=False)

    n_new_points = new["point_id"].nunique()
    n_old_points = old["point_id"].nunique()
    n_common_points = merged["point_id"].nunique()

    print("=" * 78)
    print("100-POINT ENGINE-MODEL COMPARISON: evaluate_scramjet_point (new) vs. "
          "Run_ObjFunc.calc_vehicle (HyperMDAO)")
    print("=" * 78)
    print(f"Points converged -- new stack: {n_new_points}/100, old stack: {n_old_points}/100, "
          f"compared (both sides ok): {n_common_points}/100")
    print("-" * 78)
    print("VALUES -- per (component, quantity), over all compared points")
    print("-" * 78)
    val_summary = (
        merged.groupby(["component", "quantity"])["pct_diff"]
        .agg(n="count", median_abs_pct=lambda s: s.abs().median(), max_abs_pct=lambda s: s.abs().max())
        .reset_index()
    )
    print(val_summary.to_string(index=False))

    print("-" * 78)
    print("VALUES -- worst 10 offenders (by |pct_diff|)")
    print("-" * 78)
    worst_val = merged.reindex(merged["pct_diff"].abs().sort_values(ascending=False).index).head(10)
    print(worst_val[KEY + ["value_new", "value_old", "abs_diff", "pct_diff"]].to_string(index=False))

    # ---- Sensitivities ----
    new_sens = new.melt(id_vars=KEY, value_vars=sens_cols, var_name="dv", value_name="sens_new")
    old_sens = old.melt(id_vars=KEY, value_vars=sens_cols, var_name="dv", value_name="sens_old")
    new_sens["dv"] = new_sens["dv"].str[len("sens_"):]
    old_sens["dv"] = old_sens["dv"].str[len("sens_"):]

    smerged = new_sens.merge(old_sens, on=KEY + ["dv"], how="inner")
    smerged = smerged.dropna(subset=["sens_new", "sens_old"])
    smerged["abs_diff"] = smerged["sens_new"] - smerged["sens_old"]
    smerged["pct_diff"] = pct_diff(smerged["sens_new"], smerged["sens_old"])
    smerged.to_csv("merged_sensitivities.csv", index=False)

    print("-" * 78)
    print(f"SENSITIVITIES -- per (component, quantity), over all compared points x {len(dv_names)} geometry DVs")
    print("(equivalence_ratio sensitivity NOT compared -- new stack can't compute it via "
          "evaluate_scramjet_point; see README.md)")
    print("-" * 78)
    sens_summary = (
        smerged.groupby(["component", "quantity"])["pct_diff"]
        .agg(n="count", median_abs_pct=lambda s: s.abs().median(), max_abs_pct=lambda s: s.abs().max())
        .reset_index()
    )
    print(sens_summary.to_string(index=False))

    print("-" * 78)
    print("SENSITIVITIES -- worst 15 offenders (by |pct_diff|, sens_old magnitude > 1e-6 to avoid "
          "near-zero-denominator noise)")
    print("-" * 78)
    ssub = smerged[smerged["sens_old"].abs() > 1e-6]
    worst_sens = ssub.reindex(ssub["pct_diff"].abs().sort_values(ascending=False).index).head(15)
    print(worst_sens[KEY + ["dv", "sens_new", "sens_old", "abs_diff", "pct_diff"]].to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
