# scramjet_mdao

Top-level run repo for the SJ2D scramjet vehicle on the MDAO stack (`hypervehicle2`, `hyperpanel`, `hypergrad`,
`hyperProp`, `hypervisc`, `utilities`). Environment: `source /home/s4679920/MDAO/MDOenv/bin/activate` (the packages are
editable installs from the sibling checkouts; `pyproject.toml` here is not used for the runs).

- `single_point/` — single flight-condition shape optimisation (SNOPT via `pyoptsparse`), a port of HyperMDAO's
  `Surrogate_Opt_v3`. `Run_Single_Point.py` does one evaluation, `Opt_Main.py [case_config.py]` runs an optimisation.
  Status, architecture and validation against the old repo: `single_point/PROJECT_STATE.md`.
- `co_design/` — DV -> geometry -> engine/aero -> CasADi-surrogate pipeline and (in progress) the bi-level co-design loop.
  Status: `co_design/PROJECT_STATE.md`.

`single_point/` reuses `co_design/Scramjet2D.py`, `geometry_generation.py`, `scramjet_point.py` and `quantity_catalog.py`
directly (one copy each). Changing any of them affects both pipelines; each file carries a "SHARED WITH single_point" banner.

The earlier plan of a `vehicle/Eng_Build.py` on the HyperMDAO stack (`hypervehicle`, `pysagas`, `HyperPro`) was never built
here and is superseded. `HyperMDAO/RunFolder_dev/Surrogate_Opt_v3` remains the old reference.

Old-vs-new optimisation comparison (all 16 non-archived old runs): `/home/s4679920/MDAO/Work/opt_comparison/REPORT.md`.
