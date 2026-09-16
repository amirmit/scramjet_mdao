# scramjet_mdao

Top-level run repo for the scramjet waverider vehicle analysis tool.

- `vehicle/` — the shared vehicle definition. Currently just `Eng_Build.py`
  (geometry + engine model: waverider fuselage, inlet/cowl/isolator/engine/
  nozzle, `Nozzle_2D_Surrogate`), copied unmodified from
  `HyperMDAO/RunFolder_dev/Surrogate_Opt_v3` on 2026-09-16 as the known-working
  reference/check file for verifying future model updates don't silently break
  things. **Both drivers below import from here — there is exactly one copy.**
- `single_point/` — single flight-condition shape optimization (SNOPT via
  `pyoptsparse`), the direct continuation of `Opt_Main.py` /
  `Run_ObjFunc.py` / `Run_SensFunc.py` as they exist today. `Run_ObjFunc.py`
  evaluates `vehicle/Eng_Build.py` at one fixed freestream condition (from
  `case_config.py`) and returns objective/constraint values and their
  design-parameter sensitivities; `Opt_Main.py` drives SNOPT over that.
- `co_design/` — not yet built. Will hold the bi-level co-design loop (outer
  shape optimization + inner CasADi trajectory optimization), modelled on
  Jack's `spaceplanes-2026` framework, importing the same `vehicle/Eng_Build.py`
  rather than its own copy.

## Dependencies: one copy per library, path-referenced

Per the "no duplicate copies" decision, `pyproject.toml` points `uv` at the
existing on-disk checkouts rather than vendoring anything:

| Import | Location |
|---|---|
| `hypervehicle` (old API, not hypervehicle2) | `~/HyperMDAO/hyperVehicle` |
| `pysagas` (dist name `hypysagas`) | `~/HyperMDAO/pysagas-dev` |
| `hypervisc` | `~/HyperMDAO/HyperVisc` |
| `HyperPro` | `~/HyperMDAO/HyperPro` |

## Reconciling with MDAO's current stack

`vehicle/Eng_Build.py` and `single_point/Run_ObjFunc.py` still import the
**old** stack by name (`hypervehicle.generator.Generator`,
`hypervehicle.components.swept.SweptComponent`, `pysagas.flow.FlowState`,
`pysagas.cfd.OPMVec`, `hypervisc`). Pointing `pyproject.toml` at MDAO's
`hyperpanel`/`hypervehicle2`/`hypergrad` instead would **not** make this run —
those packages have different module paths and classes entirely (confirmed by
reading both APIs), so the imports would just fail differently, not resolve.

Making this actually use MDAO's current stack means porting the code itself:
- `vehicle/Eng_Build.py`'s geometry onto hypervehicle2's `Component`/`Vehicle`
  + `add_patch_generator` pattern (the way `MDAO/scramjet2D/SJ_run/Scramjet2D.py`
  already does for the same vehicle family — see `PROJECT_STATE.md`).
- `Run_ObjFunc.py`'s panel/friction-drag calls onto `hyperpanel.PanelSolver` +
  `hypergrad`, following the same per-tag `PanelSolver` pattern
  `scramjet2D/SJ_run`'s `Force_Calculator` already demonstrates.
- The nozzle force already goes through `HyperPro.Nozzle_2D_Surrogate`, which
  doesn't depend on the mesh at all — that part should carry over with fewer
  changes than the panel-method geometry/aero.

Until that port happens, `pyproject.toml` deliberately keeps pointing at the
old stack, because that's what makes `vehicle/Eng_Build.py` actually
importable right now. Swapping just the dependency names without touching the
code would break every import outright.

## Known gaps (not yet resolved, flagged rather than fixed)

- **Not installed/run yet.** `uv sync` has not been attempted here — no conda
  env with `pythonocc`/`pyoptsparse`/SNOPT was available in this session.
  Treat the dependency table above as traced-from-imports, not verified.
- `problem_catalog.py` has a stray `from sphinx.addnodes import desc_type`
  import (line 26) that looks unintentional (Sphinx is a docs tool, unrelated
  to this code) and will likely break import if `sphinx` isn't installed.
- The four dependency repos above have local, uncommitted or unpushed state:
  `hyperVehicle` is on branch `AM_RootFind` (1 ahead of origin), `pysagas-dev`
  is on `AM_Surrogate_Nozzle`, `HyperPro` has an uncommitted modified
  `nozzle_surrogate.py`. Worth confirming these are the versions actually
  intended before treating a run as a real baseline check.
- There are other copies of `hypervehicle`/`pysagas`/`HyperPro` under
  `~/repos/` and `~/HV2/` not used here — picked the `HyperMDAO/`-colocated
  ones as most likely to match what `Surrogate_Opt_v3` was last run against,
  but not confirmed.
- `single_point/Run_ObjFunc.py` reaches `vehicle/Eng_Build.py` via a
  `sys.path.insert` at import time (see top of the file), not a proper
  installed package. Simplest option that needed no build-system changes;
  revisit if it gets awkward once `co_design/` also needs it.

## Next steps (separate conversation)

- Verify this baseline actually runs (single point, in `single_point/`).
- Port `vehicle/Eng_Build.py` and the engine/nozzle model onto
  hypervehicle2 + hyperpanel + the `Nozzle_2D_Surrogate` approach, replacing
  the old `hypervehicle`/`pysagas`/`hypervisc` stack (see "Reconciling with
  MDAO's current stack" above).
- Reconcile the hyperpanel branch divergence (`AM_scramjet2D` vs `main`) —
  prerequisite for the above.
- Build out `co_design/` into an `update_models()` + mission/`trim_3DOF_*`
  loop against a to-be-defined flight envelope (mach/alpha/equivalence
  ratio/dynamic pressure — not yet fixed).
