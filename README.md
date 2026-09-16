# scramjet-mdo

Top-level run repo for the scramjet waverider vehicle analysis tool, split into
two drivers that will diverge over time:

- `single_point/` — single flight-condition shape optimization (SNOPT via
  `pyoptsparse`), the direct continuation of `Opt_Main.py` /
  `Run_ObjFunc.py` / `Run_SensFunc.py` as they exist today.
- `co_design/` — starting point for the bi-level co-design loop (outer shape
  optimization + inner CasADi trajectory optimization), modelled on Jack's
  `spaceplanes-2026` framework. This will be refactored into an
  `update_models()`-style loop; it is not that yet.

Both folders currently hold an **identical, unmodified copy** of the vehicle
analysis chain from `HyperMDAO/RunFolder_dev/Surrogate_Opt_v3` (copied
2026-09-16), kept as the known-working reference/check file for verifying
model updates don't silently break things. `Eng_Build.py` is the geometry +
engine model (waverider fuselage, inlet/cowl/isolator/engine/nozzle,
`Nozzle_2D_Surrogate`); `Run_ObjFunc.py`/`Run_SensFunc.py` evaluate it at one
fixed freestream condition (from `case_config.py`) and return objective/
constraint values and their design-parameter sensitivities; `Opt_Main.py`
drives SNOPT over that.

## Dependencies: one copy per library, path-referenced

Per the "no duplicate copies" decision, `pyproject.toml` points `uv` at the
existing on-disk checkouts rather than vendoring anything:

| Import | Location |
|---|---|
| `hypervehicle` (old API, not hypervehicle2) | `~/HyperMDAO/hyperVehicle` |
| `pysagas` (dist name `hypysagas`) | `~/HyperMDAO/pysagas-dev` |
| `hypervisc` | `~/HyperMDAO/HyperVisc` |
| `HyperPro` | `~/HyperMDAO/HyperPro` |

These four are **not** managed elsewhere in `MDAO/` — they only exist under
`HyperMDAO/`, since this vehicle analysis predates the hypervehicle2/hyperpanel
migration. `hypervehicle2`/`hyperpanel`/`hypergrad`/`hyperDynamics` etc. live
in `MDAO/` and aren't dependencies of this copy yet — that's what the planned
model update (geometry port + engine surrogate wiring) will change.

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

## Next steps (separate conversation)

- Verify this baseline actually runs (single point, in `single_point/`).
- Port the geometry (`Eng_Build.py`) and engine/nozzle model onto
  hypervehicle2 + hyperpanel + the `Nozzle_2D_Surrogate` approach, replacing
  the old `hypervehicle`/`pysagas`/`hypervisc` stack.
- Reconcile the hyperpanel branch divergence (`AM_scramjet2D` vs `main`) —
  prerequisite for the above.
- Build out `co_design/` into an `update_models()` + mission/`trim_3DOF_*`
  loop against a to-be-defined flight envelope (mach/alpha/equivalence
  ratio/dynamic pressure — not yet fixed).
