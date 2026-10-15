"""Loads and validates a case config: a plain .py file that Opt_Main.py takes
as its sole CLI argument (python Opt_Main.py opt_input.py).

Validation runs ONCE at startup (Opt_Main calls load_case_config at module
level), not per optimisation iteration. It exists because the pipeline is
completely permissive about names and fails silently rather than loudly:
Parametric_Scramjet2D sets any key it is given as an attribute, and a
typo'd or non-existent DV name just gets an all-zero gradient column instead
of an error. A config file makes this more likely (no IDE autocomplete on
dict keys), so it's checked here.

SHARED WITH co_design: the valid geometry DV names come from
co_design/geometry_generation.py's SJ2D_GEOM_PARAM_NAMES (single source of
truth) -- changing that set changes what a single-point case may name.
"""

import importlib.util
import os
import sys
from pathlib import Path

import hyperProp.Nozzle.models as _nozzle_models
from hypervisc.registry import _MODEL_CLASSES as _CF_MODEL_CLASSES

# # geometry_generation.py lives in the sibling co_design/ folder (shared code,
# # see the docstring above).
# _CO_DESIGN = str(Path(__file__).resolve().parents[1] / "co_design")
# if _CO_DESIGN not in sys.path:
#     sys.path.insert(0, _CO_DESIGN)

from geometry_generation import SJ2D_GEOM_PARAM_NAMES
from problem_catalog import CATALOG

# The 13 geometry DVs plus equivalence_ratio (an engine-side DV, promoted to
# FWS together with the geometry ones in single-point -- in co-design it is a
# trajectory control instead). bez_start_ratio is a geometry default that is
# never a DV.
VALID_DV_NAMES = (SJ2D_GEOM_PARAM_NAMES - {"bez_start_ratio"}) | {"equivalence_ratio"}

# Every named scalar Run_ObjFunc.calc_vehicle returns -- the names an
# objective/constraint may use (single source: problem_catalog.CATALOG).
VALID_QUANTITY_NAMES = frozenset(CATALOG)


class CaseConfigError(ValueError):
    """Raised when a case config file fails validation."""


def load_case_config(path):
    """Load the case config module at `path` and validate it. Returns the
    module so callers read case_name/DV/constraints/opt_params/
    snopt_options as plain attributes."""

    path = Path(path)
    if not path.is_file():
        raise CaseConfigError(f"Case config not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _validate(module, path)
    return module


def _validate(cfg, path):
    errors = []

    if not getattr(cfg, "case_name", None):
        errors.append("case_name is required and must be non-empty.")

    output_dir = getattr(cfg, "output_dir", None)
    if output_dir is not None and (not isinstance(output_dir, (str, os.PathLike)) or not str(output_dir).strip()):
        errors.append("output_dir, if set, must be a non-empty path string (relative paths are relative to the launch directory).")

    DV = getattr(cfg, "DV", None)
    if not DV:
        errors.append("DV is required and must define at least one design variable.")
    else:
        for name, spec_ in DV.items():
            if name not in VALID_DV_NAMES:
                errors.append(
                    f"DV '{name}' is not a recognised SJ2D design variable. "
                    f"Valid names: {sorted(VALID_DV_NAMES)}"
                )
                continue
            flb, fub, x0 = spec_.get("flb"), spec_.get("fub"), spec_.get("x0")
            if flb is None or fub is None or x0 is None:
                errors.append(f"DV '{name}' must define flb, fub, and x0.")
                continue
            if not (flb < fub):
                errors.append(f"DV '{name}': flb ({flb}) must be < fub ({fub}).")
            if not (flb <= x0 <= fub):
                errors.append(f"DV '{name}': x0 ({x0}) must be within [flb, fub].")
            active = spec_.get("active", True)
            if active not in (True, False, "default"):
                errors.append(
                    f"DV '{name}': active must be True, False, or \"default\" "
                    f"(got {active!r})."
                )

    constraints = getattr(cfg, "constraints", None) or {}
    for name, spec_ in constraints.items():
        if name not in VALID_QUANTITY_NAMES:
            errors.append(
                f"constraint '{name}' is not a Run_ObjFunc quantity. "
                f"Valid names: {sorted(VALID_QUANTITY_NAMES)}"
            )
            continue
        glb, gub = spec_.get("glb"), spec_.get("gub")
        if glb is None and gub is None:
            errors.append(f"constraint '{name}' must set glb and/or gub.")
        if glb is not None and gub is not None and not (glb <= gub):
            errors.append(f"constraint '{name}': glb ({glb}) must be <= gub ({gub}).")
        if spec_.get("scale", 1.0) == 0:
            errors.append(f"constraint '{name}': scale must not be 0.")

    objective = getattr(cfg, "objective", None)
    if not objective:
        errors.append("objective is required and must define at least one term.")
    else:
        for name, spec_ in objective.items():
            if name not in VALID_QUANTITY_NAMES:
                errors.append(
                    f"objective term '{name}' is not a Run_ObjFunc quantity. "
                    f"Valid names: {sorted(VALID_QUANTITY_NAMES)}"
                )
                continue
            if spec_.get("scale", 1.0) == 0:
                errors.append(f"objective term '{name}': scale must not be 0.")
        if len(objective) > 1 and not getattr(cfg, "objective_name", None):
            errors.append(
                "objective_name is required when `objective` defines more than "
                "one term."
            )

    version = getattr(cfg, "nozzle_surrogate", None)
    if not version:
        errors.append("nozzle_surrogate is required (e.g. \"v_3_1\" or \"v_3_2\").")
    else:
        models_dir = Path(next(iter(_nozzle_models.__path__)))
        cfg_path = models_dir / f"config_{version}.py"
        pkl_path = models_dir / f"{version}.pkl"
        if not cfg_path.is_file():
            errors.append(f"nozzle_surrogate '{version}': no config at {cfg_path}")
        if not pkl_path.is_file():
            errors.append(f"nozzle_surrogate '{version}': no model pickle at {pkl_path}")

    opt_params = getattr(cfg, "opt_params", None)
    if not opt_params:
        errors.append("opt_params is required.")
    else:
        cf_model = opt_params.get("cf_model", "constant")
        if cf_model not in _CF_MODEL_CLASSES:
            errors.append(
                f"opt_params['cf_model'] '{cf_model}' is not registered in "
                f"hypervisc. Available: {sorted(_CF_MODEL_CLASSES)}"
            )
        for flag_name in ("engine_visc_flg", "aero_visc_flg"):
            if flag_name in opt_params and not isinstance(opt_params[flag_name], bool):
                errors.append(f"opt_params['{flag_name}'] must be True or False.")
        if "package_logs" in opt_params and not isinstance(opt_params["package_logs"], bool):
            errors.append("opt_params['package_logs'] must be True or False.")
        if "save_geometry" in opt_params and not isinstance(opt_params["save_geometry"], bool):
            errors.append("opt_params['save_geometry'] must be True or False.")
        mesh_params = opt_params.get("mesh_params")
        if mesh_params is not None and not isinstance(mesh_params, dict):
            errors.append("opt_params['mesh_params'] must be a dict or None.")

    if not getattr(cfg, "snopt_options", None):
        errors.append("snopt_options is required.")

    if errors:
        raise CaseConfigError(
            f"Case config '{path}' failed validation:\n  - " + "\n  - ".join(errors)
        )
