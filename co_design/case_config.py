"""Loads and validates a case config: a plain .py file under cases/ that
Opt_Main.py takes as its sole CLI argument (python Opt_Main.py cases/x.py).

Validation exists because two things downstream are completely permissive
and were found, during this refactor, to fail silently rather than loudly:
Eng_Build.Parametric_EngModel's **kwargs path setattrs any key it's given
(Eng_Build.py:19-65 -> hypervehicle/generator.py:22-26), and dvdp() will
happily hand a FloatWithSens seed vector to a name that nothing reads
(hypervehicle/generator.py:28-37) -- a typo'd or non-existent DV name gets
an all-zero gradient column instead of an error. A config file makes this
more likely (no IDE autocomplete on dict keys), so it's checked here.
"""

import importlib.util
import os
from pathlib import Path

import HyperPro.nozzle_surrogate

from hypervisc.registry import _MODEL_CLASSES as _CF_MODEL_CLASSES

from problem_catalog import CATALOG

# Resolved the same way Run_ObjFunc.get_surrogate() resolves it, but via a
# lightweight import (just HyperPro.nozzle_surrogate, not the full
# Eng_Build/hypervehicle/pysagas stack) so validating a config stays cheap.
_SURROGATE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(HyperPro.nozzle_surrogate.__file__))),
    "SurrogateGen",
    "nozzle_v3",
)

# The exact parameter names Eng_Build.Parametric_EngModel actually reads
# (create_vehicle_geom / create_inlet / create_cowl / create_engine /
# create_nozzle -- see the refactor plan's exploration notes). Kept as an
# explicit list rather than introspected, since introspecting **kwargs
# consumption from setattr calls isn't reliable.
VALID_DV_NAMES = frozenset(
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
        "equivalence_ratio",
    }
)


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

    DV = getattr(cfg, "DV", None)
    if not DV:
        errors.append("DV is required and must define at least one design variable.")
    else:
        for name, spec_ in DV.items():
            if name not in VALID_DV_NAMES:
                errors.append(
                    f"DV '{name}' is not a recognised Eng_Build parameter. "
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
        if name not in CATALOG:
            errors.append(
                f"constraint '{name}' is not in problem_catalog.CATALOG. "
                f"Valid names: {sorted(CATALOG)}"
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
            if name not in CATALOG:
                errors.append(
                    f"objective term '{name}' is not in problem_catalog.CATALOG. "
                    f"Valid names: {sorted(CATALOG)}"
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
        errors.append("nozzle_surrogate is required (e.g. \"v3_1\" or \"v3_2\").")
    else:
        version_dir = os.path.join(_SURROGATE_ROOT, version)
        cfg_path = os.path.join(version_dir, "config.py")
        pkl_path = os.path.join(version_dir, "nozzle_v3_surrogate.pkl")
        if not os.path.isfile(cfg_path):
            errors.append(f"nozzle_surrogate '{version}': no config.py at {cfg_path}")
        if not os.path.isfile(pkl_path):
            errors.append(
                f"nozzle_surrogate '{version}': no nozzle_v3_surrogate.pkl at {pkl_path}"
            )

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

    if not getattr(cfg, "snopt_options", None):
        errors.append("snopt_options is required.")

    if errors:
        raise CaseConfigError(
            f"Case config '{path}' failed validation:\n  - " + "\n  - ".join(errors)
        )
