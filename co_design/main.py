import matplotlib
import os
import numpy as np

from codesignTools.constraint import ConstraintType, Constraint, ConstraintSet
from codesignTools.parameter import Parameter, ParameterSet
from codesignTools.design_point import DesignPoint

# from missions import (
#     FirstStageEnergyGroundLaunch,
#     MaxRangeGroundLaunch,
#     SprintGroundLaunch,
#     MaxRangeObstructionGroundLaunch,
#     TargetGroundLaunch,
#     TargetObstructionGroundLaunch,
#     Target2GroundLaunch,
#     Sprint2GroundLaunch,
# )
# from spaceplane_missions import (
#     BoostSpaceplane,
#     BoostSpaceplaneFlyback,
#     SpaceplaneFlyback,
# )
from glide_missions import Sprint2Glider, Target2Glider, TargetObstructionGlider, MaxRangeObstructionGlider
from multistage_manager import Scramjet2DManager


# This is needed to overwrite TkAgg option in pyslsqp
# which causes multithreading issues
matplotlib.use("TkAgg")
matplotlib.pyplot.ioff()
d2r = np.deg2rad

def main():
    """Load Initial Parameters from Other Solutions"""
    load_params = None

    """scramjer_2D"""
    SJ_type = "sj2D"

    """ Mission """
    # directory_name = "max_range_ground_launch"
    # problem = MaxRangeGroundLaunch(scaling=1e-6)

    directory_name = "target_2_obstruction_glider_L"
    problem = Target2Glider(scaling=1e-1)


    """Define parameters"""
    # DV catalog: {name: {llim, ulim, init, active}}. `active`:
    #   True      -> optimised (goes into paramset, seen by the outer loop)
    #   False     -> held fixed at `init`, passed downstream as an override
    #   "default" -> omitted entirely; the model's own hardcoded default applies
    # Geometry and engine DVs are kept in separate dicts here only so the
    # split is visible at a glance; they still get combined into one FWS
    # call downstream (geometry and engine DVs must share one sensitivity
    # vector/context).
    if SJ_type == "sj2D":

        geom_DV = {
            "inlet_length_1": dict(llim=0.5, ulim=1.2, init=1.1, active=True),
            "inlet_length_2": dict(llim=0.001, ulim=2.0, init=0.3, active=True),
            "inlet_theta_1": dict(llim=d2r(5.0), ulim=d2r(10.0), init=d2r(6.0), active=True),
            "inlet_theta_2": dict(llim=d2r(1.0), ulim=d2r(10.0), init=d2r(5.0), active=True),
            "throat_height": dict(llim=0.05, ulim=0.12, init=0.08, active=True),
            "engine_tilt_ang": dict(llim=d2r(0.0), ulim=d2r(3.0), init=d2r(0.1), active=True),
            "cowl_LE": dict(llim=0.02, ulim=0.3, init=0.1, active=True),
            "engine_length": dict(llim=0.5, ulim=1.5, init=1.0, active=True),
            "isolator_length": dict(llim=0.5, ulim=1.5, init=1.0, active=True),
            "engine_h5_h3": dict(llim=1.0, ulim=1.1, init=1.02, active=True),
            "nozzle_length": dict(llim=0.5, ulim=3.5, init=1.0, active=True),
            "nozzle_angle": dict(llim=d2r(1.0), ulim=d2r(25.0), init=d2r(10.0), active=True),
            "nozzle_mid_frac": dict(llim=0.0, ulim=0.8, init=0.5, active=True),

            # "inlet_length_1": dict(llim=0.5, ulim=1.2, init=0.9777468527824827, active=True),
            # "inlet_length_2": dict(llim=0.001, ulim=2.0, init=0.8783840091497884, active=True),
            # "inlet_theta_1": dict(llim=d2r(5.0), ulim=d2r(10.0), init=0.11992018042912, active=True),
            # "inlet_theta_2": dict(llim=d2r(1.0), ulim=d2r(10.0), init=0.1101421974158474, active=True),
            # "throat_height": dict(llim=0.05, ulim=0.12, init=0.0538150220493907, active=True),
            # "engine_tilt_ang": dict(llim=d2r(0.0), ulim=d2r(3.0), init=0.0395489953647779, active=True),
            # "cowl_LE": dict(llim=0.02, ulim=0.3, init=0.2096404718220312, active=True),
            # "engine_length": dict(llim=0.5, ulim=1.5, init=1.2805811375265843, active=True),
            # "isolator_length": dict(llim=0.5, ulim=1.5, init=0.6493741563681508, active=True),
            # "engine_h5_h3": dict(llim=1.0, ulim=1.1, init=1.0874575139855571, active=True),
            # "nozzle_length": dict(llim=0.5, ulim=3.5, init=3.0818912041883904, active=True),
            # "nozzle_angle": dict(llim=d2r(1.0), ulim=d2r(25.0), init=0.2232307304671963, active=True),
            # "nozzle_mid_frac": dict(llim=0.0, ulim=0.8, init=0.7106923624502725, active=True),


            # "inlet_length_1": dict(llim=0.5, ulim=1.2, init=1.2, active=True),
            # "inlet_length_2": dict(llim=0.001, ulim=2.0, init=0.88, active=True),
            # "inlet_theta_1": dict(llim=d2r(5.0), ulim=d2r(10.0), init=d2r(10.0), active=True),
            # "inlet_theta_2": dict(llim=d2r(1.0), ulim=d2r(10.0), init=d2r(6.93), active=True),
            # "throat_height": dict(llim=0.05, ulim=0.12, init=0.1, active=True),
            # "engine_tilt_ang": dict(llim=d2r(0.0), ulim=d2r(3.0), init=d2r(0), active=True),
            # "cowl_LE": dict(llim=0.02, ulim=0.3, init=0.1, active=False),
            # "engine_length": dict(llim=0.5, ulim=1.5, init=0.5, active=False),
            # "isolator_length": dict(llim=0.5, ulim=1.5, init=0.5, active=True),
            # "engine_h5_h3": dict(llim=1.0, ulim=1.1, init=1.0, active=True),
            # "nozzle_length": dict(llim=0.5, ulim=3.5, init=1.0, active=True),
            # "nozzle_angle": dict(llim=d2r(1.0), ulim=d2r(25.0), init=0.15788088456676624, active=True),
            # "nozzle_mid_frac": dict(llim=0.0, ulim=0.8, init=0.5, active=True),
        }

        # equivalence_ratio deliberately NOT here: it's an NLP/trajectory
        # control (like alpha), not an outer-loop shape DV -- see
        # PROJECT_STATE.md architecture decision 12. It belongs in whatever
        # the trajectory OCP's own controls end up being defined (not yet
        # built), not in this outer-loop shape-DV catalog. Kept as an empty
        # dict (not removed outright) so the geom_DV/eng_DV split stays
        # available for a genuine future engine-side *shape* DV.
        eng_DV = {}

    else:
        raise ValueError(f"sj_type of {SJ_type} not recognised.")

    DV = {**geom_DV, **eng_DV}

    # Active DVs -> what the outer optimiser actually sees.
    paramset = ParameterSet(
        [
            Parameter(name, spec["init"], spec["llim"], spec["ulim"])
            for name, spec in DV.items()
            if spec["active"] is True
        ]
    )

    # Fixed (active=False) DVs -> constants injected downstream at `init`.
    overrides = {name: spec["init"] for name, spec in DV.items() if spec["active"] is False}
    # "default" DVs -> omitted downstream; the model's own default applies.
    default_dvs = [name for name, spec in DV.items() if spec["active"] == "default"]

    if load_params:
        paramset = ParameterSet.from_csv(f"parameter_initialisations/{load_params}_final_params.csv")

    """ Define constraints """
    internal_vol = Constraint("internal_vol", ConstraintType.LOWER, lim=0.81245, scale=1)
    constraints = [internal_vol]

    constset = ConstraintSet(constraints)

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    paramset.to_csv(directory_name + "/parameters.csv")

    veh_props = ["vol", "LD", "stab_HV", "mass_HV", "width", "height", "com_HV"]

    manager = Scramjet2DManager(
        paramset,
        constset,
        problem,
        veh_props=veh_props,
        options={
            "SJ_type": SJ_type,
            "overrides": overrides,
            "default_dvs": default_dvs
        },
    )

    ###############################################################

    design_point = DesignPoint(
        iter_idx=1,
        theta=paramset.init,
        paramset=paramset,
        prop_names=veh_props,
    )
    manager.solve(design_point)


    ##############################################################








    manager.cold_start(directory_name, force=True, acc=1e-4)
    # manager.warmstart(directory_name, acc=1e-4)
    # manager.eval_from_solution("./sprint-final/sqp_res.csv", "sprint_test_on_obs")


if __name__ == "__main__":
    main()
