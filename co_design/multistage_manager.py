from codesignTools import logger
from codesignTools.optimisation_manager import OptManager
from codesignTools.design_point import DesignPoint
# from solve_traj import solve_trajectory
from geometry_generation import generate_scramjet_sj2d
from scramjet_deck import generate_scramjet_decks

from hyperProp.Nozzle.nozzle_surrogate_model import load_surrogate_model
from hypervisc import get_friction_model
from scramjet_point import evaluate_scramjet_point
import os
import numpy as np

SJ_type_list = ['sj2D']

def check_SJ_type(SJ_type):
    if SJ_type not in SJ_type_list:
        raise ValueError(f"{SJ_type} is not a recognised SJ vehicle type.")


class Scramjet2DManager(OptManager):
    def __init__(
        self,
        paramset,
        constset,
        problem,
        filepath="sqp_res.csv",
        veh_props=None,
        options=None,
        nozzle_surrogate_version="v_3_1",
        cf_model_name="meador_smart",
        cf_params=None,
    ):
        """
        nozzle_surrogate_version, cf_model_name, cf_params: config only (a
        version name / model name + params) -- resolved into loaded objects
        here, once, rather than main.py constructing them directly. Mirrors
        single_point's own get_surrogate()/get_friction_model() caching --
        the loaded surrogate/friction model are expensive to build (or, for
        the friction model, cheap but still shouldn't be reloaded from a
        registry on every solve() call) and don't change across iterations
        or within a flight-condition sweep, only the geometry/DVs do.
        """
        super().__init__(
            paramset, constset, problem, filepath=filepath, veh_props=veh_props, options=options
        )
        self.nozzle_surrogate = load_surrogate_model(nozzle_surrogate_version)

        if cf_model_name == "meador_smart" and cf_params==None:
            cf_params = dict(pkl_path="/home/s4679920/MDAO/hypervisc/BuildRefTempModel/cf_grid_v1_q/cf_x_surrogate.pkl")
        self.cf_model = get_friction_model(cf_model_name, **(cf_params or {}))

    def solve(self, design_point: DesignPoint):
        """
        The outer optimisation loop.
        Follows the XDSM outline
            - generate vehicle
            - create models
            - evaluate vehicle constraints
            - solve trajectory
        This information is stored using self.save_generation(...) for the outer optimiser to use.
        """
        # Move to iteration directory
        working_dir = "./iter_" + str(design_point.iter_idx)
        base_dir = os.getcwd()
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        os.chdir(working_dir)

        if self.options is None:
            self.options = {}

        x = design_point.theta
        parameter_dict = self.paramset.vec2dict(x)
        overrides = self.options.get("overrides")
        mesh_params = self.options.get("mesh_params")

        SJ_type = self.options.get("SJ_type")
        check_SJ_type(SJ_type)

        """ 1. Make vehicle """
        logger.info("Generating vehicle")
        geom_out = generate_scramjet_sj2d(
            active_dv_dict=parameter_dict,
            overrides=overrides,
            mesh_params=mesh_params,
            save=True,
            name="test",
        )
        vehicle = geom_out["vehicle"]
        FWS_param_dict = geom_out["FWS_param_dict"]
        context = geom_out["context"]

        """ 2. Physics simulations and surrogate modelling """

        # evaluate_scramjet_point(
        #     alpha=np.deg2rad(3.755937340194485),
        #     mach=6.5,
        #     altitude=30e3,
        #     equivalence_ratio=0.475857363388641,
        #     vehicle=vehicle,
        #     fws=FWS_param_dict,
        #     context=context,
        #     overrides=overrides,
        #     nozzle_surrogate=None,
        #     cf_model=None,
        #     S_ref=1.0,
        #     verbosity=0,
        # )


        grid = self.constant_characteristics(self.options.get("grid_name", "small_test"))
        decks = generate_scramjet_decks(
            vehicle=vehicle,
            fws=FWS_param_dict,
            context=context,
            overrides=overrides,
            nozzle_surrogate=self.nozzle_surrogate,
            cf_model=self.cf_model,
            aoa_grid=grid["aoa_grid"],
            mach_grid=grid["mach_grid"],
            altitude=grid["altitude"],
            equivalence_ratio=grid["equivalence_ratio"],
        )
        design_point.decks = decks

        # 3. Vehicle constraints / 4. Solve optimal trajectory: not yet
        # implemented for SJ2D -- needs the new 2-DOF (x,z) powered-flight
        # dynamics/mission that actually consumes these three CasADi decks
        # (aero/propulsion/friction). solve() stops here for now; it does
        # NOT call self.save_generation()/self.save_solution(), so
        # design_point.generated/.solved stay False and the outer SQP loop
        # (manager.cold_start(...)) still won't run end-to-end past this
        # point -- only a direct manager.solve(design_point) call completes.
        os.chdir(base_dir)

    @staticmethod
    def constant_characteristics(grid_name: str = "small_test") -> dict:
        """
        Flight-condition grid (+ whatever else stays constant across an
        alpha/mach sweep) for generate_scramjet_decks(). Named presets, not
        a single hardcoded grid, so a wider/different sweep can be added
        later as a new entry without disturbing this one -- "small_test" is
        the exact grid already validated (16/16 points converge, matches
        HyperMDAO where compared), kept so we can always rewind to it.
        """
        grids = {
            "small_test": {
                "aoa_grid": np.deg2rad([0, 2, 4, 6]),
                "mach_grid": [6.0, 6.5, 7.0, 7.5],
                "altitude": 30e3,
                "equivalence_ratio": 0.2,
            },
        }
        if grid_name not in grids:
            raise ValueError(f"Unknown grid '{grid_name}'. Available: {list(grids)}")
        return grids[grid_name]