#!/usr/bin/env python3
# SHARED WITH single_point: scramjet_mdao/single_point/ imports this file directly
# (Parametric_Scramjet2D incl. create_*/calculate_mass/calculate_fuel_mass, Force_Calculator, sum_aero_results, PatchTag).
# Changing anything it uses affects BOTH the co-design pipeline and single-point
# optimisation -- check scramjet_mdao/single_point/ (Run_ObjFunc.py, case_config.py)
# and the note in scramjet_mdao/co_design/PROJECT_STATE.md before editing.
"""
Example for generating a generic 2-D extruded geometry of a hypersonic
vehicle with underslung scramjet engine.

                            e7
                    ______________---------------T0
    N0--------------                         ---  e6
      ---___              _______---------T1
     e0     ---I1---------               /       (note: T1 is Bezier control point)
                  \                     / e5
               e1   \       ____------C2
                     C1-----          |
                    /                 | e4
               e2  /                  |
                  E1-----------------E2
        +x                  e3
        <-----+
              |
              V +z


Authors: Amir Mittelman, Ingo Jahn
Last Modified: 30/06/2025
"""

import numpy as np
import pyvista as pv
import enum
from typing import Dict, Any

from autodiff import FloatWithSens
from autodiff.array_with_sens import ArrayWithSens
from hyperProp import Inlet_2D, Cowl_Turn, Engine_1D, Nozzle_2D
from hyperProp import Inlet_2D_Geom, Cowl_Geom, Engine_1D_Geom, Nozzle_Geom
from hyperProp.conserve import ConserveState
from hypergrad import SurfaceIntegrator, isentropic_pressure_sensitivity
from hypervehicle import Component, logger
from hypervehicle.geom.path import Line, Bezier, SubRangedPath
from hypervehicle.geom.surface import CoonsPatch
from hypervehicle.geom.compsurface.swept import SweptComponent
from hyperpanel import PanelSolver, PanelScheme
from hypervisc import friction_drag

from utilities import Vector3, FlowState, AeroResults, SurfaceIntegratorWithMoments

class PatchTag(enum.Enum):
   DONT_CALC = 0
   FREE_STREAM = 1
   INLET = 2
   OUTLET = 3
   NOZZLE = 4
   TOP = 5
   COWL = 6
   INTAKE = 7
   SIDE = 8


def sum_aero_results(results, flow_state, moment_ref_point=None) -> AeroResults:
    """Sum a list of AeroResults (force/moment add) into one combined
    AeroResults. Safe regardless of what each entry's own flow_state/
    moment_ref_point was: every hyperProp-derived contribution (inlet,
    cowl-turn, engine, nozzle) carries an identically-zero moment (that
    physics isn't tracked at the 1D/lumped level), so only the panel-method
    entries (top, cowl-out) contribute a real moment, always about
    Force_Calculator.reference_point -- passed in here explicitly rather
    than inferred from an arbitrary entry.
    """
    force = Vector3(0, 0, 0)
    moment = Vector3(0, 0, 0)
    for r in results:
        force = force + r.force
        moment = moment + r.moment
    kwargs = {} if moment_ref_point is None else {"moment_ref_point": moment_ref_point}
    return AeroResults(force=force, moment=moment, flow_state=flow_state, **kwargs)


class Scramjet2D_Comp(Component):

    def __init__(self,
                 params: dict[str, float],
                 ref_params: dict[str, float] = None,
                 mesh_params: dict[str, float] = None,
                 label: str = "sj2D",
                 **kwargs
                 ) -> None:

        super().__init__(label=label, params=params, ref_params=ref_params, mesh_params=mesh_params, **kwargs)

    def set_defaults(self) -> None:
        self.params = {  # Geometry
            "inlet_length_1": 1,
            "inlet_length_2": 0.3,
            "inlet_theta_1": np.deg2rad(3),
            "inlet_theta_2": np.deg2rad(5),
            "throat_height": 0.1,
            "engine_tilt_ang": np.deg2rad(0),
            "cowl_LE": 0.1,
            "engine_length": 1,
            "isolator_length": 2,
            "engine_h5_h3": 1,
            "nozzle_length": 0.1,
            "nozzle_angle": np.deg2rad(10),
            "nozzle_mid_frac": 0.5,
            "bez_start_ratio": 0.1,
        }

        self.mesh_params = {"mesh_resolution": 2}
        self.export_params = {}
        return

    def geometry(self):

        def scramjet2D_patch_generator(*args):

            input_params = dict(zip(self.parameters.keys(), args))
            scramjet2D = Parametric_Scramjet2D(input_params)

            patch_dict, tags = scramjet2D._patch_generator()
            self.scramjet2D = scramjet2D

            patch_list = []
            for i, key in enumerate(patch_dict.patches.keys()):
                logger.info(f"Adding Patch to patch_list[{i}]: label={key}; key={i}")
                patch_list += [
                    {
                        "patch_func": patch_dict.patches[key],
                        "n_r": patch_dict.patch_res_r[key],
                        "n_s": patch_dict.patch_res_s[key],
                        "cf_list": None,
                        "keys": {"PatchTag": tags[key].name},
                    }
                ]

            return patch_list

        self.add_patch_generator(label=self.label,
                                 generator=scramjet2D_patch_generator,
                                 params=list(self.parameters.keys())
                                 )



class Parametric_Scramjet2D:

    def __init__(self, params, verbosity=1):
        for key, value in params.items():
            setattr(self, key, value)
        self.verbosity = verbosity

        self.check_defaults()

    def check_defaults(self):
        """
        Setting defaults for values that are not used in hypervehicle.

        Includes the same geometry defaults as Scramjet2D_Comp.set_defaults()
        -- needed because Parametric_Scramjet2D is also constructed directly
        (e.g. scramjet_point.evaluate_scramjet_point), not only via
        Scramjet2D_Comp/Component.__init__, which is the only other place a
        DV/override dict gets topped up with the nominal geometry. Keep
        these two defaults dicts in sync.
        """
        geom_defaults = {
            "inlet_length_1": 1,
            "inlet_length_2": 0.3,
            "inlet_theta_1": np.deg2rad(3),
            "inlet_theta_2": np.deg2rad(5),
            "throat_height": 0.1,
            "engine_tilt_ang": np.deg2rad(0),
            "cowl_LE": 0.1,
            "engine_length": 1,
            "isolator_length": 2,
            "engine_h5_h3": 1,
            "nozzle_length": 0.1,
            "nozzle_angle": np.deg2rad(10),
            "nozzle_mid_frac": 0.5,
            "bez_start_ratio": 0.1,
        }
        eng_defaults = {
            "f_st": 1 / 34.3,  # for Hydrogen combustion in air
            "eta_pr": 0.8,  # from "Scramjets", 2007, M.Smart
            "h_pr": 119954e3,  # for Hydrogen combustion from "Scramjets", 2007, M.Smart
            "source_mdot": 0,
            "source_mom": Vector3(x=0, y=0, z=0),
            "source_tE": 0,
            "x43_w_l_ratio": 36.8 / 143,  # X-43 width/length aspect ratio
            "viscosityFlg": False, # Engine friction claculation
        }

        for key, value in (geom_defaults | eng_defaults).items():
            if not hasattr(self, key):
                setattr(self, key, value)


    def create_vehicle_geom(self):

        self._geom = {}

        # Calculate Nodes defining cross-section

        # Inlet
        self.inlet_lengths = [self.inlet_length_1, self.inlet_length_2]
        self.inlet_thetas = [self.inlet_theta_1, self.inlet_theta_2]

        theta_i = 0
        inlet_x = [0]
        inlet_z = [0]

        for i in range(len(self.inlet_thetas)):
            theta_i = theta_i + self.inlet_thetas[i]
            inlet_x = inlet_x + [inlet_x[i] - self.inlet_lengths[i]]
            inlet_z = inlet_z + [inlet_z[i] + self.inlet_lengths[i] * np.tan(theta_i)]

        N0 = Vector3(x=inlet_x[0], y=0, z=inlet_z[0])
        I1 = Vector3(x=inlet_x[1], y=0, z=inlet_z[1])
        C1 = Vector3(x=inlet_x[2], y=0, z=inlet_z[2])

        # Engine
        h3 = self.throat_height
        L_eng = self.engine_length + self.isolator_length
        zeta_eng = self.engine_tilt_ang

        h5 = h3 * self.engine_h5_h3
        gamma_eng = 2 * np.arctan((h5 - h3) / (2 * L_eng))
        Lc = L_eng / np.cos(gamma_eng / 2)
        omega_u = zeta_eng + gamma_eng / 2
        omega_d = zeta_eng - gamma_eng / 2

        C2 = Vector3(x=C1.x - Lc * np.cos(omega_u), y=0, z=C1.z - Lc * np.sin(omega_u))
        E2 = Vector3(
            x=C2.x - h5 * np.sin(zeta_eng), y=0, z=C2.z + h5 * np.cos(zeta_eng)
        )
        E1 = Vector3(
            x=E2.x + (Lc + self.cowl_LE) * np.cos(omega_d),
            y=0,
            z=E2.z + (Lc + self.cowl_LE) * np.sin(omega_d),
        )

        # Nozzle (4-point Bezier: C2 -> T2 -> T1 -> T0), ported from
        # HyperMDAO's Eng_Build.py "Nozzle v2 geometry" to replace the
        # previous 3-point Bezier + back_angle/back_length construction.
        C1C2 = Line(p0=C1, p1=C2)
        bez_start_length = (
            self.nozzle_length * np.cos(self.nozzle_angle) * self.bez_start_ratio
        )
        l_T2 = 1 + bez_start_length / C1C2.length()
        T2 = Vector3(x=C1C2(l_T2).x, y=0, z=C1C2(l_T2).z)

        nozzle_end_ang = omega_u + self.nozzle_angle

        T0 = Vector3(
            x=T2.x - self.nozzle_length * np.cos(nozzle_end_ang),
            y=0,
            z=T2.z - self.nozzle_length * np.sin(nozzle_end_ang),
        )

        T2T0 = Line(p0=T2, p1=T0)

        y_vec = Vector3(x=0.0, y=-1.0, z=0.0)
        n = T2T0.p1mp0.cross(y_vec).unit
        mid = T2T0(0.5)

        nozzle_mid_length = (
            self.nozzle_mid_frac * self.nozzle_length * np.tan(self.nozzle_angle) / 2
        )

        T1 = Vector3(
            x=mid.x - n.x * nozzle_mid_length,
            y=0.0,
            z=mid.z - n.z * nozzle_mid_length,
        )

        C1E1 = Line(p0=C1, p1=E1)

        # Calculate width from X43 aspect ratio
        self.length = N0.x - T0.x
        self.width = self.length * self.x43_w_l_ratio

        self._geom["N0"] = N0
        self._geom["I1"] = I1
        self._geom["C1"] = C1
        self._geom["C2"] = C2
        self._geom["E1"] = E1
        self._geom["E2"] = E2
        self._geom["T0"] = T0
        self._geom["T1"] = T1
        self._geom["T2"] = T2
        self._geom["C1E1"] = C1E1
        self._geom["inlet_lengths"] = self.inlet_lengths
        self._geom["inlet_thetas"] = self.inlet_thetas
        self._geom["throat_height"] = self.throat_height
        self._geom["engine_tilt_ang"] = self.engine_tilt_ang
        self._geom["omega_u"] = omega_u
        self._geom["engine_h5_h3"] = self.engine_h5_h3
        self._geom["isolator_length"] = self.isolator_length
        self._geom["engine_length"] = self.engine_length
        self._geom["nozzle_length"] = self.nozzle_length
        self._geom["nozzle_angle"] = self.nozzle_angle
        self._geom["nozzle_mid_frac"] = self.nozzle_mid_frac
        self._geom["width"] = self.width
        self._geom["length"] = self.length

    def _patch_generator(self):
        self.create_vehicle_geom()

        # Define patches forming wedge
        sections = []
        for i in [-1, 1]:
            y_loc = 0.5 * i * self.width
            axial_shift = Vector3(x=0, y=y_loc, z=0)

            e0 = Line(p0=self._geom["N0"] + axial_shift, p1=self._geom["I1"] + axial_shift)
            e1 = Line(p0=self._geom["I1"] + axial_shift, p1=self._geom["C1"] + axial_shift)
            e2 = Line(p0=self._geom["C1"] + axial_shift, p1=self._geom["E1"] + axial_shift)
            e3 = Line(p0=self._geom["E1"] + axial_shift, p1=self._geom["E2"] + axial_shift)
            e4 = Line(p0=self._geom["E2"] + axial_shift, p1=self._geom["C2"] + axial_shift)
            nozzle = Bezier(
                [
                    self._geom["C2"] + axial_shift,
                    self._geom["T2"] + axial_shift,
                    self._geom["T1"] + axial_shift,
                    self._geom["T0"] + axial_shift,
                ]
            )
            e5 = SubRangedPath(underlying_path=nozzle, t0=0, t1=0.5)
            e6 = SubRangedPath(underlying_path=nozzle, t0=0.5, t1=1.0)
            e7 = Line(p0=self._geom["T0"] + axial_shift, p1=self._geom["N0"] + axial_shift)

            sections.append([e0, e1, e2, e3, e4, e5, e6, e7])

        mesh_resolution = {
            "e0": 2 * 2,
            "e1": 2 * 2,
            "e2": 2 * 2,
            "e3": 2 * 2,
            "e4": 2 * 2,
            "e5": 2 * 2,
            "e6": 2 * 2,
            "e7": 2 * 2,
            "sweep": 2 * 2,
        }

        def end_callback0(cs):
            line0 = Line(p0=cs[0](1), p1=cs[5](1))
            line1 = Line(p0=cs[1](1), p1=cs[4](1))
            patch0 = CoonsPatch(
                west=cs[0].reversed(), east=cs[6], south=line0, north=cs[7].reversed()
            )
            patch1 = CoonsPatch(
                west=cs[1].reversed(), east=cs[5], south=line1, north=line0
            )
            patch2 = CoonsPatch(
                west=cs[2].reversed(), east=cs[4], south=cs[3], north=line1
            )

            res = mesh_resolution
            out = []
            out.append(["swept_patch_end_0_sec_0", patch0, res["e3"], res["e0"]])
            out.append(["swept_patch_end_0_sec_1", patch1, res["e3"], res["e1"]])
            out.append(["swept_patch_end_0_sec_2", patch2, res["e3"], res["e2"]])
            return out

        def end_callback1(cs):
            line0 = Line(p0=cs[5](1), p1=cs[0](1))
            line1 = Line(p0=cs[4](1), p1=cs[1](1))
            patch0 = CoonsPatch(
                west=cs[0].reversed(), east=cs[6], south=line0, north=cs[7]
            )
            patch1 = CoonsPatch(
                west=cs[1].reversed(), east=cs[5], south=line1, north=line0
            )
            patch2 = CoonsPatch(
                west=cs[2].reversed(), east=cs[4], south=cs[3].reversed(), north=line1
            )

            res = mesh_resolution
            out = []
            out.append(["swept_patch_end_1_sec_0", patch0, res["e3"], res["e0"]])
            out.append(["swept_patch_end_1_sec_1", patch1, res["e3"], res["e1"]])
            out.append(["swept_patch_end_1_sec_2", patch2, res["e3"], res["e2"]])
            return out

        fuselage_tags = {
            "swept_patch_0": PatchTag.INLET,  # inlet ramp1
            "swept_patch_1": PatchTag.INLET,  # inlet ramp2
            "swept_patch_2": PatchTag.INTAKE,  # intake
            "swept_patch_3": PatchTag.COWL,  # cowl
            "swept_patch_4": PatchTag.OUTLET,  # outlet
            "swept_patch_5": PatchTag.NOZZLE,  # nozzle
            "swept_patch_6": PatchTag.NOZZLE,  # nozzle
            "swept_patch_7": PatchTag.TOP,  # top front
            "swept_patch_end_0_sec_0": PatchTag.SIDE,  # side
            "swept_patch_end_0_sec_1": PatchTag.SIDE,  # side
            "swept_patch_end_0_sec_2": PatchTag.SIDE,  # side
            "swept_patch_end_1_sec_0": PatchTag.SIDE,  # side
            "swept_patch_end_1_sec_1": PatchTag.SIDE,  # side
            "swept_patch_end_1_sec_2": PatchTag.SIDE,  # side
        }

        fuselage_swept = SweptComponent(
            cross_sections=sections,
            mesh_resolution=mesh_resolution,
            close_ends=False,
            end_callback0=end_callback0,
            end_callback1=end_callback1,
            # check_mesh=False,
            # name="fuselage",
        )

        fuselage_swept.generate_patches()
        return fuselage_swept, fuselage_tags


    def create_inlet(self):

        nominal_inlet_geom = Inlet_2D_Geom(
            l=self._geom["inlet_lengths"],
            theta=self._geom["inlet_thetas"],
            h3=self._geom["throat_height"],
            zeta=self._geom["engine_tilt_ang"],
            C1E1=self._geom["C1E1"],
            width=self._geom["width"],
        )
        return Inlet_2D(
            nominal_inlet_geom, self.freestream, verbosity=self.verbosity
        )

    def create_cowl(self, conserve_2: ConserveState):

        nominal_cowl_geom = Cowl_Geom(
            theta=self._geom["inlet_thetas"],
            h3=self._geom["throat_height"],
            zeta=self._geom["engine_tilt_ang"],
            C1E1=self._geom["C1E1"],
            width=self._geom["width"],
        )

        return Cowl_Turn(
            conserve_2=conserve_2,
            cowl_geom=nominal_cowl_geom,
            free_stream=self.freestream,
            verbosity=self.verbosity,
        )

    def create_engine(self, conserve_3: ConserveState):

        eng_geom = Engine_1D_Geom(
            h3=self._geom["throat_height"],
            h5_h3=self._geom["engine_h5_h3"],
            zeta=self._geom["engine_tilt_ang"],
            C1=self._geom["C1"],
            C2=self._geom["C2"],
            isolator_length=self._geom["isolator_length"],
            engine_length=self._geom["engine_length"],
            width=self._geom["width"],
        )

        if self.equivalence_ratio is not None:
            m_dot_f = conserve_3.m_dot * self.f_st * self.equivalence_ratio
            self.source_tE = self.h_pr * self.eta_pr * m_dot_f
            self.source_mdot = m_dot_f

        eng_params = ConserveState(
            m_dot=self.source_mdot, mom=self.source_mom, tE=self.source_tE
        )

        return Engine_1D(
            conserve_3=conserve_3,
            eng_geom=eng_geom,
            eng_params=eng_params,
            freestream=self.freestream,
            viscosityFlg=self.viscosityFlg,
            verbosity=self.verbosity,
        )

    def create_nozzle(
        self,
        outflow,
        surrogate,
    ) -> Nozzle_2D:
        """Build a Nozzle surrogate model instance

        Nozzle_Geom assembled from the current geometry state.
        `surrogate` is a pre-built NozzleSurrogate, loaded once by the
        caller and shared across all calls.

        Must be called after create_instance() so that self._geom is populated.
        """
        nozzle_geom = Nozzle_Geom(
            h5=self._geom["throat_height"] * self._geom["engine_h5_h3"],
            nozzle_length=self._geom["nozzle_length"],
            nozzle_angle=self._geom["nozzle_angle"],
            nozzle_mid_frac=self._geom["nozzle_mid_frac"],
            omega_u=self._geom["omega_u"],
            C2=self._geom["C2"],
            width=self.width,
        )

        return Nozzle_2D(
            nozzle_geom=nozzle_geom,
            outflow=outflow,
            surrogate=surrogate,
            verbosity=self.verbosity,
        )

    def calculate_mass(self):
        """
        Calculates the vehicle mass. Based on the correlation from Fleeman, 2022, Missile Design Guide, Fig. 4.8
        WL = 0.04 * l * d^2 (WL in lb, l and d in inches)
        """
        body_hieght = abs(self._geom["T0"].z - self._geom["C1"].z)
        d2 = (4 * body_hieght * self.width) / np.pi
        l_d2 = self.length * d2
        mass_vol_slope = 1.1071996094465121e3  # [kg/m^3] from Fleeman, 2022, Missile Design Guide, Fig. 4.8
        mass = mass_vol_slope * l_d2

        return mass

    def calculate_fuel_mass(self, volume):
        """
        Calculates the max fuel mass based on these assumptions:
        1. Fuel tank volume = 10% of vehicle volume (assumed to be the case for the X43-A)
        2. Fuel tank pressure = 8,500 psi (from X43-A - ref?)
        3. rho_H2 @ 8,500 psi = 35 kg/m^3
        """
        vol_f = 0.1 * volume
        mass_f = 35 * vol_f

        return mass_f


class Force_Calculator():
    """
    A class for calculating the aerodynamic forces and moments on patches of a vehicle.
    Can calculate each patch using different methods from hyperpanel.panel_solver.PanelSolver.
    Can calculate each patch using a different FlowState (for calculating external aerodynamics and nozzle forces)
    """
    def __init__(
            self,
            mesh: pv.UnstructuredGrid,
            tags: np.ndarray,
            context,
    ):
        self.mesh = mesh
        self.tags = tags
        self.context = context

        self.normals = ArrayWithSens.from_FWS_array(self.mesh.normals, self.context)
        self.areas = ArrayWithSens.from_FWS_array(self.mesh.areas, self.context)
        self.centroids = ArrayWithSens.from_FWS_array(self.mesh.centroids, self.context)

        # self.normals = np.ndarray([self.mesh.normals[i] for i in range(self.mesh.normals.shape[0])])
        # self.areas = ArrayWithSens.from_FWS_array(self.mesh.areas, self.context)
        # self.centroids = ArrayWithSens.from_FWS_array(self.mesh.centroids, self.context)

        self.reference_point = np.array([-0.7, 0, -0.05])  # Center of mass is estimate

        self._prep_results()
        self._parse_vehicle()
        self._generate_PanelSolvers()


    def calculate_panel_flow(
            self,
            freestream: FlowState,
            eng_flow: FlowState,
    ):

        for tag, panel in self.panels.items():
            if tag==PatchTag.NOZZLE.name:
                panel["flow_state"] = eng_flow
            else:
                panel["flow_state"] = freestream

            results = panel["PanelSolver"].compute_flow_characteristics(flow_state=panel["flow_state"])
            results["P"] = panel["flow_state"].cp_to_p(results["cp"])
            results["T"] = results["temp_ratio"] * panel["flow_state"].T
            # results["aero_forces"] = self._compute_integral_forces(panel=panel, pressure=results["P"])

            panel["PanelSolver_results"] = results

            self.panels[tag] = panel
            self._unparse_vehicle(panel)

    def calculate_friction_force(self, freestream: FlowState, cf_model, length) -> AeroResults:
        """Skin-friction drag over every real wall -- all patches except the
        INTAKE/OUTLET duct-opening cross-sections, which have no wall to
        carry friction. SIDE patches count here even though they're excluded
        from the inviscid panel calc in _parse_vehicle.
        """
        wetted_indices = np.where(
            (self.tags != PatchTag.INTAKE.name) & (self.tags != PatchTag.OUTLET.name)
        )[0]
        total_area = self.areas[wetted_indices].sum()
        return friction_drag(total_area, freestream, cf_model, length)

    def patch_aero_result(self, tag: "PatchTag") -> AeroResults:
        """AeroResults for a single tag's panels. calculate_panel_flow()
        must have been called first. Only meaningful for TOP/COWL -- every
        other tag was given PanelScheme.DONT_CALC in _parse_vehicle, so its
        panel-method output isn't a real force (those forces come from
        hyperProp's lumped calc instead).
        """
        panel = self.panels[tag.name]
        return self._compute_integral_forces_with_sens(
            panel, panel["PanelSolver_results"]["P"]
        )

    def calculate_panel_sens(self):
        """Per-PatchTag force/moment sensitivity, one {"force","moment"}
        pair per tag that has panels -- same panels dict / skip-if-empty
        pattern as `Run_HyperMDAO_Compare.py::integrate_forces`, not
        restricted to any specific tag(s).

        Uses `hypergrad.SurfaceIntegrator` (same class/pattern as
        `Jack_example/spaceplanes-2026/aero_check.py`'s
        `generate_aerodeck_first_stage_fairing`), not
        `utilities.SurfaceIntegratorWithMoments` (the plain-ndarray
        integrator `Run_HyperMDAO_Compare.py::integrate_forces` uses for
        forces) -- `SurfaceIntegrator.integrate_solution` chain-rules a
        local pressure sensitivity together with the panel area/normal
        sensitivity it derives itself from the `.sens` arrays passed in,
        giving the full force/moment sensitivity in one call instead of
        just the pressure-slope term.

        Unlike Jack's script, this passes raw pressure `P` (not `cp`) and
        `isentropic_pressure_sensitivity`'s dP/dtheta (not
        `SurfaceIntegrator.calculate_pressure_sensitivity`'s dCp/dtheta) into
        `integrate_solution` -- `integrate_solution` just does linear algebra
        on whatever per-panel scalar field it's given, so this keeps the
        result in the same Newton/N.m basis as `integrate_forces`'s
        pressure-based force/moment above, instead of a dimensionless
        Cp-area basis.
        """
        n_params = len(self.context.variable_names)
        sens = {}
        for tag_name, panel in self.panels.items():
            results = panel["PanelSolver_results"]
            pressure = results["P"]
            if pressure.size == 0:
                continue
            integrator = SurfaceIntegrator(
                area=panel["areas"].values,
                normal=panel["normals"].values,
                centroid=panel["centroids"].values,
                com=self.reference_point,
                area_sens=panel["areas"].sens,
                normal_sens=panel["normals"].sens,
                centroid_sens=panel["centroids"].sens,
                nparams=n_params,
            )
            p_sens = isentropic_pressure_sensitivity(
                mach_local=results["mach_local"],
                p_local=pressure,
                velocity_dir_local=results["unit_vel_vector"],
                normal_sens=panel["normals"].sens,
                gamma=panel["flow_state"].gmodel.gamma,
            )
            _, force_sens, _, moment_sens = integrator.integrate_solution(pressure, p_sens)
            sens[tag_name] = {"force": force_sens, "moment": moment_sens}
        return sens


    def _parse_vehicle(self):

        panels = {}
        for tag in PatchTag:

            indices = np.where(self.tags == tag.name)[0]
            normals = self.normals[indices]
            areas = self.areas[indices]
            centroids = self.centroids[indices]

            if tag in (PatchTag.TOP, PatchTag.COWL):
                # Exact Prandtl-Meyer, not PRANDTL_MEYER_INTERP: the interpolated
                # scheme (M2 from a 1-deg table) is piecewise-smooth, so its
                # analytic gradient and finite differences disagree by 1-4% (SNOPT
                # verify flags L_W_ratio/range) -- see PROJECT_STATE.md.
                myScheme = [PanelScheme.PRANDTL_MEYER, PanelScheme.OBLIQUE_SHOCK]
                angle_threshold = [0.0]
            else:
                # INLET/COWL(intake-turn)/engine/nozzle forces come from
                # hyperProp's 1D lumped calc, not the panel method -- running
                # the inviscid panel scheme on them would waste compute and
                # double-count their pressure force. INTAKE/OUTLET are duct
                # opening cross-sections, not real walls. SIDE patches are
                # excluded from the inviscid calc here but still count
                # toward wetted area for friction (see
                # calculate_friction_force).
                myScheme = [PanelScheme.DONT_CALC]
                angle_threshold = []

            panels[tag.name] = {
                "indices": indices,
                "normals": normals,
                "areas": areas,
                "centroids": centroids,
                "myScheme": myScheme,
                "angle_threshold": angle_threshold,
            }
        self.panels = panels

    def _generate_PanelSolvers(self):

        for key in self.panels.keys():
            panel_solver = PanelSolver(
                normals=self.panels[key]["normals"].values,
                solver_methods=self.panels[key]["myScheme"],
                angle_thresholds=self.panels[key]["angle_threshold"],
            )

            self.panels[key]["PanelSolver"] = panel_solver

    def _unparse_vehicle(self, panel):
        ind = panel["indices"]
        gamma = panel["flow_state"].gmodel.gamma
        self.vehicle_results["M"][ind] = panel["PanelSolver_results"]["mach_local"]
        self.vehicle_results["P"][ind] = panel["PanelSolver_results"]["P"]
        self.vehicle_results["T"][ind] = panel["PanelSolver_results"]["T"]
        self.vehicle_results["unit_vel_vector"][ind] = panel["PanelSolver_results"]["unit_vel_vector"]
        self.vehicle_results["gamma"][ind] = gamma

    def _prep_results(self):
        n_cells = len(self.normals)
        self.vehicle_results: Dict[str, Any] = {
            "M": np.zeros(n_cells),
            "P": np.zeros(n_cells),
            "T": np.zeros(n_cells),
            "unit_vel_vector": np.zeros([n_cells, 3]),
            "gamma": np.zeros(n_cells),
            "aero_forces": None,
        }

    def _calculate_vehicle_force(self, freestream: FlowState):
        force = Vector3(0,0,0)
        moment = Vector3(0, 0, 0)
        for panel in self.panels.values():
            force += panel["PanelSolver_results"]["aero_forces"].force
            moment += panel["PanelSolver_results"]["aero_forces"].moment
        self.vehicle_results["aero_forces"] = AeroResults(
            force=force,
            moment=moment,
            flow_state=freestream,
            moment_ref_point = self.reference_point,
        )

    def _compute_integral_forces_with_sens(self, panel, pressure) -> AeroResults:
        """Force/moment of one panel patch, with the FULL sensitivity.

        Uses hypergrad.SurfaceIntegrator (as Jack_example's aerodecks do), whose
        integrate_solution chain-rules the pressure-vs-surface-angle term
        (isentropic_pressure_sensitivity's dP/dtheta) together with the
        area/normal/centroid geometry term. The plain utilities integrator
        used by _compute_integral_forces only carries the geometry term (the
        pressure is a constant to it), which was ~10x too small for TOP/COWL.
        Force/moment values are identical between the two integrators.
        """
        results = panel["PanelSolver_results"]
        integrator = SurfaceIntegrator(
            area=panel["areas"].values,
            normal=panel["normals"].values,
            centroid=panel["centroids"].values,
            com=self.reference_point,
            area_sens=panel["areas"].sens,
            normal_sens=panel["normals"].sens,
            centroid_sens=panel["centroids"].sens,
            nparams=len(self.context.variable_names),
        )
        pressure_sens = isentropic_pressure_sensitivity(
            mach_local=results["mach_local"],
            p_local=pressure,
            velocity_dir_local=results["unit_vel_vector"],
            normal_sens=panel["normals"].sens,
            gamma=panel["flow_state"].gmodel.gamma,
        )
        force, force_sens, moment, moment_sens = integrator.integrate_solution(
            pressure, pressure_sens
        )

        # Pack value + sensitivity row into FloatWithSens components BEFORE
        # building AeroResults, so everything downstream that sums/rotates it
        # carries the full gradient.
        force_vec = Vector3(
            *(FloatWithSens(number=float(force[i]), sens=force_sens[i]) for i in range(3))
        )
        moment_vec = Vector3(
            *(FloatWithSens(number=float(moment[i]), sens=moment_sens[i]) for i in range(3))
        )
        return AeroResults(force=force_vec, moment=moment_vec, flow_state=panel["flow_state"])

    def _compute_integral_forces(self, panel, pressure):
        integrator = SurfaceIntegratorWithMoments.from_arrays(
            areas=panel["areas"],
            normals=panel["normals"],
            centroids=panel["centroids"],
            reference_point=self.reference_point
        )
        moment = integrator.calculate_pressure_moment(pressure)
        force = integrator.calculate_pressure_force(pressure)

        # calculate_pressure_force/_moment return a 3-element ArrayWithSens;
        # AeroResults requires Vector3 (or a plain np.ndarray) -- this method
        # was previously never actually invoked (the only call site was
        # commented out), so this conversion had never been exercised.
        force_vec = Vector3(x=force[0], y=force[1], z=force[2])
        moment_vec = Vector3(x=moment[0], y=moment[1], z=moment[2])

        return AeroResults(force=force_vec, moment=moment_vec, flow_state=panel["flow_state"])