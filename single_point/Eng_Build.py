import numpy as np

from HyperPro.Nozzle_2D_Surrogate import Nozzle_2D_Surrogate

from hypervehicle import Vehicle
from hypervehicle.generator import Generator
from hypervehicle.components.swept import SweptComponent
from hypervehicle.geometry import Vector3, Line, Bezier, CoonsPatch
from hypervehicle.utilities import PatchTag

from HyperPro.Inlet2D import Inlet_2D
from HyperPro.Cowl_turn import CowlTurn
from HyperPro.Engine_1D import Engine_1D
from HyperPro.geometry import Inlet_2D_Geom, Engine_1D_Geom, Cowl_Geom, Nozzle_Geom
from HyperPro.Conserve import ConserveState


class Parametric_EngModel(Generator):
    def __init__(self, verbosity: int = 1, **kwargs) -> None:

        self.inlet_length_1 = 1
        self.inlet_length_2 = 0.3
        self.inlet_theta_1 = np.deg2rad(3)
        self.inlet_theta_2 = np.deg2rad(5)
        self.throat_height = 0.1
        self.engine_tilt_ang = np.deg2rad(0)
        self.cowl_LE = 0.1
        self.engine_length = 1
        self.isolator_length = 2
        self.engine_h5_h3 = 1
        self.nozzle_length = 0.1
        self.nozzle_angle = np.deg2rad(10)
        self.nozzle_mid_frac = 0.5

        self.S_ref = 1
        self.c_ref = 1

        self.source_mdot = 0
        self.source_mom_x = 0
        self.source_mom_y = 0
        self.source_mom_z = 0
        self.source_tE = 0

        self.equivalence_ratio = 0.5
        self.flight_duration = 10

        self.verbosity = verbosity

        self.bez_start_ratio = 0.1

        self.freestream = None
        self.f_st = 1 / 34.3  # for Hydrogen combustion in air
        self.eta_pr = 0.8  # from "Scramjets", 2007, M.Smart
        self.h_pr = 119954e3  # for Hydrogen combustion from "Scramjets", 2007, M.Smart

        self.x43_w_l_ratio = (
            36.8 / 143
        )  # X43-A width to length ratio, from Marchall 2005

        self.viscosityFlg = True

        self.mesh_resolution = 2

        # Complete instantiation
        super().__init__(**kwargs)

    def create_vehicle_geom(self):
        self.inlet_lengths = [
            self.inlet_length_1,
            self.inlet_length_2,
        ]
        self.inlet_thetas = [self.inlet_theta_1, self.inlet_theta_2]
        self.source_mom = Vector3(
            x=self.source_mom_x, y=self.source_mom_y, z=self.source_mom_z
        )

        #####################################################
        ##                 VEHICLE GEOMETRY                ##
        #####################################################

        # Vehicle cross-section (x-z plane).  +x toward nose (left), +z down.
        #
        #               e7            e6
        #         N0 ──────── N1 ────────────── T0
        #       e0 \                         ,/'
        #           I1                      T1    e5: Bezier C2─T2─T1─T0
        #  +x ←   e1 \                     /      T2: Bezier ctrl pt
        #  +z ↓       C1 ─────────── C2 ─ T2
        #          e2 /            e4 │
        #            E1 ──────────── E2
        #                    e3
        #

        self._geom = {}

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

        # Nozzle v2 geometry
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

    def create_instance(self) -> Vehicle:
        # Create vehicle object
        eng_model = Vehicle()
        eng_model.configure(name="Eng_Model", verbosity=self.verbosity)
        self.create_vehicle_geom()

        # Define patches forming wedge
        sections = []
        for i in [-1, 1]:
            y_loc = 0.5 * i * self.width
            axial_shift = Vector3(x=0, y=y_loc, z=0)

            top_line = Line(
                p0=self._geom["T0"] + axial_shift, p1=self._geom["N0"] + axial_shift
            )
            N1 = top_line(0.5)

            e0 = Line(
                p0=self._geom["N0"] + axial_shift, p1=self._geom["I1"] + axial_shift
            )
            e1 = Line(
                p0=self._geom["I1"] + axial_shift, p1=self._geom["C1"] + axial_shift
            )
            e2 = Line(
                p0=self._geom["C1"] + axial_shift, p1=self._geom["E1"] + axial_shift
            )
            e3 = Line(
                p0=self._geom["E1"] + axial_shift, p1=self._geom["E2"] + axial_shift
            )
            e4 = Line(
                p0=self._geom["E2"] + axial_shift, p1=self._geom["C2"] + axial_shift
            )
            e5 = Bezier(
                [
                    self._geom["C2"] + axial_shift,
                    self._geom["T2"] + axial_shift,
                    self._geom["T1"] + axial_shift,
                    self._geom["T0"] + axial_shift,
                ]
            )
            e6 = Line(p0=self._geom["T0"] + axial_shift, p1=N1)
            e7 = Line(p0=N1, p1=self._geom["N0"] + axial_shift)

            sections.append([e0, e1, e2, e3, e4, e5, e6, e7])

        fuselage_tags = {
            "swept_patch_0": PatchTag.INLET,  # inlet ramp1
            "swept_patch_1": PatchTag.INLET,  # inlet ramp2
            "swept_patch_2": PatchTag.INLET,  # intake
            "swept_patch_3": PatchTag.COWL,  # cowl
            "swept_patch_4": PatchTag.OUTLET,  # outlet
            "swept_patch_5": PatchTag.NOZZLE,  # nozzle
            "swept_patch_6": PatchTag.TOP,  # top back
            "swept_patch_7": PatchTag.TOP,  # top front
            "swept_patch_end_0_sec_1": PatchTag.INLET,  # end
            "swept_patch_end_0_sec_2": PatchTag.INLET,  # end
            "swept_patch_end_0_sec_3": PatchTag.INLET,  # end
            "swept_patch_end_0_sec_4": PatchTag.INLET,  # end
            "swept_patch_end_1_sec_1": PatchTag.INLET,  # end
            "swept_patch_end_1_sec_2": PatchTag.INLET,  # end
            "swept_patch_end_1_sec_3": PatchTag.INLET,  # end
            "swept_patch_end_1_sec_4": PatchTag.INLET,  # end
        }

        mesh_resolution = {
            "e0": self.mesh_resolution * 2,
            "e1": self.mesh_resolution * 2,
            "e2": self.mesh_resolution * 2,
            "e3": self.mesh_resolution * 2,
            "e4": self.mesh_resolution * 2,
            "e5": self.mesh_resolution * 4,
            "e6": self.mesh_resolution * 2,
            "e7": self.mesh_resolution * 2,
            "sweep": self.mesh_resolution * 2,
        }

        def end_callback(cs):
            C1C2 = Line(p0=cs[2].p0, p1=cs[4].p1)
            O2 = C1C2(0.5)
            O2N1 = Line(p0=O2, p1=cs[7].p0)
            O1 = O2N1(0.5)
            I1O1 = Line(p0=cs[1].p0, p1=O1)
            C1O2 = Line(p0=cs[2].p0, p1=O2)
            O2C2 = Line(p0=O2, p1=cs[4].p1)
            O2O1 = Line(p0=O2, p1=O1)
            O1N1 = Line(p0=O1, p1=cs[7].p0)

            return I1O1, C1O2, O2C2, O2O1, O1N1, O2N1, C1C2

        def end_callback0(cs):
            I1O1, C1O2, O2C2, O2O1, O1N1, O2N1, C1C2 = end_callback(cs)

            patch1 = CoonsPatch(
                west=cs[0].reversed(), north=cs[7].reversed(), south=I1O1, east=O1N1
            )
            patch2 = CoonsPatch(
                west=cs[1].reversed(), north=I1O1, south=C1O2, east=O2O1
            )
            patch3 = CoonsPatch(
                west=cs[2].reversed(), north=C1C2, south=cs[3], east=cs[4]
            )
            patch4 = CoonsPatch(
                west=O2N1, north=cs[6].reversed(), south=O2C2, east=cs[5]
            )

            res = mesh_resolution
            out = []
            out.append(["swept_patch_end_0_sec_1", patch1, res["e7"], res["e0"]])
            out.append(["swept_patch_end_0_sec_2", patch2, res["e3"], res["e1"]])
            out.append(["swept_patch_end_0_sec_3", patch3, res["e3"], res["e2"]])
            out.append(["swept_patch_end_0_sec_4", patch4, res["e6"], res["e5"]])

            return out

        def end_callback1(cs):
            I1O1, C1O2, O2C2, O2O1, O1N1, O2N1, C1C2 = end_callback(cs)

            patch1 = CoonsPatch(
                south=cs[0].reversed(), east=cs[7].reversed(), west=I1O1, north=O1N1
            )
            patch2 = CoonsPatch(
                south=cs[1].reversed(), east=I1O1, west=C1O2, north=O2O1
            )
            patch3 = CoonsPatch(
                south=cs[2].reversed(), east=C1C2, west=cs[3], north=cs[4]
            )
            patch4 = CoonsPatch(
                south=O2N1, east=cs[6].reversed(), west=O2C2, north=cs[5]
            )

            res = mesh_resolution
            out = []
            out.append(["swept_patch_end_1_sec_1", patch1, res["e0"], res["e7"]])
            out.append(["swept_patch_end_1_sec_2", patch2, res["e1"], res["e3"]])
            out.append(["swept_patch_end_1_sec_3", patch3, res["e2"], res["e3"]])
            out.append(["swept_patch_end_1_sec_4", patch4, res["e5"], res["e6"]])
            return out

        # fuselage = SweptComponentMultiFace(
        fuselage = SweptComponent(
            cross_sections=sections,
            mesh_resolution=mesh_resolution,
            close_ends=True,
            end_callback0=end_callback0,
            end_callback1=end_callback1,
            check_mesh=False,
            # name="fuselage",
            tags=fuselage_tags,
        )

        ###################################
        ##        Build Components       ##
        ###################################
        eng_model.add_component(fuselage)

        return eng_model

    def create_inlet(self):

        nominal_inlet_geom = Inlet_2D_Geom(
            l=self._geom["inlet_lengths"],
            theta=self._geom["inlet_thetas"],
            h3=self._geom["throat_height"],
            zeta=self._geom["engine_tilt_ang"],
            C1E1=self._geom["C1E1"],
            width=self._geom["width"],
        )
        return Inlet_2D(nominal_inlet_geom, self.freestream, verbosity=self.verbosity)

    def create_cowl(self, conserve_2: ConserveState):

        nominal_cowl_geom = Cowl_Geom(
            theta=self._geom["inlet_thetas"],
            h3=self._geom["throat_height"],
            zeta=self._geom["engine_tilt_ang"],
            C1E1=self._geom["C1E1"],
            width=self._geom["width"],
        )

        return CowlTurn(
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
    ) -> Nozzle_2D_Surrogate:
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

        return Nozzle_2D_Surrogate(
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

    def calculate_fuel_mass(self, cells):
        """
        Calculates the max fuel mass based on these assumptions:
        1. Fuel tank volume = 10% of vehicle volume (assumed to be the case for the X43-A)
        2. Fuel tank pressure = 8,500 psi (from X43-A - ref?)
        3. rho_H2 @ 8,500 psi = 35 kg/m^3
        """
        vol_f = 0.1 * cells.vol
        mass_f = 35 * vol_f

        return mass_f


if __name__ == "__main__":

    pass
