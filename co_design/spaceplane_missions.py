import numpy as np
import casadi as ca
from codesignTools.casadi_problem import CasadiProblem

from hyperdynamics.system import DynamicSystem
from hyperdynamics.mission import Mission
from hyperdynamics.utils import ramp_approx
from glide_missions import v0_from_energy, energy

eps_val = np.inf
boost_u_llim = [0.7, np.deg2rad(-90), np.deg2rad(-10)]
boost_u_ulim = [1, np.deg2rad(90), np.deg2rad(10)]
spaceplane_u_llim = [np.deg2rad(-5), np.deg2rad(-90), 0]
spaceplane_u_ulim = [np.deg2rad(15), np.deg2rad(90), 1]
mach_max = 25
mach_min = 1.5


class BoostSpaceplane(CasadiProblem):
    def __init__(self, name: str = "boost_spaceplane", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop_sym = system.fun["m_prop"]
        m_fuel_sym = system.fun["m_fuel"]
        m_prop_val = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )
        m_fuel_sym = float(
            system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        # Energy
        E = 0.5 * x[3] ** 2 + 9.81 * x[2]

        boost_spaceplane_mission = Mission(
            name="boost_spaceplane",
            end_cost=[-E / 1e6],
            int_cost=[[0], [1 / 50]],
            ineq=[
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    (mach_max - mach_number),
                ],
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    ((f_q(x[2], x[3])) - 1e3) / 100e3,
                    (mach_max - mach_number),
                    (mach_number - mach_min),
                ],
            ],
            # ineq=[],
            # term=[],
            term=[
                [],
                [
                    np.deg2rad(20) - x[5],
                    x[5],
                    u[1][1] + np.deg2rad(2),
                    np.deg2rad(2) - u[1][1],
                    -x[4] + np.deg2rad(45),
                ],
            ],
            x0=[
                0.0,
                0.0,
                z0,
                v0,
                np.deg2rad(0),
                np.deg2rad(85),
                m_prop_sym,
                m_fuel_sym,
            ],
            x_llim=[
                [-ca.inf, -1e3, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
                [-ca.inf, -1e3, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
            ],
            x_ulim=[ca.inf, 1e3, 100e3, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                spaceplane_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                spaceplane_u_ulim,
            ],
            x_scale=[
                [200e3, 1e3, 50e3, 2500, 1, 1, m_prop_val, m_fuel_sym],
                [1000e3, 1e3, 100e3, 5000, 1, 1, 1, m_fuel_sym],
            ],
            u_scale=[[1, 1, 0.1], [0.1, 1, 1]],
            t_scale=[100, 400],
            max_time=[150, 1e6],
            stages=2,
            # u_rate_lim=[[100, 100], [0.2, 0.2]],
        )

        return boost_spaceplane_mission


class BoostSpaceplaneFlyback(CasadiProblem):
    def __init__(self, name: str = "boost_spaceplane_flyback", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop_sym = system.fun["m_prop"]
        m_fuel_sym = system.fun["m_fuel"]
        m_prop_val = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )
        m_fuel_val = float(
            system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        # Energy
        E = 0.5 * x[3] ** 2 + 9.81 * x[2]

        def flyback_cost(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 0
            y_t = 0
            z_t = 5e3
            return (x - x_t) ** 2 + (y - y_t) ** 2  # **0.5

        def flyback_constraint(x, y, z):
            x_t = 0
            y_t = 0
            r = 10e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        def target_constraint(x, y, z):
            x_t = 600e3
            y_t = 0
            r = 10e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        boost_spaceplane_flyback_mission = Mission(
            name="boost_spaceplane_flyback",
            # end_cost=[[0], [-E / 1e6], [0]],
            end_cost=[[0], [-E / 1e6], [flyback_cost(x[0], x[1], x[2])]],
            # end_cost = [0],
            int_cost=[[0], [1 / 1000], [1 / 1000]],
            ineq=[
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    (mach_max - mach_number),
                ],
                *[
                    [
                        (100e3 - f_q(x[2], x[3])) / 100e3,
                        # ((f_q(x[2], x[3])) - 1e3)/100e3,
                        (mach_max - mach_number),
                        (mach_number - mach_min),
                    ]
                ]
                * 2,
            ],
            term=[
                [],
                [
                    np.deg2rad(20) - x[5],
                    x[5],
                    u[1][1] + np.deg2rad(2),
                    np.deg2rad(2) - u[1][1],
                    -x[4] + np.deg2rad(45),
                ],
                [],
            ],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0, 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf, ca.inf],
            init_ineq=[
                x[6] - m_prop_sym,
                m_prop_sym - x[6],
                x[7] - m_fuel_sym,
                m_fuel_sym - x[7],
            ],
            x_llim=[
                [0, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
                [0, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
                [-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
            ],
            x_ulim=[ca.inf, ca.inf, 100e3, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                spaceplane_u_llim,
                spaceplane_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                spaceplane_u_ulim,
                spaceplane_u_ulim,
            ],
            x_scale=[
                [100e3, 100e3, 50e3, 2500, 1, 1, m_prop_val, m_fuel_val],
                [500e3, 500e3, 100e3, 5000, 1, 1, 1, m_fuel_val],
                [500e3, 500e3, 100e3, 5000, 1, 1, 1, 10],
            ],
            u_scale=[[1, 1, 0.1], [0.1, 1, 1], [0.1, 1, 1]],
            t_scale=[100, 400, 400],
            max_time=[150, 1e6, 1e6],
            min_time=[1, 10, 10],
            stages=3,
            u_rate_lim=[
                [0.1, np.deg2rad(1), np.deg2rad(5.7)],
                [np.deg2rad(0.5), np.deg2rad(1), 0.05],
                [np.deg2rad(0.5), np.deg2rad(1), 0.05],
            ],  # Taken from Sholto's thesis
        )

        return boost_spaceplane_flyback_mission


class BoostSpaceplaneFlyback_w0(CasadiProblem):
    def __init__(
        self, name: str = "boost_spaceplane_flyback_w0", scaling: float = 1e-3
    ):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop_sym = system.fun["m_prop"]
        m_fuel_sym = system.fun["m_fuel"]
        m_prop_val = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )
        m_fuel_val = float(
            system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        # Energy
        E = 0.5 * x[3] ** 2 + 9.81 * x[2]

        def flyback_cost(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 0
            y_t = 0
            z_t = 5e3
            return (x - x_t) ** 2 + (y - y_t) ** 2  # **0.5

        def flyback_constraint(x, y, z):
            x_t = 0
            y_t = 0
            r = 100e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        def target_constraint(x, y, z):
            x_t = 600e3
            y_t = 0
            r = 100e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        boost_spaceplane_flyback_w0 = Mission(
            name="boost_spaceplane_flyback_w0",
            end_cost=[flyback_cost(x[0], x[1], x[2]) / 1e6],
            int_cost=[[0], [0], [1 / 100]],
            ineq=[
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    (mach_max - mach_number),
                ],
                *[
                    [
                        (100e3 - f_q(x[2], x[3])) / 100e3,
                        # ((f_q(x[2], x[3])) - 1e3)/100e3,
                        (mach_max - mach_number),
                        (mach_number - mach_min),
                    ]
                ]
                * 2,
            ],
            term=[
                [],
                [
                    np.deg2rad(20) - x[5],
                    x[5],
                    x[4] - np.deg2rad(45),
                    target_constraint(x[0], x[1], x[2]),
                ],
                [],
            ],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0, 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf, ca.inf],
            init_ineq=[
                x[6] - m_prop_sym,
                m_prop_sym - x[6],
                x[7] - m_fuel_sym,
                m_fuel_sym - x[7],
            ],
            x_llim=[
                [0, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
                [0, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
                [-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0],
            ],
            x_ulim=[ca.inf, ca.inf, 100e3, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                spaceplane_u_llim,
                spaceplane_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                spaceplane_u_ulim,
                spaceplane_u_ulim,
            ],
            x_scale=[
                [200e3, 100e3, 50e3, 2500, 1, 1, m_prop_val, m_fuel_val],
                [1000e3, 100e3, 100e3, 5000, 1, 1, 1, m_fuel_val],
                [1000e3, 100e3, 100e3, 5000, 1, 1, 1, m_fuel_val / 2],
            ],
            u_scale=[[1, 1, 0.1], [0.1, 1, 1], [0.1, 1, 1]],
            t_scale=[100, 400, 400],
            max_time=[150, 1e6, 1e6],
            min_time=[1, 10, 10],
            stages=3,
            # u_rate_lim=[
            #     [0.1, np.deg2rad(1), np.deg2rad(5.7)],
            #     [np.deg2rad(0.5), np.deg2rad(1), 0.05],
            #     [np.deg2rad(0.5), np.deg2rad(1), 0.05],
            # ],  # Taken from Sholto's thesis
        )

        return boost_spaceplane_flyback_w0


class SpaceplaneFlyback(CasadiProblem):
    def __init__(self, name: str = "spaceplane_flyback", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_fuel_sym = system.fun["m_fuel"]
        m_vehicle_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))
        m_fuel_val = float(
            system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        m0_n = 3579
        v0_n = 2653
        z0_n = 28142
        eta = 1
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81)*eta
        v0 = v0_from_energy(E0, z0, m_vehicle_sym + m_fuel_sym)
        m0 = m_vehicle_sym + m_fuel_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        # Energy
        E = 0.5 * x[3] ** 2 + 9.81 * x[2]

        def flyback_cost(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 0
            y_t = 0
            z_t = 5e3
            return (x - x_t) ** 2 + (y - y_t) ** 2  # **0.5

        def flyback_constraint(x, y, z):
            x_t = 0
            y_t = 0
            r = 10e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        def target_constraint(x, y, z):
            x_t = 600e3
            y_t = 0
            r = 10e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        spaceplane_flyback_mission = Mission(
            name="spaceplane_flyback_mission",
            end_cost=[[-E / 1e6], [flyback_cost(x[0], x[1], x[2])]],
            int_cost=[[1 / 1000], [1 / 1000]],
            ineq=[
                (100e3 - f_q(x[2], x[3])) / 100e3,
                # ((f_q(x[2], x[3])) - 1e3)/100e3,
                (mach_max - mach_number),
                (mach_number - mach_min),
            ],
            term=[
                [
                    np.deg2rad(20) - x[5],
                    x[5],
                    u[1][1] + np.deg2rad(2),
                    np.deg2rad(2) - u[1][1],
                    -x[4] + np.deg2rad(45),
                ],
                [],
            ],
            x0_llim=[0.0, 0.0, 0, 0, np.deg2rad(0), np.deg2rad(0), 0],
            x0_ulim=[0.0, 0.0, ca.inf, ca.inf, np.deg2rad(0), np.deg2rad(90), ca.inf],
            init_ineq=[
                (x[6] - m_fuel_sym),
                (m_fuel_sym - x[6]),
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
            ],
            x_llim=[
                -ca.inf,
                -ca.inf,
                0e3,
                0,
                -ca.inf,
                -ca.inf,
                0,
            ],
            x_ulim=[ca.inf, ca.inf, 100e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=spaceplane_u_llim,
            u_ulim=spaceplane_u_ulim,
            x_scale=[
                [500e3, 500e3, 100e3, 5000, 1, 1, m_fuel_val],
                [500e3, 500e3, 100e3, 5000, 1, 1, 10],
            ],
            u_scale=[0.1, 1, 1],
            t_scale=400,
            max_time=1e6,
            min_time=10,
            stages=2,
            u_rate_lim=[
                np.deg2rad(0.5),
                np.deg2rad(1),
                0.05,
            ],  # Taken from Sholto's thesis
        )

        return spaceplane_flyback_mission


class SpaceplaneFlyback_w0(CasadiProblem):
    def __init__(self, name: str = "spaceplane_flyback_w0", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_fuel_sym = system.fun["m_fuel"]
        m_vehicle_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))
        m_fuel_val = float(
            system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        m0_n = 3579
        v0_n = 2653
        z0_n = 28142
        eta = 1
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81)*eta
        v0 = v0_from_energy(E0, z0, m_vehicle_sym + m_fuel_sym)
        m0 = m_vehicle_sym + m_fuel_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        # Energy
        E = 0.5 * x[3] ** 2 + 9.81 * x[2]

        def flyback_cost(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 0
            y_t = 0
            z_t = 5e3
            return (x - x_t) ** 2 + (y - y_t) ** 2  # **0.5

        def flyback_constraint(x, y, z):
            x_t = 0
            y_t = 0
            r = 100e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        def target_constraint(x, y, z):
            x_t = 600e3
            y_t = 0
            r = 100e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        spaceplane_flyback_w0 = Mission(
            name="spaceplane_flyback_w0",
            # end_cost=[[0], [-E / 1e6], [miss_location(x[0], x[1], x[2]) / 1e6]],
            # end_cost=[[0], [-E / 1e6], [0]],
            end_cost=[flyback_cost(x[0], x[1], x[2]) / 1e6],
            # end_cost = [0],
            int_cost=[[0], [1 / 100]],
            ineq=[
                (100e3 - f_q(x[2], x[3])) / 100e3,
                # ((f_q(x[2], x[3])) - 1e3)/100e3,
                (mach_max - mach_number),
                (mach_number - mach_min),
            ],
            # term=[[], [x[0] - 100e3], []],
            term=[
                [
                    np.deg2rad(20) - x[5],
                    x[5],
                    x[4] - np.deg2rad(45),
                    target_constraint(x[0], x[1], x[2]),
                ],  # u[1][1], -u[1][1]],
                # [flyback_constraint(x[0], x[1], x[2])],
                [],
            ],
            # x0=[0.0, 0.0, z0, v0, np.deg2rad(0), np.deg2rad(0), m_fuel_sym],
            x0_llim=[0.0, 0.0, 0, 0, np.deg2rad(0), np.deg2rad(0), 0],
            x0_ulim=[0.0, 0.0, ca.inf, ca.inf, np.deg2rad(0), np.deg2rad(90), ca.inf],
            init_ineq=[
                (x[6] - m_fuel_sym),
                (m_fuel_sym - x[6]),
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
            ],
            x_llim=[
                -ca.inf,
                -ca.inf,
                0e3,
                0,
                -ca.inf,
                -ca.inf,
                0,
            ],
            x_ulim=[ca.inf, ca.inf, 100e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=spaceplane_u_llim,
            u_ulim=spaceplane_u_ulim,
            x_scale=[
                [1000e3, 100e3, 100e3, 5000, 1, 1, m_fuel_val],
                [1000e3, 100e3, 100e3, 5000, 1, 1, m_fuel_val / 2],
            ],
            u_scale=[0.1, 1, 1],
            t_scale=400,
            max_time=1e6,
            min_time=10,
            stages=2,
            # u_rate_lim=[
            #     [0.1, np.deg2rad(1), np.deg2rad(5.7)],
            #     [np.deg2rad(0.5), np.deg2rad(1), 0.05],
            #     [np.deg2rad(0.5), np.deg2rad(1), 0.05],
            # ],  # Taken from Sholto's thesis
        )

        return spaceplane_flyback_w0
