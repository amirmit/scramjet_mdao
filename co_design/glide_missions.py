import numpy as np
import casadi as ca
from codesignTools.casadi_problem import CasadiProblem

from hyperdynamics.system import DynamicSystem
from hyperdynamics.mission import Mission
from hyperdynamics.utils import ramp_approx

eps_val = np.inf
HGV_u_llim = [np.deg2rad(-5), np.deg2rad(-90)]
HGV_u_ulim = [np.deg2rad(15), np.deg2rad(90)]
mach_max = 25
mach_min = 2
# HV_u_rate_lim = [np.deg2rad(0.5), np.deg2rad(1)]  # Taken from Sholto's thesis
HV_u_rate_lim = [np.deg2rad(5), np.deg2rad(10)]


def v0_from_energy(E0, h0, m):
    v0 = ((E0 - m * 9.81 * h0) / (0.5 * m)) ** 0.5
    return v0


def energy(v0, z0, m):
    return (0.5 * v0**2 + 9.81 * z0) * m


class TargetObstructionGlider(CasadiProblem):
    def __init__(self, name: str = "target_obstruction_glider", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]

        m_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        s_x0_n = 0
        m0_n = 704
        v0_n = 4841
        z0_n = 33400
        eta = 1  # Percentage of initial energy used
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81) * eta
        v0 = v0_from_energy(E0, z0, m_sym)
        m0 = m_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def obstruction1(x, y):
            x_c = 500e3
            y_c = 0e3
            r = 150e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction2(x, y):
            x_c = 1000e3
            y_c = 100e3
            r = 150e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction3(x, y):
            x_c = 700e3
            y_c = 250e3
            r = 100e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction4(x, y):
            x_c = 350e3
            y_c = -100e3
            r = 100e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction(x, y):
            x_c = 800e3
            y_c = 275e3
            r = 300e3

            return ramp_approx(-((x - x_c) ** 2) - (y - y_c) ** 2 + r**2, eps_val=1e5)

        def obstruction1_cost(x, y):
            x_c = 600e3
            y_c = 250e3
            r = 300e3
            return r**2 / ((x - x_c) ** 2 + (y - y_c) ** 2)

        def miss_location(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 1200e3
            y_t = 0e3
            return (x - x_t) ** 2 + (y - y_t) ** 2

        target_obstruction = Mission(
            name="target_obstruction_glider",
            end_cost=[miss_location(x[0], x[1], x[2]) / 1e6],
            int_cost=[1 / 10],
            ineq=[
                (100e3 - f_q(x[2], x[3])) / 100e3,
                mach_max - mach_number,
                mach_number - mach_min,
                obstruction1(x[0], x[1]),
                obstruction2(x[0], x[1]),
                obstruction3(x[0], x[1]),
                obstruction4(x[0], x[1]),
            ],
            term=[],
            x0_llim=[s_x0_n, 0, 0, 0, np.deg2rad(-45), np.deg2rad(0)],
            x0_ulim=[s_x0_n, 0, ca.inf, ca.inf, np.deg2rad(45), np.deg2rad(15)],
            init_ineq=[
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
                (100e3 - f_q(x[2], x[3])) / 100e3,
            ],
            x_llim=[
                -ca.inf,
                -100e3,
                30e3,
                0,
                -ca.inf,
                -ca.inf,
            ],
            x_ulim=[ca.inf, 300e3, 60e3, ca.inf, ca.inf, ca.inf],
            u_llim=HGV_u_llim,
            u_ulim=HGV_u_ulim,
            x_scale=[
                1200e3,
                100e3,
                100e3,
                5000,
                1,
                10,
            ],
            u_scale=[0.1, 1],
            t_scale=400,
            max_time=1e6,
            stages=1,
            u_rate_lim=HV_u_rate_lim,
        )

        return target_obstruction


class MaxRangeObstructionGlider(CasadiProblem):
    def __init__(
        self, name: str = "max_range_obstruction_glider", scaling: float = 1e-3
    ):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]

        m_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        s_x0_n = 0e3
        m0_n = 704
        v0_n = 4705
        z0_n = 30000
        eta = 1.2  # Percentage of initial energy used
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81) * eta
        v0 = v0_from_energy(E0, z0, m_sym)
        m0 = m_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def obstruction1(x, y):
            x_c = 700e3
            y_c = 225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction2(x, y):
            x_c = 1200e3
            y_c = -225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction3(x, y):
            x_c = 1700e3
            y_c = 225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction4(x, y):
            x_c = 2200e3
            y_c = -225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction5(x, y):
            x_c = 2700e3
            y_c = 225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction6(x, y):
            x_c = 3200e3
            y_c = -225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        def obstruction7(x, y):
            x_c = 3700e3
            y_c = 225e3
            r = 300e3
            return ((x - x_c) ** 2 + (y - y_c) ** 2 - r**2) / 1e4

        max_range_obstruction_glider = Mission(
            name="max_range_obstruction_glider",
            end_cost=[-x[0] / 1e6],
            int_cost=[0],
            ineq=[
                (100e3 - f_q(x[2], x[3])) / 100e3,
                mach_max - mach_number,
                mach_number - mach_min,
                obstruction1(x[0], x[1]),
                obstruction2(x[0], x[1]),
                obstruction3(x[0], x[1]),
                obstruction4(x[0], x[1]),
                obstruction5(x[0], x[1]),
                obstruction6(x[0], x[1]),
                obstruction7(x[0], x[1]),
            ],
            term=[],
            # x0=[0.0, 0.0, z0, v0, np.deg2rad(0), np.deg2rad(85)],
            x0_llim=[s_x0_n, 0, 0, 0, np.deg2rad(-45), np.deg2rad(0)],
            x0_ulim=[s_x0_n, 0, ca.inf, ca.inf, np.deg2rad(45), np.deg2rad(90)],
            init_ineq=[
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
                (100e3 - f_q(x[2], x[3])) / 100e3,
            ],
            x_llim=[
                -ca.inf,
                -200e3,
                30e3,
                0,
                -ca.inf,
                -ca.inf,
            ],
            x_ulim=[ca.inf, 200e3, 60e3, ca.inf, ca.inf, ca.inf],
            u_llim=HGV_u_llim,
            u_ulim=HGV_u_ulim,
            x_scale=[
                3000e3,
                200e3,
                60e3,
                2500,
                1,
                10,
            ],
            u_scale=[0.1, 1],
            t_scale=1000,
            max_time=1e6,
            stages=1,
            u_rate_lim=HV_u_rate_lim,
        )

        return max_range_obstruction_glider


class SprintGlider(CasadiProblem):
    def __init__(self, name: str = "sprint_glider", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]

        m_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        s_x0_n = 28e3
        m0_n = 704
        v0_n = 4753
        z0_n = 32700
        eta = 1  # Percentage of initial energy used
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81) * eta
        v0 = v0_from_energy(E0, z0, m_sym)
        m0 = m_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        sprint_glider = Mission(
            name="sprint_glider",
            end_cost=[-x[0]],
            int_cost=[0],
            ineq=[
                (100e3 - f_q(x[2], x[3])),
                mach_max - mach_number,
                mach_number - mach_min,
            ],
            term=[(450 - t) / 450],
            x0_llim=[s_x0_n, 0, 0, 0, np.deg2rad(-45), np.deg2rad(0)],
            x0_ulim=[s_x0_n, 0, ca.inf, ca.inf, np.deg2rad(45), np.deg2rad(90)],
            init_ineq=[
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
                (100e3 - f_q(x[2], x[3])) / 100e3,
            ],
            x_llim=[
                -ca.inf,
                -100e3,
                30e3,
                0,
                -ca.inf,
                -ca.inf,
            ],
            x_ulim=[ca.inf, 100e3, 60e3, ca.inf, ca.inf, ca.inf],
            u_llim=HGV_u_llim,
            u_ulim=HGV_u_ulim,
            x_scale=[
                1000e3,
                100e3,
                100e3,
                2000,
                1,
                10,
            ],
            u_scale=[0.1, 1],
            t_scale=430,
            max_time=630,
            stages=1,
            u_rate_lim=HV_u_rate_lim,
        )

        return sprint_glider


class Target2Glider(CasadiProblem):
    def __init__(self, name: str = "target_2_glider", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        f_rho = system.fun["f_rho"]

        m_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))
        thermal_fun = system.fun["thermal_fun"]["HV"]

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        s_x0_n = 0e3
        m0_n = 761
        v0_n = 6092
        z0_n = 60000
        eta = 1  # Percentage of initial energy used
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81) * eta
        v0 = v0_from_energy(E0, z0, m_sym)
        m0 = m_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def location_1(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 500e3
            y_t = 0e3
            return (x - x_t) ** 2 + (y - y_t) ** 2

        def location_2(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 1000e3
            y_t = 300e3
            return (x - x_t) ** 2 + (y - y_t) ** 2

        def location_3(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 1000e3
            y_t = 0e3
            return (x - x_t) ** 2 + (y - y_t) ** 2

        def target_constraint(x, y, z):
            x_t = 1000e3
            y_t = 0
            r = 1e3
            return r**2 - (x - x_t) ** 2 - (y - y_t) ** 2  # **0.5

        def heat_load_cost(z, v, lim):
            rho = f_rho(z)
            q = thermal_fun(rho, v, *ca.vertsplit(p)) / 1e6

            return 5 ** (q - lim) / 1e6

        target_2 = Mission(
            name="target_2_glider",
            end_cost=[
                [location_1(x[0], x[1], x[2]) / 1e9],
                [location_2(x[0], x[1], x[2]) / 1e9],
                # [location_3(x[0], x[1], x[2]) / 1e9],
                [0],
            ],
            int_cost=[1 / 100 + heat_load_cost(x[2], x[3], 6)],
            ineq=[
                (100e3 - f_q(x[2], x[3])) / 100e3,
                mach_max - mach_number,
                mach_number - mach_min,
                (6e6 - thermal_fun(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
            ],
            term=[target_constraint(x[0], x[1], x[2]) / 1e6],
            x0_llim=[s_x0_n, 0, 0, 0, np.deg2rad(-45), np.deg2rad(0)],
            x0_ulim=[s_x0_n, 0, ca.inf, ca.inf, np.deg2rad(45), np.deg2rad(90)],
            init_ineq=[
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
                (100e3 - f_q(x[2], x[3])) / 100e3,
            ],
            x_llim=[
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, ca.inf, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                HGV_u_llim,
                HGV_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                HGV_u_ulim,
                HGV_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [500e3, 100e3, 60e3, 2000, 1, 10],
                [1000e3, 500e3, 60e3, 2000, 1, 10],
                [1000e3, 500e3, 60e3, 2000, 1, 10],
            ],
            u_scale=[
                [0.1, 1],
                [0.1, 1],
                [0.1, 1],
            ],
            t_scale=[100, 400, 400],
            max_time=1e6,
            min_time=10,
            stages=3,
            u_rate_lim=[
                HV_u_rate_lim,
                HV_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return target_2


class Sprint2Glider(CasadiProblem):
    def __init__(self, name: str = "sprint_2_glider", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        f_rho = system.fun["f_rho"]

        m_sym = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))
        thermal_fun = system.fun["thermal_fun"]["HV"]

        #  Calc properties to get vmax in terms of Mach no
        z0 = 50e3
        s_x0_n = 0e3
        m0_n = 713
        v0_n = 4461
        z0_n = 40563
        eta = 1  # Percentage of initial energy used
        E0 = (0.5 * m0_n * v0_n**2 + m0_n * z0_n * 9.81) * eta
        v0 = v0_from_energy(E0, z0, m_sym)
        m0 = m_sym

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def end_location_cost(x):
            x_t = 1800e3

            return (x - x_t)**2 

        def heat_load_cost(z, v, lim):
            rho = f_rho(z)
            q = thermal_fun(rho, v, *ca.vertsplit(p)) / 1e6

            return 5 ** (q - lim) / 1e6

        sprint = Mission(
            name="sprint_glider",
            # end_cost=[end_location_cost(x[0])/1e8],
            end_cost=[0],
            int_cost=[
                1 / 100 + heat_load_cost(x[2], x[3], 6),
            ],
            ineq=[
                (100e3 - f_q(x[2], x[3])) / 100e3,
                mach_max - mach_number,
                mach_number - mach_min,
                (6e6 - thermal_fun(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
            ],
            # term = [],
            term=[(x[0] - 1800e3) / 1000e3],
            x0_llim=[s_x0_n, 0, 0, 0, np.deg2rad(-45), np.deg2rad(0)],
            x0_ulim=[s_x0_n, 0, ca.inf, ca.inf, np.deg2rad(45), np.deg2rad(90)],
            init_ineq=[
                (E0 - energy(x[3], x[2], m0)),
                (energy(x[3], x[2], m0) - E0),
                (100e3 - f_q(x[2], x[3])) / 100e3,
            ],
            x_llim=[
                [-ca.inf, -100e3, 30e3, 0, -ca.inf, -ca.inf],
            ],
            x_ulim=[ca.inf, 100e3, 60e3, ca.inf, ca.inf, ca.inf],
            u_llim=[
                HGV_u_llim,
            ],
            u_ulim=[
                HGV_u_ulim,
            ],
            x_scale=[
                1000e3,
                100e3,
                100e3,
                5000,
                1,
                10,
            ],
            u_scale=[0.1, 1],
            t_scale=500,
            max_time=1e6,
            stages=1,
            u_rate_lim=[
                HV_u_rate_lim,
            ],
        )

        return sprint
