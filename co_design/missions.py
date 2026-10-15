import numpy as np
import casadi as ca
from codesignTools.casadi_problem import CasadiProblem

from hyperdynamics.system import DynamicSystem
from hyperdynamics.mission import Mission
from hyperdynamics.utils import ramp_approx

eps_val = np.inf
# eps_val = 0.1
boost_u_llim = [np.deg2rad(-10), np.deg2rad(-90), 0.7]
boost_u_ulim = [np.deg2rad(10), np.deg2rad(90), 1]
HGV_u_llim = [np.deg2rad(-5), np.deg2rad(-90)]
HGV_u_ulim = [np.deg2rad(15), np.deg2rad(90)]
mach_max = 25
mach_min = 2
boost_u_rate_lim = [np.deg2rad(5.7), np.deg2rad(1), 0.05]  # Taken from Sholto's thesis
# HV_u_rate_lim = [np.deg2rad(0.5), np.deg2rad(1)] # Taken from Sholto's thesis
HV_u_rate_lim = [np.deg2rad(5), np.deg2rad(10)]


class MaxRangeGroundLaunch(CasadiProblem):
    def __init__(self, name: str = "max_range_ground_launch", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v)
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 2 ** (a * (qa - lim) / 1e6) / 1e2

        max_range = Mission(
            name="max_range_ground_launch",
            end_cost=[-x[0] / 1e7],
            int_cost=[
                [
                    q_alpha(x[2], x[3], u[0][0], -168e3, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 168e3, lim_type="upper")
                ],
                [0],
            ],
            ineq=[
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # 125e3 - f_q(x[2], x[3]),
                    mach_max - mach_number,
                ],
                [
                    (100e3 - f_q(x[2], x[3])),
                    mach_max - mach_number,
                    mach_number - mach_min,
                ],
            ],
            term=[],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, ca.inf, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [200e3, 10e3, 50e3, 2000, 1, 10, mass_prop],
                [2000e3, 10e3, 60e3, 2000, 1, 10, 1],
            ],
            u_scale=[[0.01, 1, 0.1], [0.1, 1]],
            t_scale=[100, 500],
            max_time=1e6,
            min_time=[20, 150],
            stages=2,
            u_rate_lim=[
                boost_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return max_range


class TargetGroundLaunch(CasadiProblem):
    def __init__(self, name: str = "target_ground_launch", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v)
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 2 ** (a * (qa - lim) / 1e9) / 1e2

        def miss_location(x, y, z):
            """Distance from target without altitude requirement"""
            x_t = 900e3
            y_t = 100e3
            return (x - x_t) ** 2 + (y - y_t) ** 2  # **0.5

        target = Mission(
            name="target",
            end_cost=[miss_location(x[0], x[1], x[2]) / 1e3],
            int_cost=[
                [
                    1 / 500
                    + q_alpha(x[2], x[3], u[0][0], -168e3, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 168e3, lim_type="upper")
                ],
                [1 / 500],
            ],
            ineq=[
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # 125e3 - f_q(x[2], x[3]),
                    mach_max - mach_number,
                ],
                [
                    (100e3 - f_q(x[2], x[3])),
                    mach_max - mach_number,
                    mach_number - mach_min,
                ],
            ],
            term=[],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, ca.inf, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [200e3, 100e3, 50e3, 2000, 1, 10, mass_prop],
                [900e3, 100e3, 100e3, 2000, 1, 10, 1],
            ],
            u_scale=[[0.01, 1, 0.1], [0.1, 1]],
            t_scale=[150, 400],
            max_time=[150, 1e6],
            stages=2,
            u_rate_lim=[
                boost_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return target


class TargetObstructionGroundLaunch(CasadiProblem):
    def __init__(
        self, name: str = "target_obstruction_ground_launch", scaling: float = 1e-3
    ):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v)
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 2 ** (a * (qa - lim) / 1e9) / 1e2

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
            name="target_obstruction_ground_launch",
            end_cost=[miss_location(x[0], x[1], x[2]) / 1e6],
            int_cost=[
                [
                    1 / 10
                    + q_alpha(x[2], x[3], u[0][0], -168e3, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 168e3, lim_type="upper")
                ],
                [1 / 10],
            ],
            ineq=[
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # (200e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                ],
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                    mach_number - mach_min,
                    obstruction1(x[0], x[1]),
                    obstruction2(x[0], x[1]),
                    obstruction3(x[0], x[1]),
                    obstruction4(x[0], x[1]),
                ],
            ],
            term=[],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -100e3, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -100e3, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, 300e3, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [200e3, 100e3, 50e3, 2000, 1, 10, mass_prop],
                [1200e3, 100e3, 100e3, 2000, 1, 10, 1],
            ],
            u_scale=[[0.01, 1, 0.1], [0.1, 1]],
            t_scale=[100, 400],
            max_time=[100, 1e6],
            stages=2,
            u_rate_lim=[
                boost_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return target_obstruction


class MaxRangeObstructionGroundLaunch(CasadiProblem):
    def __init__(
        self, name: str = "max_range_obstruction_ground_launch", scaling: float = 1e-3
    ):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v)
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 2 ** (a * (qa - lim) / 1e9) / 1e2

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

        max_range_obstruction = Mission(
            name="max_range_obstruction_ground_launch",
            end_cost=[-x[0] / 1e6],
            int_cost=[
                [
                    # 0,
                    q_alpha(x[2], x[3], u[0][0], -168e3, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 168e3, lim_type="upper")
                ],
                [0],
            ],
            ineq=[
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # (200e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                ],
                [
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
            ],
            term=[],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -200e3, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -200e3, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[
                [ca.inf, 200e3, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
                [ca.inf, 200e3, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            ],
            u_llim=[
                boost_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [500e3, 200e3, 50e3, 5000, 1, 10, mass_prop],
                [3000e3, 200e3, 60e3, 5000, 1, 10, 1],
            ],
            u_scale=[[0.01, 1, 0.1], [0.1, 1]],
            t_scale=[100, 1000],
            max_time=1e6,
            min_time=[20, 150],
            stages=2,
            u_rate_lim=[
                boost_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return max_range_obstruction


class SprintGroundLaunch(CasadiProblem):
    def __init__(self, name: str = "sprint_ground_launch", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v)
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 2 ** (a * (qa - lim) / 1e9) / 1e2

        sprint = Mission(
            name="sprint_ground_launch",
            end_cost=[-x[0] / 1e6],
            int_cost=[
                [
                    # 0,
                    q_alpha(x[2], x[3], u[0][0], -168e3, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 168e3, lim_type="upper")
                ],
                [0],
            ],
            ineq=[
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # (200e3 - f_q(x[2], x[3]))/100e3,
                    mach_max - mach_number,
                ],
                [
                    (100e3 - f_q(x[2], x[3])),
                    mach_max - mach_number,
                    mach_number - mach_min,
                ],
            ],
            term=[(500 - t) / 500],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -100e3, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -100e3, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, 100e3, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                boost_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                boost_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [200e3, 100e3, 50e3, 2000, 1, 10, mass_prop],
                [1000e3, 100e3, 100e3, 2000, 1, 10, 1],
            ],
            u_scale=[[0.01, 1, 0.1], [0.1, 1]],
            t_scale=[100, 430],
            max_time=1e6,
            min_time=[20, 150],
            stages=2,
            u_rate_lim=[
                boost_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return sprint


class Target2GroundLaunch(CasadiProblem):
    def __init__(self, name: str = "target_2_ground_launch", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        f_rho = system.fun["f_rho"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )
        thermal_fun_boost = system.fun["thermal_fun"]["boost"]
        thermal_fun_HGV = system.fun["thermal_fun"]["HV"]

        #  Calc properties to get vmax in terms of Mach no
        z0 = 0
        v0 = 0

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v) / 1e3
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            # return 2 ** (a * (qa - lim) / 1e9) / 1e2
            return 5 ** (a * (qa - lim)) / 1e6

        def heat_load_cost(z, v, lim):
            rho = f_rho(z)
            q = thermal_fun_HGV(rho, v, *ca.vertsplit(p)) / 1e6

            return 5 ** (q - lim) / 1e6

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

        target_2 = Mission(
            name="target_2_ground_launch",
            end_cost=[
                [0],
                [0],
                [location_1(x[0], x[1], x[2]) / 1e9],
                [location_2(x[0], x[1], x[2]) / 1e9],
                # [location_3(x[0], x[1], x[2]) / 1e9],
                [0],
            ],
            int_cost=[
                [1 / 100],
                [
                    1 / 100
                    + q_alpha(x[2], x[3], u[0][0], -170, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 170, lim_type="upper")
                    + heat_load_cost(x[2], x[3], 6)
                ],
                [1 / 100 + heat_load_cost(x[2], x[3], 6)],
                [1 / 100 + heat_load_cost(x[2], x[3], 6)],
                [1 / 100 + heat_load_cost(x[2], x[3], 6)],
            ],
            ineq=[
                [],
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # (200e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                    (6e6 - thermal_fun_boost(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                    mach_number - mach_min,
                    (6e6 - thermal_fun_HGV(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                    mach_number - mach_min,
                    (6e6 - thermal_fun_HGV(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                    mach_number - mach_min,
                    (6e6 - thermal_fun_HGV(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
            ],
            term=[target_constraint(x[0], x[1], x[2]) / 1e6],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, ca.inf, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                [0, 0, 1],
                boost_u_llim,
                HGV_u_llim,
                HGV_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                [0, 0, 1],
                boost_u_ulim,
                HGV_u_ulim,
                HGV_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [200e3, 100e3, 50e3, 2000, 1, 10, mass_prop],
                [200e3, 100e3, 50e3, 2000, 1, 10, mass_prop],
                [500e3, 100e3, 100e3, 2000, 1, 10, 1],
                [1000e3, 500e3, 100e3, 2000, 1, 10, 1],
                [1000e3, 500e3, 100e3, 2000, 1, 10, 1],
            ],
            u_scale=[
                [0.01, 1, 0.1],
                [0.01, 1, 0.1],
                [0.1, 1],
                [0.1, 1],
                [0.1, 1],
            ],
            t_scale=[1, 100, 100, 400, 400],
            max_time=[3, 1e6, 1e6, 1e6, 1e6],
            min_time=[3, 10, 10, 10, 10],
            stages=5,
            u_rate_lim=[
                boost_u_rate_lim,
                boost_u_rate_lim,
                HV_u_rate_lim,
                HV_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return target_2


class Sprint2GroundLaunch(CasadiProblem):
    def __init__(self, name: str = "sprint_2_ground_launch", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        f_rho = system.fun["f_rho"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )
        thermal_fun_boost = system.fun["thermal_fun"]["boost"]
        thermal_fun_HGV = system.fun["thermal_fun"]["HV"]

        #  Calc properties to get vmax in terms of Mach no
        z0 = 0
        v0 = 0

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def end_location_cost(x):
            x_t = 1800e3

            return (x - x_t)**2

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v) / 1e3
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 5 ** (a * (qa - lim)) / 1e6
            # return 2 ** (a * (qa - lim) / 1e6) / 1e1 # Used for glide only optimised vehicles to converge

        def heat_load_cost(z, v, lim):
            rho = f_rho(z)
            q = thermal_fun_HGV(rho, v, *ca.vertsplit(p)) / 1e6

            return 5 ** (q - lim) / 1e6

        sprint = Mission(
            name="sprint_ground_launch",
            end_cost=[end_location_cost(x[0])/1e8],
            int_cost=[
                [1 / 100],
                [
                    1 / 100
                    + q_alpha(x[2], x[3], u[0][0], -170, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 170, lim_type="upper")
                    + heat_load_cost(x[2], x[3], 6)
                ],
                [
                    1 / 100 + heat_load_cost(x[2], x[3], 6)
                    ],
            ],
            ineq=[
                [],
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # (200e3 - f_q(x[2], x[3]))/100e3,
                    mach_max - mach_number,
                    (6e6 - thermal_fun_boost(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
                [
                    (100e3 - f_q(x[2], x[3])) / 100e3,
                    mach_max - mach_number,
                    mach_number - mach_min,
                    (6e6 - thermal_fun_HGV(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
            ],
            term=[],
            # term=[(x[0] - 1800e3) / 1000e3],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -100e3, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -100e3, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -100e3, 30e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, 100e3, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                [0, 0, 1],
                boost_u_llim,
                HGV_u_llim,
            ],
            u_ulim=[
                [0, 0, 1],
                boost_u_ulim,
                HGV_u_ulim,
            ],
            x_scale=[
                [100e3, 1e3, 50e3, 5000, 1, 10, mass_prop],
                [100e3, 1e3, 50e3, 5000, 1, 10, mass_prop],
                [1000e3, 100e3, 100e3, 5000, 1, 10, 1],
            ],
            u_scale=[[0.01, 1, 1], [0.01, 1, 1], [0.1, 1]],
            t_scale=[1, 100, 500],
            max_time=[3, 1e6, 1e6],
            min_time=[3, 1, 1],
            stages=3,
            u_rate_lim=[
                boost_u_rate_lim,
                boost_u_rate_lim,
                HV_u_rate_lim,
            ],
        )

        return sprint


class GroundLaunchw0(CasadiProblem):
    def __init__(self, name: str = "ground_launch_w0", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem, boost_stages=1, HGV_stages=1):
        x = system.x
        u = system.u
        t = system.t

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        m_prop = system.fun["m_prop"]
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )

        #  Calc properties to get vmax in terms of Mach no
        z0 = 100
        v0 = 50

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v)
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 2 ** (a * (qa - lim) / 1e6) / 1e2

        ground_launch_w0 = Mission(
            name="ground_launch_w0",
            end_cost=[0],
            int_cost=[
                [
                    q_alpha(x[2], x[3], u[0][0], -168e3, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 168e3, lim_type="upper")
                ],
                [0],
            ],
            ineq=[
                *[
                    [
                        (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                        / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                        (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                        / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                        # 125e3 - f_q(x[2], x[3]),
                        mach_max - mach_number,
                    ]
                ]
                * boost_stages,
                *[
                    [
                        (100e3 - f_q(x[2], x[3])),
                        mach_max - mach_number,
                        mach_number - mach_min,
                    ]
                ]
                * HGV_stages,
            ],
            term=[],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0, 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf, 0],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                *[[-ca.inf, -ca.inf, 0e3, 0, -ca.inf, -ca.inf, 0, 0]] * boost_stages,
                *[[-ca.inf, -ca.inf, 30e3, 0, -ca.inf, -ca.inf, 0, 0]] * HGV_stages,
            ],
            x_ulim=[ca.inf, ca.inf, 60e3, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                *[boost_u_llim] * boost_stages,
                *[HGV_u_llim] * HGV_stages,
            ],
            u_ulim=[
                *[boost_u_ulim] * boost_stages,
                *[HGV_u_ulim] * HGV_stages,
            ],
            x_scale=[
                *[[200e3, 10e3, 50e3, 2000, 1, 10, mass_prop, 100e3]] * boost_stages,
                *[[2000e3, 10e3, 60e3, 2000, 1, 10, 1, 100e3]] * HGV_stages,
            ],
            u_scale=[
                *[[0.01, 1, 0.1]] * boost_stages,
                *[[0.1, 1]] * HGV_stages,
            ],
            t_scale=[
                *[100] * boost_stages,
                *[500] * HGV_stages,
            ],
            max_time=1e6,
            min_time=50,
            stages=boost_stages + HGV_stages,
            u_rate_lim=[
                *[boost_u_rate_lim] * boost_stages,
                *[HV_u_rate_lim] * HGV_stages,
            ],
        )

        return ground_launch_w0

class FirstStageEnergyGroundLaunch(CasadiProblem):
    def __init__(self, name: str = "first_stage_energy_ground_launch", scaling: float = 1e-3):
        super().__init__(name, scaling)

    def __call__(self, system: DynamicSystem):
        x = system.x
        u = system.u
        t = system.t
        p = system.p

        f_a = system.fun["f_a"]
        f_q = system.fun["f_q"]
        f_rho = system.fun["f_rho"]
        m_prop = system.fun["m_prop"]
        m_HGV = system.fun["mass_fun"]["HV"](*ca.vertsplit(p))
        m_dry = system.fun["mass_fun"]["boost_dry"](*ca.vertsplit(p))
        mass_prop = float(
            system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
        )
        thermal_fun_boost = system.fun["thermal_fun"]["boost"]
        thermal_fun_HGV = system.fun["thermal_fun"]["HV"]

        #  Calc properties to get vmax in terms of Mach no
        z0 = 0
        v0 = 0
        m1 = m_HGV + m_dry

        mach_number = x[3] / f_a(x[2], eps_val=eps_val)

        def end_location_cost(x):
            x_t = 1800e3

            return (x - x_t)**2

        def q_alpha(z, v, alpha, lim, lim_type="lower"):
            q = f_q(z, v) / 1e3
            alpha_deg = alpha * 180 / np.pi
            qa = q * alpha_deg
            if lim_type == "lower":
                a = -1
            elif lim_type == "upper":
                a = 1

            return 5 ** (a * (qa - lim)) / 1e6
            # return 2 ** (a * (qa - lim) / 1e6) / 1e1 # Used for glide only optimised vehicles to converge

        def heat_load_cost(z, v, lim):
            rho = f_rho(z)
            q = thermal_fun_HGV(rho, v, *ca.vertsplit(p)) / 1e6

            return 5 ** (q - lim) / 1e6

        def E(z, v):
            return (0.5 * v**2 + 9.81 * z) * m1

        first_stage_energy_ground_launch = Mission(
            name="first_stage_energy_ground_launch",
            end_cost=[-E(x[2], x[3]) / 1e9],
            int_cost=[
                [0],
                [
                    q_alpha(x[2], x[3], u[0][0], -170, lim_type="lower")
                    + q_alpha(x[2], x[3], u[0][0], 170, lim_type="upper")
                    + heat_load_cost(x[2], x[3], 6)
                ],
            ],
            ineq=[
                [],
                [
                    (168e3 - f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    (168e3 + f_q(x[2], x[3]) * u[0][0] * 180 / np.pi)
                    / 100e3,  # Taken from page 575 of Design or Rockets and Space Launch Vehicles
                    # (200e3 - f_q(x[2], x[3]))/100e3,
                    mach_max - mach_number,
                    (6e6 - thermal_fun_boost(f_rho(x[2]), x[3], *ca.vertsplit(p))) / 6e6,
                ],
            ],
            term=[],
            # term=[(x[0] - 1800e3) / 1000e3],
            x0_llim=[0, 0, z0, v0, np.deg2rad(-45), np.deg2rad(45), 0],
            x0_ulim=[0, 0, z0, v0, np.deg2rad(45), np.deg2rad(90), ca.inf],
            init_ineq=[x[6] - m_prop, m_prop - x[6]],
            x_llim=[
                [-ca.inf, -100e3, 0e3, 0, -ca.inf, -ca.inf, 0],
                [-ca.inf, -100e3, 00e3, 0, -ca.inf, -ca.inf, 0],
            ],
            x_ulim=[ca.inf, 100e3, 60e3, ca.inf, ca.inf, ca.inf, ca.inf],
            u_llim=[
                [0, 0, 1],
                boost_u_llim,
            ],
            u_ulim=[
                [0, 0, 1],
                boost_u_ulim,
            ],
            x_scale=[
                [100e3, 1e3, 50e3, 5000, 1, 10, mass_prop],
                [100e3, 1e3, 50e3, 5000, 1, 10, mass_prop],
            ],
            u_scale=[[0.01, 1, 1], [0.01, 1, 1]],
            t_scale=[1, 100],
            max_time=[3, 1e6],
            min_time=[3, 1],
            stages=2,
            u_rate_lim=[
                boost_u_rate_lim,
                boost_u_rate_lim,
            ],
        )

        return first_stage_energy_ground_launch


