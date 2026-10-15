import numpy as np
import casadi as ca
from hyperdynamics.utils import f_p, f_a, f_q, f_g, f_rho, r0
from hyperdynamics.system import DynamicSystem

eps_val=np.inf
# eps_val=0.1

def trim_3DOF_boost_glide(parameters, aero_fun, mass_fun, thrust_fun, thermal_fun, boost_stages=2, HGV_stages=1):
    """
    Create ODE's for vehicle dynamics.
    """

    # Construct CasADi parameters
    casadi_params = {param: ca.MX.sym(param) for param in parameters}
    p = ca.vertcat(*casadi_params.values())

    # State variables
    t = ca.MX.sym("t")  # Time
    sx = ca.MX.sym("sx")  # x displacement - downrange projected to ground
    sy = ca.MX.sym("sy")  # y displacement - cross range, projected to ground
    sz = ca.MX.sym("sz")  # z vertical displacement - altitude above ground
    v = ca.MX.sym("v")  # velocity
    chi = ca.MX.sym("chi")  # Heading angle
    gamma = ca.MX.sym("gamma")  # Flight path angle
    m_prop = ca.MX.sym("m_prop")
    # Q = ca.MX.sym("Q")
    x = ca.vertcat(sx, sy, sz, v, chi, gamma, m_prop)
    x_dict = {"s_x": 0, "s_y": 1, "s_z": 2, "v": 3, "chi": 4, "gamma": 5, "m_prop": 6}

    # 1st Stage Control variables
    throttle = ca.MX.sym("throttle")  # Throttle
    alpha = ca.MX.sym("alpha")  # Angle of attack
    phi = ca.MX.sym("phi")  # Bank angle
    u1 = ca.vertcat(alpha, phi, throttle)
    u1_dict = {"alpha": 0, "phi": 1, "throttle": 2}

    # 2nd Stage Control variables
    u2 = ca.vertcat(alpha, phi)
    u2_dict = {"alpha": 0, "phi": 1}

    # Mass
    m_dry_rocket = mass_fun["boost_dry"](*casadi_params.values())
    m_prop_rocket = mass_fun["boost_prop"](*casadi_params.values())
    m_2nd = mass_fun["HV"](*casadi_params.values())
    m_1st = m_dry_rocket + m_2nd + m_prop

    # Aerodynamics
    mach = v / f_a(sz, eps_val=eps_val)
    cd_1st, cl_1st = aero_fun["boost"](mach, alpha, *casadi_params.values())
    cd_2nd, cl_2nd = aero_fun["HV"](mach, alpha, *casadi_params.values())

    # Propulsion
    p3 = f_p(sz, eps_val=eps_val)
    T1, m_dot1 = thrust_fun["boost"](p3, throttle, t)
    T2 = 0

    # Dynamic pressure
    q = f_q(sz, v, eps_val=eps_val)

    # Gravity
    g = f_g(sz)

    # Density 
    rho = f_rho(sz, eps_val=eps_val)

    # Heat flux
    q_flux1 = thermal_fun["boost"](rho, v, *casadi_params.values())
    q_flux2 = thermal_fun["HV"](rho, v, *casadi_params.values())

    # Dynamic equations
    sx_dot = ca.cos(gamma) * ca.cos(chi) * v * (r0 / (r0 + sz))
    sy_dot = ca.cos(gamma) * ca.sin(chi) * v * (r0 / (r0 + sz))
    sz_dot = ca.sin(gamma) * v
    v_dot_1st = ca.sin(gamma) * g + (T1 * ca.cos(alpha) - q * cd_1st) / m_1st
    v_dot_2nd = ca.sin(gamma) * g + (T2 * ca.cos(alpha) - q * cd_2nd) / m_2nd
    chi_dot_1st = (1 / (v * ca.cos(gamma))) * (
        ca.sin(phi) / m_1st * (q * cl_1st + T1 * ca.sin(alpha))
    )
    chi_dot_2nd = (1 / (v * ca.cos(gamma))) * (
        ca.sin(phi) / m_2nd * (q * cl_2nd + T2 * ca.sin(alpha))
    )
    gamma_dot_1st = (1 / v) * (
        ca.cos(gamma) * g
        + 1 / m_1st * (q * cl_1st + T1 * ca.sin(alpha))
        + v**2 * ca.cos(gamma) / (r0 + sz)
    )
    gamma_dot_2nd = (1 / v) * (
        ca.cos(gamma) * g
        + ca.cos(phi) / m_2nd * (q * cl_2nd + T2 * ca.sin(alpha))
        + v**2 * ca.cos(gamma) / (r0 + sz)
    )
    m_prop_dot_1st = -m_dot1
    m_prop_dot_2nd = 0
    # Q_dot_1st = q_flux1
    # Q_dot_2nd = q_flux2
    xdot_1st = ca.vertcat(
        sx_dot, sy_dot, sz_dot, v_dot_1st, chi_dot_1st, gamma_dot_1st, m_prop_dot_1st,
    )
    xdot_2nd = ca.vertcat(
        sx_dot, sy_dot, sz_dot, v_dot_2nd, chi_dot_2nd, gamma_dot_2nd, m_prop_dot_2nd,
    )

    xdot = [*[xdot_1st]*boost_stages, *[xdot_2nd]*HGV_stages]
    u = [*[u1]*boost_stages, *[u2]*HGV_stages]
    u_dict = [*[u1_dict]*boost_stages, *[u2_dict]*HGV_stages]
    stages = boost_stages + HGV_stages

    fun = {
        "f_Thrust": [T1, T2],
        "f_q": f_q,
        "f_a": f_a,
        "f_rho": f_rho,
        "f_g": f_g,
        "mass_fun": mass_fun,
        "aero_fun": aero_fun,
        "m_dot": m_dot1,
        "m_prop": m_prop_rocket,
        "thermal_fun": thermal_fun,
    }

    system = DynamicSystem(
        x,
        u,
        p,
        t,
        xdot,
        x_dict,
        u_dict,
        parameters,
        stages=stages,
        fun=fun,
    )

    return system, casadi_params

def trim_3DOF_boost_spaceplane_flyback(parameters, aero_fun, mass_fun, thrust_fun):
    """
    Create ODE's for vehicle dynamics.
    """

    # Construct CasADi parameters
    casadi_params = {param: ca.MX.sym(param) for param in parameters}
    p = ca.vertcat(*casadi_params.values())

    # State variables
    t = ca.MX.sym("t")  # Time
    sx = ca.MX.sym("sx")  # x displacement - downrange projected to ground
    sy = ca.MX.sym("sy")  # y displacement - cross range, projected to ground
    sz = ca.MX.sym("sz")  # z vertical displacement - altitude above ground
    v = ca.MX.sym("v")  # velocity
    chi = ca.MX.sym("chi")  # Heading angle
    gamma = ca.MX.sym("gamma")  # Flight path angle
    m_prop = ca.MX.sym("m_prop")
    m_fuel = ca.MX.sym("m_fuel")
    x = ca.vertcat(sx, sy, sz, v, chi, gamma, m_prop, m_fuel)
    x_dict = {"s_x": 0, "s_y": 1, "s_z": 2, "v": 3, "chi": 4, "gamma": 5, "m_prop": 6, "m_fuel": 7}

    # 1st Stage Control variables
    throttle = ca.MX.sym("throttle")  # Throttle
    alpha = ca.MX.sym("alpha")  # Angle of attack
    phi = ca.MX.sym("phi")  # Bank angle
    u1 = ca.vertcat(alpha, phi, throttle)
    u1_dict = {"alpha": 0, "phi": 1, "throttle": 2}

    # 2nd Stage Control variables
    u2 = ca.vertcat(alpha, phi, throttle)
    u2_dict = {"alpha": 0, "phi": 1, "throttle": 2}

    # Mass
    m_dry_rocket = mass_fun["boost_dry"](*casadi_params.values())
    m_prop_rocket = mass_fun["boost_prop"](*casadi_params.values())
    m_fuel_2nd = mass_fun["HV_fuel"](*casadi_params.values())
    m_2nd = mass_fun["HV"](*casadi_params.values()) + m_fuel
    m_1st = m_dry_rocket + m_2nd + m_prop

    # Aerodynamics
    mach = v / f_a(sz, eps_val=eps_val)
    cd_1st, cl_1st = aero_fun["boost"](mach, alpha, *casadi_params.values())
    cd_2nd, cl_2nd = aero_fun["HV"](mach, alpha, *casadi_params.values())

    # Propulsion
    p3 = f_p(sz, eps_val=eps_val)
    T1, m_dot1 = thrust_fun["boost"](p3, throttle, t)
    T2, m_dot2 = thrust_fun["HV"](p3, throttle, t)

    # Dynamic pressure
    q = f_q(sz, v, eps_val=eps_val)

    # Gravity
    g = f_g(sz)

    # Dynamic equations
    sx_dot = ca.cos(gamma) * ca.cos(chi) * v * (r0 / (r0 + sz))
    sy_dot = ca.cos(gamma) * ca.sin(chi) * v * (r0 / (r0 + sz))
    sz_dot = ca.sin(gamma) * v
    v_dot_1st = ca.sin(gamma) * g + (T1 * ca.cos(alpha) - q * cd_1st) / m_1st
    v_dot_2nd = ca.sin(gamma) * g + (T2 * ca.cos(alpha) - q * cd_2nd) / m_2nd
    chi_dot_1st = (1 / v * ca.cos(gamma)) * (
        ca.sin(phi) / m_1st * (q * cl_1st + T1 * ca.sin(alpha))
    )
    chi_dot_2nd = (1 / v * ca.cos(gamma)) * (
        ca.sin(phi) / m_2nd * (q * cl_2nd + T2 * ca.sin(alpha))
    )
    gamma_dot_1st = (1 / v) * (
        ca.cos(gamma) * g
        + 1 / m_1st * (q * cl_1st + T1 * ca.sin(alpha))
        + v**2 * ca.cos(gamma) / (r0 + sz)
    )
    gamma_dot_2nd = (1 / v) * (
        ca.cos(gamma) * g
        + ca.cos(phi) / m_2nd * (q * cl_2nd + T2 * ca.sin(alpha))
        + v**2 * ca.cos(gamma) / (r0 + sz)
    )
    m_prop_dot_1st = -m_dot1
    m_prop_dot_2nd = 0
    m_fuel_dot_1st = 0
    m_fuel_dot_2nd = -m_dot2
    xdot_1st = ca.vertcat(
        sx_dot, sy_dot, sz_dot, v_dot_1st, chi_dot_1st, gamma_dot_1st, m_prop_dot_1st, m_fuel_dot_1st,
    )
    xdot_2nd = ca.vertcat(
        sx_dot, sy_dot, sz_dot, v_dot_2nd, chi_dot_2nd, gamma_dot_2nd, m_prop_dot_2nd, m_fuel_dot_2nd,
    )
    xdot = [xdot_1st, xdot_2nd, xdot_2nd]

    fun = {
        "f_Thrust": [T1, T2],
        "f_q": f_q,
        "f_a": f_a,
        "f_rho": f_rho,
        "f_g": f_g,
        "mass_fun": mass_fun,
        "aero_fun": aero_fun,
        "m_dot": [m_dot1, m_dot2],
        "m_prop": m_prop_rocket,
        "m_fuel": m_fuel_2nd,
    }

    system = DynamicSystem(
        x,
        [u1, u2, u2],
        p,
        t,
        xdot,
        x_dict,
        [u1_dict, u2_dict, u2_dict],
        parameters,
        stages=3,
        fun=fun,
    )

    return system, casadi_params

def trim_3DOF_glide(parameters, aero_fun, mass_fun, thrust_fun, thermal_fun, stages=1):
    """
    Create ODE's for vehicle dynamics.
    """

    # Construct CasADi parameters
    casadi_params = {param: ca.MX.sym(param) for param in parameters}
    p = ca.vertcat(*casadi_params.values())

    # State variables
    t = ca.MX.sym("t")  # Time
    sx = ca.MX.sym("sx")  # x displacement - downrange projected to ground
    sy = ca.MX.sym("sy")  # y displacement - cross range, projected to ground
    sz = ca.MX.sym("sz")  # z vertical displacement - altitude above ground
    v = ca.MX.sym("v")  # velocity
    chi = ca.MX.sym("chi")  # Heading angle
    gamma = ca.MX.sym("gamma")  # Flight path angle
    x = ca.vertcat(sx, sy, sz, v, chi, gamma)
    x_dict = {"s_x": 0, "s_y": 1, "s_z": 2, "v": 3, "chi": 4, "gamma": 5}

    # 2nd Stage Control variables
    alpha = ca.MX.sym("alpha")  # Angle of attack
    phi = ca.MX.sym("phi")  # Angle of attack
    u = ca.vertcat(alpha, phi)
    u_dict = {"alpha": 0, "phi": 1}

    # Mass
    m = mass_fun["HV"](*casadi_params.values())

    # Aerodynamics
    mach = v / f_a(sz, eps_val=eps_val)
    cd_2nd, cl_2nd = aero_fun["HV"](mach, alpha, *casadi_params.values())

    # Propulsion
    p3 = f_p(sz, eps_val=eps_val)
    T2 = 0

    # Dynamic pressure
    q = f_q(sz, v, eps_val=eps_val)

    # Gravity
    g = f_g(sz)

    # Density 
    rho = f_rho(sz, eps_val=eps_val)

    # Heat Flux
    q_flux = thermal_fun["HV"](rho, v, *casadi_params.values())

    # Dynamic equations
    sx_dot = ca.cos(gamma) * ca.cos(chi) * v * (r0 / (r0 + sz))
    sy_dot = ca.cos(gamma) * ca.sin(chi) * v * (r0 / (r0 + sz))
    sz_dot = ca.sin(gamma) * v
    v_dot = ca.sin(gamma) * g + (T2 * ca.cos(alpha) - q * cd_2nd) / m
    chi_dot = (1 / v * ca.cos(gamma)) * (
        ca.sin(phi) / m * (q * cl_2nd + T2 * ca.sin(alpha))
    )
    gamma_dot = (1 / v) * (
        ca.cos(gamma) * g
        + ca.cos(phi) / m * (q * cl_2nd + T2 * ca.sin(alpha))
        + v**2 * ca.cos(gamma) / (r0 + sz)
    )
    xdot = ca.vertcat(
        sx_dot, sy_dot, sz_dot, v_dot, chi_dot, gamma_dot,
    )
    fun = {
        "f_q": f_q,
        "f_a": f_a,
        "f_rho": f_rho,
        "f_g": f_g,
        "mass_fun": mass_fun,
        "aero_fun": aero_fun,
        "thermal_fun": thermal_fun,
    }

    system = DynamicSystem(
        x,
        [u]*stages,
        p,
        t,
        [xdot]*stages,
        x_dict,
        [u_dict]*stages,
        parameters,
        stages=stages,
        fun=fun,
    )

    return system, casadi_params

def trim_3DOF_spaceplane_flyback(parameters, aero_fun, mass_fun, thrust_fun):
    """
    Create ODE's for vehicle dynamics.
    """

    # Construct CasADi parameters
    casadi_params = {param: ca.MX.sym(param) for param in parameters}
    p = ca.vertcat(*casadi_params.values())

    # State variables
    t = ca.MX.sym("t")  # Time
    sx = ca.MX.sym("sx")  # x displacement - downrange projected to ground
    sy = ca.MX.sym("sy")  # y displacement - cross range, projected to ground
    sz = ca.MX.sym("sz")  # z vertical displacement - altitude above ground
    v = ca.MX.sym("v")  # velocity
    chi = ca.MX.sym("chi")  # Heading angle
    gamma = ca.MX.sym("gamma")  # Flight path angle
    m_fuel = ca.MX.sym("m_fuel")
    x = ca.vertcat(sx, sy, sz, v, chi, gamma, m_fuel)
    x_dict = {"s_x": 0, "s_y": 1, "s_z": 2, "v": 3, "chi": 4, "gamma": 5, "m_fuel": 6}

    # 2nd Stage Control variables
    alpha = ca.MX.sym("alpha")  # Angle of attack
    throttle = ca.MX.sym("throttle")  # Throttle
    phi = ca.MX.sym("phi")  # Bank angle
    u = ca.vertcat(alpha, phi, throttle)
    u_dict = {"alpha": 0, "phi": 1, "throttle": 2}

    # Mass
    m_fuel_2nd = mass_fun["HV_fuel"](*casadi_params.values())
    m_2nd = mass_fun["HV"](*casadi_params.values()) + m_fuel

    # Aerodynamics
    mach = v / f_a(sz, eps_val=eps_val)
    cd, cl = aero_fun["HV"](mach, alpha, *casadi_params.values())

    # Propulsion
    p3 = f_p(sz, eps_val=eps_val)
    T, m_dot = thrust_fun["HV"](p3, throttle, t)

    # Dynamic pressure
    q = f_q(sz, v, eps_val=eps_val)

    # Gravity
    g = f_g(sz)

    # Dynamic equations
    sx_dot = ca.cos(gamma) * ca.cos(chi) * v * (r0 / (r0 + sz))
    sy_dot = ca.cos(gamma) * ca.sin(chi) * v * (r0 / (r0 + sz))
    sz_dot = ca.sin(gamma) * v
    v_dot = ca.sin(gamma) * g + (T * ca.cos(alpha) - q * cd) / m_2nd
    chi_dot = (1 / v * ca.cos(gamma)) * (
        ca.sin(phi) / m_2nd * (q * cl + T * ca.sin(alpha))
    )
    gamma_dot = (1 / v) * (
        ca.cos(gamma) * g
        + ca.cos(phi) / m_2nd * (q * cl + T * ca.sin(alpha))
        + v**2 * ca.cos(gamma) / (r0 + sz)
    )
    m_fuel_dot = -m_dot
    xdot = ca.vertcat(
        sx_dot, sy_dot, sz_dot, v_dot, chi_dot, gamma_dot, m_fuel_dot,
    )
    xdot = [xdot, xdot]

    fun = {
        "f_Thrust": T,
        "f_q": f_q,
        "f_a": f_a,
        "f_rho": f_rho,
        "f_g": f_g,
        "mass_fun": mass_fun,
        "aero_fun": aero_fun,
        "m_dot": m_dot,
        "m_fuel": m_fuel_2nd,
    }

    system = DynamicSystem(
        x,
        [u, u],
        p,
        t,
        xdot,
        x_dict,
        [u_dict, u_dict],
        parameters,
        stages=2,
        fun=fun,
    )

    return system, casadi_params


