import casadi as ca

def nose_heat_flux(parameters):
    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]
    rho = ca.MX.sym("rho")
    v = ca.MX.sym("v")
    index = list(parameters.keys()).index("r0")

    """
    Taken from Eqn 1.169 in Hypersonic and High-Temperature Gas Dynamics
    Assuming hw/h0 = 0
    """
    M = 3
    N = 0.5
    C = 1.83e-8 * p[index]**(-0.5)

    q = (rho**N * v**M * C) * 100**2

    heat_fun = ca.Function("glider_heat_fun", [rho, v, *p], [q])

    return heat_fun

def zero_heat_flux(parameters):
    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]
    rho = ca.MX.sym("rho")
    v = ca.MX.sym("v")

    heat_fun = ca.Function("zero_heat_fun", [rho, v, *p], [0])

    return heat_fun
