import numpy as np
import casadi as ca
from hyperrocket.rocket import Rocket, RocketEngine, SolidRocket, Falcon1e, Orion_50S_XL
from mass_model import rocket_mass_function, CoMRocketModel
from autodiff import SensMode, generate_parameters_with_sensitivity


def generate_liquid_rocket(parameters: dict, rocket_OG: Rocket):
    FWS_parameters = generate_parameters_with_sensitivity(parameters, SensMode.JACOBIAN)
    L_rocket = FWS_parameters["L_rocket"]
    D_rocket = FWS_parameters["D_rocket"]

    mass_structure = (
        rocket_OG.mass_structure
        * (L_rocket / rocket_OG.length)
        * (D_rocket / rocket_OG.diameter) ** 2
    )
    mass_prop = (
        rocket_OG.mass_prop
        * (L_rocket / rocket_OG.length)
        * (D_rocket / Falcon1e.diameter) ** 2
    )
    rocket = Rocket(
        name=f"{rocket_OG.name} Scaled",
        diameter=parameters["D_rocket"],
        length=parameters["L_rocket"],
        mass_structure=float(mass_structure),
        mass_prop=float(mass_prop),
        engine=rocket_OG.engine,
        engine_num=rocket_OG.engine_num,
    )

    # CoG
    mass_engines = rocket.engine.m * rocket.engine_num
    dry_cog = np.array(
        [
            -L_rocket
            * (mass_structure / 2 + mass_engines)
            / (mass_structure + mass_engines),
            0 * L_rocket,
            0 * L_rocket,
        ]
    )
    gross_cog = np.array(
        [
            -L_rocket
            * ((mass_structure + mass_prop) / 2 + mass_engines)
            / (mass_structure + mass_prop + mass_engines),
            0 * L_rocket,
            0 * L_rocket,
        ]
    )
    com_dry = CoMRocketModel(dry_cog, mass_structure + mass_engines)
    com_gross = CoMRocketModel(gross_cog, mass_structure + mass_engines + mass_prop)

    dry_mass_fun = rocket_mass_function(
        rocket.dry_mass(), parameters, mass_structure.sens, "dry_mass_fun"
    )
    prop_mass_fun = rocket_mass_function(
        float(mass_prop), parameters, mass_prop.sens, "prop_mass_fun"
    )
    thrust_fun = rocket_thrust_model(rocket)

    return thrust_fun, dry_mass_fun, prop_mass_fun, com_dry


def generate_solid_rocket(parameters: dict, rocket: SolidRocket):
    thrust_fun = solid_thrust_model(rocket)
    dry_mass_fun = rocket_mass_function(
        rocket.dry_mass(), parameters, np.zeros((len(parameters))), "dry_mass_fun"
    )
    prop_mass_fun = rocket_mass_function(
        rocket.mass_prop, parameters, np.zeros((len(parameters))), "prop_mass_fun"
    )

    dry_cog = np.array([-rocket.length / 2, 0, 0])
    com_dry = CoMRocketModel(dry_cog, rocket.dry_mass())

    return thrust_fun, dry_mass_fun, prop_mass_fun, com_dry


def rocket_thrust_model(
    rocket: Rocket | RocketEngine,
):
    """
    Function that create a Casadi function for the thrust of a rocket class
    """
    throttle = ca.MX.sym("throttle")
    p3 = ca.MX.sym("p3")
    t = ca.MX.sym("t")

    thrust = rocket.thrust(p3, throttle=throttle)
    mass_flow = rocket.mass_flow(throttle=throttle)

    return ca.Function(
        "Thrust",
        [p3, throttle, t],
        [thrust, mass_flow],
        ["p3", "throttle", "t"],
        ["thrust", "mass_flow"],
    )


def solid_thrust_model(rocket: SolidRocket):
    """
    Function that creates a Casadi function for the thrust of a solid rocket class
    """
    throttle = ca.MX.sym("throttle")
    p3 = ca.MX.sym("p3")
    t = ca.MX.sym("t")

    t_vec = np.linspace(0, rocket.t_burn + 10, 50)
    T_vec = rocket.thrust(p3=0, t=t_vec)
    mdot_vec = rocket.mass_flow(t=t_vec)

    thrust_interp = ca.interpolant("T_interp", "linear", [t_vec], T_vec)
    mdot_interp = ca.interpolant("mdot_interp", "linear", [t_vec], mdot_vec)

    return ca.Function(
        "Thrust",
        [p3, throttle, t],
        [thrust_interp(t) - rocket.A_e*p3, mdot_interp(t)],
        ["p3", "throttle", "t"],
        ["thrust", "mass_flow"],
    )


if __name__ == "__main__":
    parameters = {"L_rocket": 10, "D_rocket": 1.7, "length": 5}
    thrust, dry_maxx, prop_mass = generate_liquid_rocket(parameters)
