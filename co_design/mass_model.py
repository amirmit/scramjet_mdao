import numpy as np
import casadi as ca


def HV_mass_function(vehicle, parameters, mass_dict, FWS_parameters, vehicle_cs=None):
    density = mass_dict["rho"]
    thickness = mass_dict["t"]
    fixed_mass = mass_dict["fm"]

    total_area = np.sum(vehicle.mesh_data.area)
    area = total_area.number
    area_sens = total_area.sens

    mass = area * density * thickness + fixed_mass
    mass_jac = area_sens * density * thickness

    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]

    for i, p_val in enumerate(parameters.values()):
        mass += (p[i] - p_val) * mass_jac[i]

    mass_fun = ca.Function("glider_mass_fun", [*p], [mass])

    com_model = CoMModel(vehicle.mesh_data.area, vehicle.mesh_data.centroid, mass_dict)

    if vehicle_cs:
        A_cs_avg = np.sum(vehicle_cs.mesh_data.area) / 5
        D_eq = (4 * A_cs_avg / np.pi)**0.5

        mass_2 = (0.04 * (FWS_parameters["length"] * 1000 / 25.4) * (D_eq*1000/25.4)**2) * 0.45392
        mass_2_jac = mass_2.sens
        mass_2 = mass_2.number

        for i, p_val in enumerate(parameters.values()):
            mass_2 += (p[i] - p_val) * mass_2_jac[i]

        mass_2_fun = ca.Function("glider_mass_2_fun", [*p], [mass_2])


    return mass_fun, com_model


#
# def spaceplane_mass_function(vehicle, parameters, rocket, mass_dict):
#     density = mass_dict["rho"]
#     thickness = mass_dict["t"]
#     fixed_mass = mass_dict["fm"]
#
#     total_area = np.sum(vehicle.mesh_data.area)
#     area = total_area.number
#     area_sens = total_area.sens
#
#     mass = area * density * thickness + fixed_mass + rocket.m
#     mass_jac = area_sens * density * thickness
#
#     p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]
#
#     for i, p_val in enumerate(parameters.values()):
#         mass += (p[i] - p_val) * mass_jac[i]
#
#     mass_fun = ca.Function("glider_mass_fun", [*p], [mass])
#
#     return mass_fun
#


def rocket_mass_function(mass, parameters, mass_jac, name="rocket_mass_fun"):
    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]

    for i, p_val in enumerate(parameters.values()):
        mass += (p[i] - p_val) * mass_jac[i]

    mass_fun = ca.Function(name, [*p], [mass])

    return mass_fun


def fuel_mass_function(vehicle, parameters, mass_opts, fuel_opts):
    """
    Function to calculate the volume of fuel in a HV given the vehicle volume,
    fuel volume percentage and fuel density.
    """
    fuel_vol = fuel_opts["vol_%"]
    rho = fuel_opts["rho"]
    V_prop = vehicle.mesh_data.calculate_volume(thickness=mass_opts["t"]) * fuel_vol
    fuel_mass = V_prop * rho

    fuel_mass_fun = rocket_mass_function(
        float(fuel_mass), parameters, fuel_mass.sens, name="fuel_mass_fun"
    )

    return fuel_mass_fun


def boost_components_mass_function(
    boost_vehicle, rocket_body_vehicle, parameters, mass_dict
):
    density = mass_dict["rho"]
    thickness = mass_dict["t"]
    fixed_mass = mass_dict["fm"]

    total_area = np.sum(boost_vehicle.mesh_data.area) - np.sum(
        rocket_body_vehicle.mesh_data.area
    )
    area = total_area.number
    area_sens = total_area.sens

    mass = area * density * thickness + fixed_mass
    mass_jac = area_sens * density * thickness

    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]

    for i, p_val in enumerate(parameters.values()):
        mass += (p[i] - p_val) * mass_jac[i]

    mass_fun = ca.Function("boost_components_mass_fun", [*p], [mass])

    com_model_boost = CoMModel(
        boost_vehicle.mesh_data.area, boost_vehicle.mesh_data.centroid, mass_dict
    )
    com_model_rocket = CoMModel(
        rocket_body_vehicle.mesh_data.area,
        rocket_body_vehicle.mesh_data.centroid,
        mass_dict,
    )
    com_boost_components = CoMModelSubtract(com_main=com_model_boost, subtract_coms=[com_model_rocket], subtract_offsets=[np.array([0, 0, 0])])

    return mass_fun, com_boost_components


def mass_function_add(mass_funs: list, parameters: dict, name="Combined_mass_fun"):
    """
    Function to add multiple Casadi mass functions
    """
    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]

    mass = 0

    for mass_fun in mass_funs:
        mass += mass_fun(*p)

    combined_mass_fun = ca.Function(name, [*p], [mass])

    return combined_mass_fun


def mass_function_substract(
    main_mass_fun,
    subtract_mass_funs: list,
    parameters: dict,
    name="substracted_mass_fun",
):
    """
    Function to subtract multiple mass functions from a main mass function
    """
    p = [ca.MX.sym(f"p{i}") for i in range(len(parameters))]

    mass = main_mass_fun(*p)

    for mass_fun in subtract_mass_funs:
        mass -= mass_fun(*p)

    subtracted_mass_fun = ca.Function(name, [*p], [mass])

    return subtracted_mass_fun


class CoMModel:
    # Note: Need to edit if I want fixed mass to be a parameter
    def __init__(
        self,
        area: np.ndarray,  # Shape (N,)
        centroid: np.ndarray,  # Shape (N,3)
        mass_dict: dict[str, float] | None = None,
    ):
        if mass_dict is None:
            mass_dict = default_dict

        rho = mass_dict["rho"]
        t = mass_dict["t"]
        fixed_mass = mass_dict["fm"]
        fixed_mass_loc = mass_dict["fm_loc"]
        rho_t = rho * t
        total_mass = np.sum(area) * rho_t + fixed_mass
        self.mass = total_mass

        CoM = (
            rho_t * np.sum(area[:, np.newaxis] * centroid, axis=0)
            + fixed_mass * fixed_mass_loc
        ) / total_mass
        self.cm = CoM
        # if hasattr(CoM[0], "sens"):
        #     self.cm_sens = np.array([x.sens for x in CoM])
        # else:
        #     self.cm_sens = np.zeros(CoM)

    def eval(self, _) -> np.ndarray:
        return self.cm.astype(float)

    def eval_jac(self, *args) -> np.ndarray:
        return np.array([x.sens for x in self.cm])

    def eval_mass(self, *args) -> float:
        return self.mass.astype(float)

    def eval_mass_jac(self, *args):
        return self.mass.sens


class CoMRocketModel(CoMModel):
    def __init__(self, CoM, mass):
        self.cm = CoM
        self.mass = mass


class CoMModelCombined(CoMModel):
    def __init__(
        self,
        coms: list[CoMModel],
        offsets: list[np.ndarray],
    ):
        if len(coms) != len(offsets):
            raise ValueError(
                f"Lengths of coms ({len(coms)}) and offsets ({len(offsets)}) lists do not match."
            )

        total_mass = 0
        mass_com_product = 0
        for com, offset in zip(coms, offsets):
            total_mass += com.mass
            mass_com_product += com.mass * (com.cm + offset)

        self.mass = total_mass
        self.cm = mass_com_product / total_mass


class CoMModelSubtract(CoMModel):
    def __init__(
        self,
        com_main: CoMModel,
        subtract_coms: list[CoMModel],
        subtract_offsets: list[np.ndarray],
    ):
        if len(subtract_coms) != len(subtract_offsets):
            raise ValueError(
                f"Lengths of coms ({len(subtract_coms)}) and offsets ({len(subtract_offsets)}) lists do not match."
            )

        total_mass = com_main.mass
        mass_com_product = com_main.mass * com_main.cm
        for com, offset in zip(subtract_coms, subtract_offsets):
            total_mass -= com.mass
            mass_com_product -= -com.mass * (com.cm + offset)

        self.mass = total_mass
        self.cm = mass_com_product / total_mass
