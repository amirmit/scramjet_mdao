"""Baseline case: the exact settings that used to live directly in
Opt_Main.py before the config-file refactor. Run with:

    python Opt_Main.py cases/baseline.py
"""

import numpy as np

d2r = np.deg2rad

case_name = "mf_dV_Visc"

nozzle_surrogate = "v3_1"  # M=6.5, matching this case's original hardcoded freestream
aoa_deg = 0.0

###################################
############ Objective ############
###################################
# objective = {"CD": dict(weight=1.0, scale=0.0057)}
objective = {"mf_dV": dict(weight=1.0, scale=4.9)}
# objective = {"T_W_ratio": dict(weight=1.0, scale=0.015)}
# objective = {"m_dot_f": dict(weight=1.0, scale=0.04567)}
# objective = {"T_W_ratio": dict(weight=0.5, scale=0.015),
#              "m_dot_f": dict(weight=0.5, scale=0.04567)}
# objective_name = "T_W_and_mf"
# objective = {"range": dict(weight=1.0, scale=47291.21)}



###################################
############### DVs ###############
###################################
DV = {
    "inlet_length_1": dict(flb=0.5, fub=1.2, x0=1.1),
    "inlet_length_2": dict(flb=0.0001, fub=2, x0=0.3),
    "inlet_theta_1": dict(flb=d2r(5.0), fub=d2r(10), x0=d2r(6)),
    "inlet_theta_2": dict(flb=d2r(1.0), fub=d2r(10), x0=d2r(5)),
    "throat_height": dict(flb=0.05, fub=0.12, x0=0.08),
    "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(0.1)),
    "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.1),
    "engine_length": dict(flb=0.5, fub=1.5, x0=1),
    "isolator_length": dict(flb=0.5, fub=3, x0=2),
    "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.02),
    "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.0),
    "nozzle_angle": dict(flb=d2r(1.0), fub=d2r(25), x0=d2r(10)),
    "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.5),
    "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.2),
}

###################################
########### Constraints ###########
###################################
constraints = {
    "Outflow_M": dict(glb=1.15, gub=3.0),
    "Outflow_P": dict(glb=10e3, gub=500e3),
    "Outflow_T": dict(glb=None, gub=1, scale=2000),
    "L_W_ratio": dict(glb=0.5, gub=None),
    "m_dot_f_ratio": dict(glb=0.0, gub=1.0),
    "P_exit_ratio": dict(glb=1.0, gub=None),
    "tail_z": dict(glb=0, gub=None),
    "internal_vol": dict(glb=0.81245, gub=None),
    # "theta_nose": dict(glb=d2r(5.0), gub=None),
    "CD": dict(glb=None, gub=0.0),
}

###################################
############ Opt_param ############
###################################
opt_params = dict(
    plot_flg=True,
    engine_visc_flg=False,  # Engine_1D isolator/combustor skin friction (Cf inside HyperPro)
    aero_visc_flg=True,     # external-body friction drag (Df, this hypervisc term)
    cf_model="constant",
    cf_params=dict(cf_value=0.001),
    flight_duration=10,
    eta_pr=0.8,
)


###################################
############## SNOPT ##############
###################################
snopt_options = {
    "Major optimality tolerance": 1.0e-4,
    "Major feasibility tolerance": 1.0e-4,
    "Major step limit": 0.03,
    "Hessian full memory": True,
    "Verify level": -1,
    # "maximize": True,
}
