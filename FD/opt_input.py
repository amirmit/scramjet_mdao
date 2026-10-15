"""Baseline case: the exact settings that used to live directly in
Opt_Main.py before the config-file refactor. Run with:

    python Opt_Main.py cases/baseline.py
"""

import numpy as np

d2r = np.deg2rad

case_name = 'mf_dV_SMvisc'
output_dir = '/home/s4679920/MDAO/Work/FD_compare/FD/BASE_check'
headless = False   # run without plot windows (figures are still saved)

nozzle_surrogate = 'v_3_1'
aoa_deg = 0.0


###################################
############ Objective ############
###################################
# objective = {"CD": dict(weight=1.0, scale=0.008)}
objective = {'mf_dV': {'weight': 1.0, 'scale': 0.6}}
# objective = {"T_W_ratio": dict(weight=1.0, scale=0.012)}
# objective = {"m_dot_f": dict(weight=1.0, scale=0.04567)}
# objective = {"T_W_ratio": dict(weight=0.5, scale=0.015),
#              "m_dot_f": dict(weight=0.5, scale=0.04567)}
# objective_name = "T_W_and_mf"
# objective = {"range": dict(weight=1.0, scale=23600)}



###################################
############### DVs ###############
###################################
# DV = {
#     "inlet_length_1": dict(flb=0.5, fub=1.2, x0=1.1),
#     "inlet_length_2": dict(flb=0.0001, fub=2, x0=0.3),
#     "inlet_theta_1": dict(flb=d2r(5), fub=d2r(10), x0=d2r(6)),
#     "inlet_theta_2": dict(flb=d2r(1), fub=d2r(10), x0=d2r(5)),
#     "throat_height": dict(flb=0.05, fub=0.12, x0=0.08),
#     "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(0.1)),
#     "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.1),
#     "engine_length": dict(flb=0.5, fub=1.5, x0=1),
#     "isolator_length": dict(flb=0.5, fub=3, x0=2),
#     "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.02),
#     "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.0),
#     "nozzle_angle": dict(flb=d2r(1), fub=d2r(25), x0=d2r(10)),
#     "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.5),
#     "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.2),
# }

################ Optimized version (don't remember which case) ##################
# DV = {
#     "inlet_length_1": dict(flb=0.5, fub=1.2, x0=0.8238),
#     "inlet_length_2": dict(flb=0.0001, fub=2, x0=1.2403),
#     "inlet_theta_1": dict(flb=d2r(5.0), fub=d2r(10), x0=d2r(7.6543)),
#     "inlet_theta_2": dict(flb=d2r(1.0), fub=d2r(10), x0=d2r(2.5829)),
#     "throat_height": dict(flb=0.05, fub=0.12, x0=0.12),
#     "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(3)),
#     "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.3),
#     "engine_length": dict(flb=0.5, fub=1.5, x0=0.5),
#     "isolator_length": dict(flb=0.5, fub=3, x0=0.5),
#     "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.1),
#     "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.0333),
#     "nozzle_angle": dict(flb=d2r(1.0), fub=d2r(25), x0=d2r(11.8310)),
#     "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.8),
#     "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.331),
# }

# DV = {
#     "inlet_length_1": dict(flb=0.5, fub=1.2, x0=0.83),
#     "inlet_length_2": dict(flb=0.0001, fub=2, x0=1.24),
#     "inlet_theta_1": dict(flb=d2r(5.0), fub=d2r(10), x0=d2r(7.65)),
#     "inlet_theta_2": dict(flb=d2r(1.0), fub=d2r(10), x0=d2r(2.58)),
#     "throat_height": dict(flb=0.05, fub=0.12, x0=0.1),
#     "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(0)),
#     "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.2),
#     "engine_length": dict(flb=0.5, fub=1.5, x0=0.5),
#     "isolator_length": dict(flb=0.5, fub=3, x0=1.0),
#     "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.1),
#     "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.5),
#     "nozzle_angle": dict(flb=d2r(1.0), fub=d2r(25), x0=d2r(15)),
#     "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.5),
#     "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.33),
# }

DV = {
    "inlet_length_1": dict(flb=0.5, fub=1.2, x0=0.83),
    "inlet_length_2": dict(flb=0.0001, fub=2, x0=1.24),
    "inlet_theta_1": dict(flb=d2r(5.0), fub=d2r(10), x0=d2r(7.65)),
    "inlet_theta_2": dict(flb=d2r(1.0), fub=d2r(10), x0=d2r(2.58)),
    "throat_height": dict(flb=0.05, fub=0.12, x0=0.1),
    "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(0)),
    "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.2),
    "engine_length": dict(flb=0.2, fub=1.5, x0=0.5),
    "isolator_length": dict(flb=0.8, fub=3, x0=1.0),
    "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.0),
    "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.5),
    "nozzle_angle": dict(flb=d2r(1.0), fub=d2r(25), x0=d2r(15)),
    "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.5),
    "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.33),
}

################# Finding new x0 ########################
# DV = {
#     "inlet_length_1": dict(flb=0.5, fub=1.2, x0=1.1),
#     "inlet_length_2": dict(flb=0.5, fub=2, x0=1.0),
#     "inlet_theta_1": dict(flb=d2r(5.0), fub=d2r(10), x0=d2r(6)),
#     "inlet_theta_2": dict(flb=d2r(1.0), fub=d2r(10), x0=d2r(5)),
#     "throat_height": dict(flb=0.05, fub=0.12, x0=0.08),
#     "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(0.1)),
#     "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.1),
#     "engine_length": dict(flb=0.5, fub=1.5, x0=0.7),
#     "isolator_length": dict(flb=0.5, fub=3.0, x0=1.0),
#     "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.02),
#     "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.0),
#     "nozzle_angle": dict(flb=d2r(1.0), fub=d2r(25), x0=d2r(20)),
#     "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.5),
#     "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.2),
# }
# DV = {
#     "inlet_length_1": dict(flb=0.5, fub=1.2, x0=0.8),
#     "inlet_length_2": dict(flb=0.0001, fub=2, x0=1.2),
#     "inlet_theta_1": dict(flb=d2r(5.0), fub=d2r(10), x0=d2r(8)),
#     "inlet_theta_2": dict(flb=d2r(1.0), fub=d2r(10), x0=d2r(3)),
#     "throat_height": dict(flb=0.05, fub=0.12, x0=0.12),
#     "engine_tilt_ang": dict(flb=d2r(0), fub=d2r(3), x0=d2r(1)),
#     "cowl_LE": dict(flb=0.02, fub=0.3, x0=0.1),
#     "engine_length": dict(flb=0.5, fub=1.5, x0=0.5),
#     "isolator_length": dict(flb=0.5, fub=3, x0=0.5),
#     "engine_h5_h3": dict(flb=1, fub=1.1, x0=1.05),
#     "nozzle_length": dict(flb=0.5, fub=3.5, x0=1.5),
#     "nozzle_angle": dict(flb=d2r(1.0), fub=d2r(25), x0=d2r(15)),
#     "nozzle_mid_frac": dict(flb=0.0, fub=0.8, x0=0.8),
#     "equivalence_ratio": dict(flb=0.1, fub=1.0, x0=0.35),
# }

######################### Warm start #################################
# DV = {
#     "inlet_length_1":    dict(flb=0.5,    fub=1.2,     x0=1.01198018768388),
#     "inlet_length_2":    dict(flb=0.0001, fub=2,       x0=0.4274271749730913),
#     "inlet_theta_1":     dict(flb=d2r(5), fub=d2r(10), x0=0.12277846103317144),
#     "inlet_theta_2":     dict(flb=d2r(1), fub=d2r(10), x0=0.05275422061629853),
#     "throat_height":     dict(flb=0.05,   fub=0.12,    x0=0.08717565327722676),
#     "engine_tilt_ang":   dict(flb=d2r(0), fub=d2r(3),  x0=0.013315415949376583),
#     "cowl_LE":           dict(flb=0.02,   fub=0.3,     x0=0.14725491686084338),
#     "engine_length":     dict(flb=0.5,    fub=1.5,     x0=0.834719014953732),
#     "isolator_length":   dict(flb=0.5,    fub=3,       x0=1.5181476878382132),
#     "engine_h5_h3":      dict(flb=1,      fub=1.1,     x0=1.046269767955522),
#     "nozzle_length":     dict(flb=0.5,    fub=3.5,     x0=1.102988246084457),
#     "nozzle_angle":      dict(flb=d2r(1), fub=d2r(25), x0=0.13681593008356455),
#     "nozzle_mid_frac":   dict(flb=0.0,    fub=0.8,     x0=0.4369475881981555),
#     "equivalence_ratio": dict(flb=0.1,    fub=1.0,     x0=0.31685784345616225),
# }
###################################
########### Constraints ###########
###################################
constraints = {
    "Outflow_M": dict(glb=1.15, gub=3.0),
    "Outflow_P": dict(glb=10e3, gub=500e3),
    "Outflow_T": dict(glb=None, gub=1, scale=2000),
    # "L_W_ratio": dict(glb=0.45, gub=0.55),
    "L_W_ratio": dict(glb=0.5, gub=None),
    "m_dot_f_ratio": dict(glb=0.0, gub=1.0),
    "P_exit_ratio": dict(glb=1.0, gub=None),
    "tail_z": dict(glb=0.001, gub=None),
    "internal_vol": dict(glb=0.81245, gub=None),
    # "theta_nose": dict(glb=d2r(5.0), gub=None),
    # "CD": dict(glb=-0.001, gub=0.001),
    "CD": dict(glb=None, gub=0.0),
}

###################################
############ Opt_param ############
###################################
opt_params = dict(
    plot_flg=True,
    engine_visc_flg=False,  # Engine_1D isolator/combustor skin friction (Cf inside HyperPro)
    aero_visc_flg=True,     # external-body friction drag (Df, this hypervisc term)
    cf_model="meador_smart",
    cf_params=dict(pkl_path="/home/s4679920/HyperMDAO/hypervisc/BuildRefTempModel/cf_grid_v1_q/cf_x_surrogate.pkl"),
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