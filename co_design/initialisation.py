from copy import copy
from hyperdynamics.mission import Mission
from hyperdynamics.transcription import LgrCollocation
from hyperdynamics.system import DynamicSystem
from hyperdynamics.utils import Result
from hyperdynamics.run_nlp import trajectory_optimisation
import numpy as np
import casadi as ca

from plot_traj import plot_from_result
from spaceplane_missions import BoostSpaceplaneFlyback_w0, SpaceplaneFlyback_w0
from missions import GroundLaunchw0

def initialisation(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True, boost_stages=1, HGV_stages=1):
    n = 50

    ts = np.array([0, *t_stages, t_init])

    m_prop = system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
    if mission.x0:
        x0 = np.array([mission.x0[:-2]]).T
        x0 = np.vstack((x0, np.array(m_prop)))
    else:
        x0 = np.array([0, 0, 20, 50, 0, np.deg2rad(85), float(m_prop)])
    u01 = np.array([[np.deg2rad(0), 0, 1]]).T
    u02 = np.array([[np.deg2rad(5), 0]]).T
    x01 = []
    x_in = x0
    t0 = []
    for i in range(boost_stages):
        ti = ts[i:i+2]
        x01.append(system.integrate(x_in, u01, ti, stage=1))
        x_in = x01[-1][:, -1]
        t0.append(ti)

    x01[-1][-2,-1] = 0
    x_in = copy(x01[-1][:, -1])
    x_in[-3] = 0
    x02 = []
    for i in range(HGV_stages):
        ti = ts[boost_stages+i:boost_stages+2+i]
        x02.append(system.integrate(x_in, u02, ti, stage=boost_stages+1))
        x_in = x02[-1][:, -1]
        t0.append(ti)

    x0 = [*x01, *x02]
    u0 = [*[np.hstack((u01, u01))]*boost_stages, *[np.hstack((u02, u02))]*HGV_stages]
    
    w0 = transcription.pack_w(x=x0, u=u0, t=t0, mission=mission)

    if plot:
        w_dict = transcription.unpack_w(ca.DM(w0), system, mission)
        x_int, u_int, t_int = transcription.interpolate(x=w_dict["x"], u=w_dict["u"], t=w_dict["tx"], system=system, n=3)
        plotting_attr = {
        "x_c": w_dict["x"],
        "u_c": w_dict["u"],
        "tx_c": w_dict["tx"],
        "tu_c": w_dict["tu"],
        "x_int": x_int,
        "u_int": u_int,
        "t_int": t_int,
    }
    result = Result(
        x_sol=w_dict["x"],
        u_sol=w_dict["u"],
        tx_sol=w_dict["tx"],
        tu_sol=w_dict["tu"],
        sensitivity=0,
        cost=0,
        w=0,
        lam_g=0,
        plotting_attr=plotting_attr,
        success=0,
        stages=system.stages,
    )
    plot_from_result(result, system, fname="Initial_guess.png")

    return w0

def initialisation_boost_spaceplane_flyback(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True):
    n = 50

    ts = np.array([10, *t_stages, t_init])

    m_prop = system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
    m_fuel = system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
    if mission.x0:
        x0 = np.array([mission.x0[:-2]]).T
        x0 = np.vstack((x0, np.array(m_prop), np.array(m_fuel)))
    else:
        x0 = np.array([0, 0, 20, 50, 0, np.deg2rad(85), float(m_prop), float(m_fuel)])
    u01 = np.array([[1, 0, 0]]).T
    u02 = np.array([[np.deg2rad(5), 0, 0.6]]).T
    u03 = np.array([[np.deg2rad(5), np.deg2rad(20), 0]]).T
    x01 = system.integrate(x0, u01, ts[:2], stage=1)
    x01[-2,-1] = 0
    x01_in = copy(x01)
    x01_in[-3, -1] = 0
    x02 = system.integrate(x01_in[:, -1], u02, ts[1:3], stage=2)
    x03 = system.integrate(x02[:, -1], u03, ts[2:], stage=3)

    x0 = [x01, x02, x03]
    u0 = [np.hstack((u01, u01)), np.hstack((u02, u02)), np.hstack((u03, u03))]
    t0 = [ts[:2], ts[1:3], ts[2:4]]
    
    w0 = transcription.pack_w(x=x0, u=u0, t=t0, mission=mission)

    if plot:
        w_dict = transcription.unpack_w(ca.DM(w0), system, mission)
        x_int, u_int, t_int = transcription.interpolate(x=w_dict["x"], u=w_dict["u"], t=w_dict["tx"], system=system, n=3)
        plotting_attr = {
        "x_c": w_dict["x"],
        "u_c": w_dict["u"],
        "tx_c": w_dict["tx"],
        "tu_c": w_dict["tu"],
        "x_int": x_int,
        "u_int": u_int,
        "t_int": t_int,
    }
    result = Result(
        x_sol=w_dict["x"],
        u_sol=w_dict["u"],
        tx_sol=w_dict["tx"],
        tu_sol=w_dict["tu"],
        sensitivity=0,
        cost=0,
        w=0,
        lam_g=0,
        plotting_attr=plotting_attr,
        success=0,
        stages=system.stages,
    )
    plot_from_result(result, system, fname="Initial_guess.png")

    return w0

def boost_spaceplane_flyback_w0(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True):
    problem_w0 = BoostSpaceplaneFlyback_w0()
    mission_w0 = problem_w0(system)

    w0 = initialisation_boost_spaceplane_flyback(
        transcription=transcription,
        mission=mission_w0,
        system=system,
        t_stages=t_stages,
        t_init=t_init,
    )

    print('Running mission with no cost to generate initial guess')

    result = trajectory_optimisation(
        system=system,
        mission=mission_w0,
        transcription=transcription,
        w0=w0,
    )

    w0 = transcription.pack_w(
        x=result.plotting_attr["x_int"],
        u=result.plotting_attr["u_int"],
        t=result.plotting_attr["t_int"],
        mission=mission,
    )

    plot_from_result(result, system, fname="Zero_cost.png")

    return w0

def initialisation_glider(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_init = 200, plot=True, stages=1):
    n = 50

    ts = np.linspace(0, t_init, 1+stages)

    if mission.x0:
        x_in = np.array([*mission.x0[:3], 2200, *mission.x0[4:]])
    else:
        x_in = np.array([0, 0, 50e3, 2200, 0, 0])
    u0 = np.array([[np.deg2rad(5), 0]]).T

    x0 = []
    t0 = []

    for i in range(stages):
        ti = ts[i:2+i]
        x0.append(system.integrate(x_in, u0, ti, stage=i))
        x_in = x0[-1][:, -1]
        t0.append(ti)

    u0 = [np.hstack((u0, u0))]*stages
    
    w0 = transcription.pack_w(x=x0, u=u0, t=t0, mission=mission)


    if plot:
        w_dict = transcription.unpack_w(ca.DM(w0), system, mission)
        x_int, u_int, t_int = transcription.interpolate(x=w_dict["x"], u=w_dict["u"], t=w_dict["tx"], system=system, n=3)
        plotting_attr = {
        "x_c": w_dict["x"],
        "u_c": w_dict["u"],
        "tx_c": w_dict["tx"],
        "tu_c": w_dict["tu"],
        "x_int": x_int,
        "u_int": u_int,
        "t_int": t_int,
    }
    result = Result(
        x_sol=w_dict["x"],
        u_sol=w_dict["u"],
        tx_sol=w_dict["tx"],
        tu_sol=w_dict["tu"],
        sensitivity=0,
        cost=0,
        w=0,
        lam_g=0,
        plotting_attr=plotting_attr,
        success=0,
        stages=system.stages,
    )
    plot_from_result(result, system, fname="Initial_guess.png")

    return w0

def initialisation_spaceplane_flyback(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True):
    n = 50

    ts = np.array([0, *t_stages, t_init])

    m_fuel = system.fun["mass_fun"]["HV_fuel"](*system.parameters.values())
    x0 = np.array([[0, 0, 50e3, 2700, 0, 0, float(m_fuel)]]).T
    u01 = np.array([[np.deg2rad(5), 0, 0.0]]).T
    u02 = np.array([[np.deg2rad(5), np.deg2rad(20), 0]]).T
    x01 = system.integrate(x0, u01, ts[:2], stage=1)
    x02 = system.integrate(x01[:, -1], u02, ts[1:], stage=2)

    x0 = [x01, x02]
    u0 = [np.hstack((u01, u01)), np.hstack((u02, u02))]
    t0 = [ts[:2], ts[1:]]
    
    w0 = transcription.pack_w(x=x0, u=u0, t=t0, mission=mission)

    if plot:
        w_dict = transcription.unpack_w(ca.DM(w0), system, mission)
        x_int, u_int, t_int = transcription.interpolate(x=w_dict["x"], u=w_dict["u"], t=w_dict["tx"], system=system, n=3)
        plotting_attr = {
        "x_c": w_dict["x"],
        "u_c": w_dict["u"],
        "tx_c": w_dict["tx"],
        "tu_c": w_dict["tu"],
        "x_int": x_int,
        "u_int": u_int,
        "t_int": t_int,
    }
    result = Result(
        x_sol=w_dict["x"],
        u_sol=w_dict["u"],
        tx_sol=w_dict["tx"],
        tu_sol=w_dict["tu"],
        sensitivity=0,
        cost=0,
        w=0,
        lam_g=0,
        plotting_attr=plotting_attr,
        success=0,
        stages=system.stages,
    )
    plot_from_result(result, system, fname="Initial_guess.png")

    return w0

def spaceplane_flyback_w0(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True):
    problem_w0 = SpaceplaneFlyback_w0()
    mission_w0 = problem_w0(system)

    w0 = initialisation_spaceplane_flyback(
        transcription=transcription,
        mission=mission_w0,
        system=system,
        t_stages=t_stages,
        t_init=t_init,
    )

    print('Running mission with no cost to generate initial guess')

    result = trajectory_optimisation(
        system=system,
        mission=mission_w0,
        transcription=transcription,
        w0=w0,
    )

    w0 = transcription.pack_w(
        x=result.plotting_attr["x_int"],
        u=result.plotting_attr["u_int"],
        t=result.plotting_attr["t_int"],
        mission=mission,
    )

    plot_from_result(result, system, fname="Zero_cost.png")

    return w0

def ground_launch_w0(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True, boost_stages=1, HGV_stages=1):
    problem_w0 = GroundLaunchw0()
    mission_w0 = problem_w0(system)

    w0 = initialisation(
        transcription=transcription,
        mission=mission_w0,
        system=system,
        t_stages=t_stages,
        t_init=t_init,
        boost_stages=boost_stages,
        HGV_stages=HGV_stages,
    )

    print('Running mission with no cost to generate initial guess')

    result = trajectory_optimisation(
        system=system,
        mission=mission_w0,
        transcription=transcription,
        w0=w0,
    )

    w0 = transcription.pack_w(
        x=result.plotting_attr["x_int"],
        u=result.plotting_attr["u_int"],
        t=result.plotting_attr["t_int"],
        mission=mission,
    )

    plot_from_result(result, system, fname="Zero_cost.png")

    return w0

def initialisation_boost_only(transcription: LgrCollocation, mission: Mission, system: DynamicSystem, t_stages: list, t_init = 200, plot=True, boost_stages=1):
    n = 50

    ts = np.array([0, *t_stages, t_init])

    m_prop = system.fun["mass_fun"]["boost_prop"](*system.parameters.values())
    if mission.x0:
        x0 = np.array([mission.x0[:-2]]).T
        x0 = np.vstack((x0, np.array(m_prop)))
    else:
        x0 = np.array([0, 0, 20, 50, 0, np.deg2rad(85), float(m_prop)])
    u01 = np.array([[np.deg2rad(0), 0, 1]]).T
    x01 = []
    x_in = x0
    t0 = []
    for i in range(boost_stages):
        ti = ts[i:i+2]
        x01.append(system.integrate(x_in, u01, ti, stage=1))
        x_in = x01[-1][:, -1]
        t0.append(ti)

    x01[-1][-2,-1] = 0
    x_in = copy(x01[-1][:, -1])
    x_in[-3] = 0

    x0 = x01
    u0 = [*[np.hstack((u01, u01))]*boost_stages]
    
    w0 = transcription.pack_w(x=x0, u=u0, t=t0, mission=mission)

    if plot:
        w_dict = transcription.unpack_w(ca.DM(w0), system, mission)
        x_int, u_int, t_int = transcription.interpolate(x=w_dict["x"], u=w_dict["u"], t=w_dict["tx"], system=system, n=3)
        plotting_attr = {
        "x_c": w_dict["x"],
        "u_c": w_dict["u"],
        "tx_c": w_dict["tx"],
        "tu_c": w_dict["tu"],
        "x_int": x_int,
        "u_int": u_int,
        "t_int": t_int,
    }
    result = Result(
        x_sol=w_dict["x"],
        u_sol=w_dict["u"],
        tx_sol=w_dict["tx"],
        tu_sol=w_dict["tu"],
        sensitivity=0,
        cost=0,
        w=0,
        lam_g=0,
        plotting_attr=plotting_attr,
        success=0,
        stages=system.stages,
    )
    plot_from_result(result, system, fname="Initial_guess.png")

    return w0


