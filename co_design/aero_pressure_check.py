import numpy as np
import pyvista as pv
from hyperpanel import PanelSolver, logger, PanelScheme
from geometry_generation import PatchTag


def main():
    """
    This example case uses a previously generated HTV-like vehicle,
    which is loaded from file. The file contains the arrays for normals,
    areas, and centroids.
    The coefficient of pressure (cp) is calculated using either a single scheme,
    or a split scheme which applies different schemes above and below the
    specified angle threshold. Computing cp only requires normals, but the
    calculation of integral coefficients require panel areas, centroids, and
    the centre of mass of the vehicle.
    """

    split_scheme = True
    vehicle_file = "target_2_obstruction_ground_launch_F/Target_2_F_13_First_Stage.vtk"
    # vehicle_file = "Mission-Sprint 2 Boost Glide_Vehicle-Sprint 2 Boost-Glide_F/Sprint-2_Boost-Glider_First_Stage.vtk"
    logger.info(f"Beginning analysis of file: {vehicle_file}")

    # Load mesh data from file
    vehicle_mesh = pv.read(vehicle_file)
    # values = vehicle_mesh.cell_data["keys"]
    # mask = values != "DONT_CALC"
    # cell_ids = np.where(mask)[0]
    # vehicle_mesh = vehicle_mesh.extract_cells(cell_ids)
    normals = vehicle_mesh.cell_data["N"]
    areas = vehicle_mesh.cell_data["A"]
    centroids = vehicle_mesh.cell_data["C"]

    # Other geometric considerations
    CoM = [-0.7, 0, -0.05]  # Center of mass is estimate
    lift0_direction = [0, 0, -1]  # -z direction (by construction)
    drag0_direction = [-1, 0, 0]  # -x direction

    # Create panel solver
    if split_scheme:
        myScheme = [PanelScheme.PRANDTL_MEYER_INTERP, PanelScheme.OBLIQUE_SHOCK]
        angle_threshold = [0.0]
    else:
        myScheme = PanelScheme.OBLIQUE_SHOCK
        angle_threshold = None  # Can be neglected from PanelSolver init

    myPanelSolver = PanelSolver(
        myScheme,
        angle_thresholds=angle_threshold,
    )
    logger.info(f"Using {repr(myPanelSolver)}")
    myPanelSolver.set_geometry(
        normals,
        drag0_vector=drag0_direction,
        lift0_vector=lift0_direction,
        areas=areas,
        centroids=centroids,
        CoM=CoM,
    )

    # Compute solution
    AoA = np.deg2rad(0.0)
    mach = 2
    results_dict = myPanelSolver.compute_flow_state(mach, AoA)
    cL, cD, cM = myPanelSolver.compute_integral_coefficients(results_dict["cp"], AoA=AoA)
    logger.info(f"Coefficients: CL={cL:.5f}, CD={cD:.5f}, CM={cM:.5f}")

    # Save cp by adding to mesh
    vehicle_mesh.cell_data["cp"] = results_dict["cp"]
    vehicle_mesh.cell_data["mach_local"] = results_dict["mach_local"]
    vehicle_mesh.cell_data["vel_local"] = results_dict["unit_vel_vector"]
    vehicle_mesh.cell_data["temp_ratio"] = results_dict["temp_ratio"]
    vehicle_mesh.cell_data["SolKey"] = results_dict["key"]

    out_file = f"{vehicle_file[:-4]}_cp.vtk"
    logger.info(f"Saving results to file: {out_file}")
    vehicle_mesh.save(out_file)

    logger.info(f"Beginning analysis of file: {vehicle_file}...done")


if __name__ == "__main__":
    main()
