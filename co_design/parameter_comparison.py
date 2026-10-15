from codesignTools import ParameterSet, Parameter
import numpy as np
import pandas as pd
import os


def main(sim_names: list[str], dir: str, iter: int =-1):
    working_dir = dir
    base_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    # os.chdir(working_dir)

    par_set = ParameterSet.from_csv(f"{sim_names[0]}/parameters.csv")

    theta=[]
    theta_norm=[]
    for i, sim in enumerate(sim_names):
        theta.append(load_design(f"{sim}/sqp_res.csv", par_set, iter=iter))
        theta_norm.append(par_set.non_dimensionalise_theta(theta[i]))

    diff_array = np.empty((len(theta[0]) + len(sim_names), len(sim_names)))
    for i, norm_A in enumerate(theta_norm):
        for j, norm_B in enumerate(theta_norm):
            diff_array[j, i]= np.linalg.norm(norm_A - norm_B) / np.sqrt(len(theta[0]))

        diff_array[j+1:, i] = norm_A

    col_labels = sim_names
    row_labels = [*sim_names, *par_set.names]
    diff_df = pd.DataFrame(diff_array, index=row_labels, columns=col_labels)
    diff_df.to_csv(f"{dir}/parameter_comparison.csv")


def load_design(filename: str, par_set: ParameterSet, iter: int = -1) -> np.ndarray:
    df = pd.read_csv(filename)

    design_values = []
    if iter < 0:
        n = len(df) - 1
    else:
        n = iter
    design_values = [df[var][n] for var in par_set.names]
    return np.array(design_values)

if __name__ == "__main__":
    fname = "Target2_Comparision"
    sim_names = ["max_range_obstruction_ground_launch_A", "max_range_obstruction_glider_A", "max_range_obstruction_glider_B"] #, "max_range_obstruction_glider_C"]


    main(sim_names, fname, iter=-1)
