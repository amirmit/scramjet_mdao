import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import pyvista as pv
import math

PLOT_3D = 0
PLOT_LD = 0
PLOT_FULL = 0
PLOT_CONVERGENCE = 0

# BASE_DIR = "./boost_spaceplane_flyback_A/"
# BASE_DIR = "./spaceplane_flyback_A/"
# BASE_DIR = "./target_obstruction_ground_launch_A"
# BASE_DIR = "./max_range_obstruction_ground_launch_B"
BASE_DIR = "./target_2_obstruction_glider_L"


def plot_convergence():
    df = pd.read_csv(BASE_DIR+"/sqp_res.csv")
    iters = list(range(len(df)))
    obj = df["J"].values
    vol = df["vol"].values
    ld_rat = df["LD"].values
    width = df["width"].values
    height = df["height"].values
    stab_HV = df["stab_HV"].values
    # stab_1st = df["stab_1st"].values

    # obj[1] = 0
    # obj[22] = 0

    # print(f"Min: {np.min(obj)}")
    # print(f"Iteration: {np.argmin(obj)}")
    print(f"J: {df['J']}")

    # print(f"mass: {df["mass"]}")
    print(f"stab: {df['stab_HV']}")

    fig, ax = plt.subplots(6, 1, sharex=True)

    if obj[-1] == 0:
        ax[0].plot(iters[:-1], obj[:-1], "-", marker="o")
    else:
        ax[0].plot(iters, obj, "-", marker="o")
    # ax[0].set_ylim([np.min(obj)-1,0])

    vol_col = np.where(vol >= 4, "b", "r")
    stab_col_2nd = np.where(stab_HV <= 0, "b", "r")
    # stab_col_1st = np.where(stab_1st <= 0, "b", "r")
    ax[1].scatter(iters, vol, c=vol_col)
    ax[2].plot(iters, height, "-", marker="o")
    ax[3].plot(iters, width, "-", marker="o")
    ax[4].scatter(iters, stab_HV, c=stab_col_2nd)
    # ax[5].scatter(iters, stab_1st, c=stab_col_1st)
    # ax[5].plot(iters, ld_rat, "-", marker="o")

    ax[0].set_ylabel("Objective")
    ax[1].set_ylabel("Volume")
    ax[2].set_ylabel("Height")
    ax[3].set_ylabel("Width")
    ax[4].set_ylabel("Stability margin Glider")
    ax[5].set_ylabel("Stability margin 1st Stage")
    ax[-1].set_xlabel("Iteration No.")

    # Parameters
    param_init = pd.read_csv(BASE_DIR+"/parameters.csv", index_col=1)
    p_len = len(param_init.index)
    plot_num = math.ceil(p_len/10)
    fig2, ax2 = plt.subplots(plot_num, 1)

    for j in range(plot_num):
        for i in param_init.index[j*10:(j+1)*10]:
            p_norm = (df[i].values - param_init["llim"][i])/ (param_init["ulim"][i] - param_init["llim"][i])
            ax2[j].plot(iters, p_norm, label=i)
        ax2[j].set_ylabel("Normalised parameter values")
        ax2[j].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.1, 1))
    ax2[-1].set_xlabel("Iteration No.")

    print("Final Parameters:")
    for i in param_init.index:
        print(f"{i}: {df[i].values[-1]}")

    print("Final Sensitivities:")
    for i in param_init.index:
        name = f"dJ/d{i}"
        print(f"{name}: {df[name].values[-1]}")

    # Save parameter
    # param_dict = {}
    # iter = 4
    # for i in param_init.index:
    #     param_dict[i] = df[i][iter]
    # # Serializing json
    # json_object = json.dumps(param_dict)

    # # Writing to sample.json
    # with open(f"parameters_{iter}.json", "w") as outfile:
    #     outfile.write(json_object)


def compare_results(results_csv: list):
    for result_csv in results_csv:
        result_df = pd.read_csv(result_csv)



if __name__ == "__main__":
    plt.ion()
    plot_convergence()
    plt.show()
    input("DONE")

    # results = ["/home/jack_williamson/Codesign-Propulsion/multistage_codesign/max_range_obstruction_ground_launch_Test_F_vehicle-obstruction_problem/trajectory-final.csv", ]
