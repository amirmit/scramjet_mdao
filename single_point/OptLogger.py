import json
import numpy as np
from collections import defaultdict


class OptLogger:
    def __init__(self, objective_name="CD"):
        self.objective_name = objective_name
        self.Objective = {
            "x": defaultdict(list),
            "Obj": [],
            "Con": defaultdict(list),
            "inlet_model": [],
            "geom": [],
            "pressure_recovery": [],
            "nozzle_diag": [],
            "comp_force": [],
            "engine_perf": [],
            "vehicle": [],
            "freestream": [],
            "out_params": [],
            "range_mod": [],
        }
        self.Sensitivity = {
            "x": defaultdict(list),
            "Sens": defaultdict(list),
        }

    def log_obj(self, xdict, funcs):
        """Called from objFunc to store objective-side data."""
        # breakpoint()
        for key, val in xdict.items():
            if key not in self.Objective["x"]:
                self.Objective["x"][key] = []
            self.Objective["x"][key].append(val)

        self.Objective["Obj"].append(funcs[self.objective_name])

        o = funcs["out4sens"]
        # pyoptsparse's np.real() turns the out4sens dict into a 0-d array.
        if isinstance(o, np.ndarray):
            o = o[()]
        q = o["quantities"]

        self.Objective["inlet_model"].append(o["inlet_model"])
        self.Objective["geom"].append(o["geom"])

        out4mat = o["out4mat"]
        self.Objective["pressure_recovery"].append(
            {station: out4mat[station]["pressure_recovery"] for station in out4mat}
        )
        self.Objective["nozzle_diag"].append(o["nozzle_diag"])
        self.Objective["comp_force"].append(o["comp_force"])

        self.Objective["engine_perf"].append(
            {
                "m_dot_air": out4mat["I"]["m_dot"],
                "m_dot_fuel": out4mat["E"]["m_dot_f"],
                "m_dot_f_lim": o["m_dot_f_lim"],
                "eta_th": out4mat["E"]["eta_th"],
            }
        )

        self.Objective["vehicle"].append(
            {
                "weight": o["weight"],
                "Sref": o["S_ref"],
                "volume": o["total_vol"],
                "internal_vol": q["internal_vol"],
                "eng_vol": o["eng_vol"],
            }
        )

        freestream = o["freestream"]
        self.Objective["freestream"].append(
            {
                "M": freestream.mach,
                "P": freestream.p,
                "T": freestream.T,
                "q": freestream.q_dyn,
            }
        )

        self.Objective["out_params"].append(
            {
                "T_W_ratio": q["T_W_ratio"],
                "CD": q["CD"],
            }
        )

        self.Objective["range_mod"].append(o["range_mod"])

    def log_sens(self, xdict, sens):
        """Called from sensFunc to store gradient-side data."""

        for key, val in xdict.items():
            if key not in self.Sensitivity["x"]:
                self.Sensitivity["x"][key] = []
            self.Sensitivity["x"][key].append(val)

        for key, val in sens[self.objective_name].items():
            if key not in self.Sensitivity["Sens"]:
                self.Sensitivity["Sens"][key] = []
            self.Sensitivity["Sens"][key].append(val)

    def save(self, filename):
        """Save everything to JSON once at the end."""
        if not filename.lower().endswith(".json"):
            filename += ".json"
        data = {
            "Objective": self.Objective,
            "Sens": self.Sensitivity,
        }
        data_json = convert_ndarrays(data)
        with open(filename, "w") as f:
            json.dump(data_json, f, indent=2)


def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
