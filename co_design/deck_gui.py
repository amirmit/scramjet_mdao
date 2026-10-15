"""Interactive GUI for exploring fitted scramjet_deck.py CasADi decks.

Build the deck ONCE (enter DVs + a training grid, click "Build Deck"), then
plot any quantity_catalog.CATALOG entry's value/sensitivity surface --
repaired (frozen) training points shown in a different color from
directly-computed ones -- without re-running the physics for every plot.

Calls the real production generate_scramjet_decks() (which internally calls
_casadi_interpolant_from_df()) for every plotted surface -- not a
reimplementation, same point as deck_smoketest.py's own docstring. The
scattered ground-truth points come from generate_scramjet_decks's own
raw_data_out (exactly what fed each deck's interpolant, after repair),
cached once at build time -- not from a separate evaluate_scramjet_point
pass, which can't know what got frozen to what and previously showed the
wrong value at repaired nodes (see the 2026-09-21 note near where
raw_cache is built, below).

USAGE
-----
    source /home/s4679920/MDAO/MDOenv/bin/activate
    cd /home/s4679920/MDAO/scramjet_mdao/co_design
    python deck_gui.py
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from geometry_generation import generate_scramjet_sj2d
from scramjet_deck import generate_scramjet_decks, PROPULSION_QUANTITIES
from quantity_catalog import CATALOG
from utilities import get_number
from hyperProp.Nozzle.nozzle_surrogate_model import load_surrogate_model
from hypervisc import get_friction_model

d2r = np.deg2rad

# The validated "small_test" DV set (PROJECT_STATE.md) as defaults.
DEFAULT_DVS = {
    "inlet_length_1": 1.1,
    "inlet_length_2": 0.3,
    "inlet_theta_1": d2r(6.0),
    "inlet_theta_2": d2r(5.0),
    "throat_height": 0.08,
    "engine_tilt_ang": d2r(0.1),
    "cowl_LE": 0.1,
    "engine_length": 1.0,
    "isolator_length": 1.0,
    "engine_h5_h3": 1.02,
    "nozzle_length": 1.0,
    "nozzle_angle": d2r(10.0),
    "nozzle_mid_frac": 0.5,
}
# These are entered/labeled directly in radians (matching the underlying
# DV values exactly) rather than auto-converted from degrees, to avoid any
# ambiguity about which unit a given entry box expects.
ANGLE_DVS = {"inlet_theta_1", "inlet_theta_2", "engine_tilt_ang", "nozzle_angle"}

DEFAULT_EQUIVALENCE_RATIO = 0.2
DEFAULT_ALTITUDE = 30e3
DEFAULT_AOA_DEG = "0,2,4,6"
DEFAULT_MACH = "6.0,6.5,7.0,7.5"
FINE_N = 50  # fine-mesh resolution per axis for the plotted surface


class DeckGUI:
    def __init__(self, root):
        self.root = root
        root.title("Scramjet deck explorer")

        self.dv_entries = {}
        self.decks = None
        self.dv_values = None
        self.dv_names = None
        self.aoa_grid = None
        self.mach_grid = None
        self.raw_cache = None       # {quantity: {(aoa,mach): (value, sens_list)}}
        self.repaired_nodes = None  # set of (aoa, mach)

        self._queue = queue.Queue()
        self._build_layout()

    # ---------------- layout ----------------
    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        ttk.Label(left, text="Geometry DVs", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        r = 1
        for name, default in DEFAULT_DVS.items():
            suffix = " (rad)" if name in ANGLE_DVS else ""
            ttk.Label(left, text=name + suffix).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(left, width=14)
            e.insert(0, f"{default:.6g}")
            e.grid(row=r, column=1, sticky="w")
            self.dv_entries[name] = e
            r += 1

        ttk.Label(left, text="equivalence_ratio").grid(row=r, column=0, sticky="w")
        self.er_entry = ttk.Entry(left, width=14)
        self.er_entry.insert(0, str(DEFAULT_EQUIVALENCE_RATIO))
        self.er_entry.grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(left, text="altitude [m]").grid(row=r, column=0, sticky="w")
        self.alt_entry = ttk.Entry(left, width=14)
        self.alt_entry.insert(0, str(DEFAULT_ALTITUDE))
        self.alt_entry.grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        ttk.Label(left, text="Training grid", font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Label(left, text="AoA grid [deg, csv]").grid(row=r, column=0, sticky="w")
        self.aoa_entry = ttk.Entry(left, width=14)
        self.aoa_entry.insert(0, DEFAULT_AOA_DEG)
        self.aoa_entry.grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(left, text="Mach grid [csv]").grid(row=r, column=0, sticky="w")
        self.mach_entry = ttk.Entry(left, width=14)
        self.mach_entry.insert(0, DEFAULT_MACH)
        self.mach_entry.grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        self.build_button = ttk.Button(left, text="Build Deck", command=self._on_build)
        self.build_button.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        self.status_label = ttk.Label(left, text="Not built yet.", wraplength=220, foreground="gray")
        self.status_label.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 10))
        r += 1

        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        ttk.Label(left, text="Plot", font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Label(left, text="Quantity").grid(row=r, column=0, sticky="w")
        self.quantity_var = tk.StringVar(value="P_out")
        self.quantity_box = ttk.Combobox(
            left, textvariable=self.quantity_var, values=list(CATALOG.keys()), state="disabled", width=13
        )
        self.quantity_box.grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(left, text="Sensitivity DV").grid(row=r, column=0, sticky="w")
        self.sens_dv_var = tk.StringVar(value="throat_height")
        self.sens_dv_box = ttk.Combobox(
            left, textvariable=self.sens_dv_var, values=list(DEFAULT_DVS.keys()), state="disabled", width=13
        )
        self.sens_dv_box.grid(row=r, column=1, sticky="w")
        r += 1
        self.plot_button = ttk.Button(left, text="Plot", command=self._on_plot, state="disabled")
        self.plot_button.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        r += 1

        self.fig = plt.Figure(figsize=(10, 5.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    # ---------------- build (threaded -- keeps the GUI responsive) ----------------
    def _on_build(self):
        try:
            dv_dict = {name: float(entry.get()) for name, entry in self.dv_entries.items()}
            equivalence_ratio = float(self.er_entry.get())
            altitude = float(self.alt_entry.get())
            aoa_grid = [d2r(float(x)) for x in self.aoa_entry.get().split(",") if x.strip()]
            mach_grid = [float(x) for x in self.mach_entry.get().split(",") if x.strip()]
        except ValueError as e:
            messagebox.showerror("Input error", f"Couldn't parse inputs: {e}")
            return
        if not aoa_grid or not mach_grid:
            messagebox.showerror("Input error", "AoA grid and Mach grid must both be non-empty.")
            return

        self.build_button.config(state="disabled")
        self.plot_button.config(state="disabled")
        self.quantity_box.config(state="disabled")
        self.sens_dv_box.config(state="disabled")
        n_pts = len(aoa_grid) * len(mach_grid)
        self.status_label.config(
            text=f"Building geometry + deck ({n_pts} grid points) -- this can take a while...",
            foreground="blue",
        )

        thread = threading.Thread(
            target=self._build_worker,
            args=(dv_dict, equivalence_ratio, altitude, aoa_grid, mach_grid),
            daemon=True,
        )
        thread.start()
        self.root.after(150, self._poll_build)

    def _build_worker(self, dv_dict, equivalence_ratio, altitude, aoa_grid, mach_grid):
        try:
            geom_out = generate_scramjet_sj2d(active_dv_dict=dv_dict, overrides={}, mesh_params=None, save=False)
            vehicle, fws, context = geom_out["vehicle"], geom_out["FWS_param_dict"], geom_out["context"]
            dv_names = list(fws.keys())
            dv_values = [get_number(v) for v in fws.values()]

            nozzle_surrogate = load_surrogate_model("v_3_1")
            cf_model = get_friction_model("constant", cf_value=0.001)

            repair_log = []
            raw_cache = {}
            decks = generate_scramjet_decks(
                vehicle=vehicle, fws=fws, context=context, overrides={},
                nozzle_surrogate=nozzle_surrogate, cf_model=cf_model,
                aoa_grid=aoa_grid, mach_grid=mach_grid,
                altitude=altitude, equivalence_ratio=equivalence_ratio,
                repair_log=repair_log, raw_data_out=raw_cache,
            )
            repaired_nodes = {(aoa, mach) for aoa, mach, _reason in repair_log}
            # raw_cache now holds {name: {(aoa,mach): (value, sens_list)}} for
            # every catalog quantity, AFTER repair -- exactly what fed each
            # deck's interpolant, cached once so switching which quantity/DV
            # to plot afterward is instant (no re-running physics). NOT an
            # independent evaluate_scramjet_point pass: that can't know what
            # got frozen to what (only generate_scramjet_decks does the
            # repairing), and an earlier version of this method did exactly
            # that, showing the pre-repair raw (unreliable) value at
            # repaired nodes instead of what the deck actually uses --
            # caught by the user noticing orange points that didn't
            # coincide with their source neighbor (2026-09-21).

            self._queue.put(("ok", dict(
                decks=decks, dv_values=dv_values, dv_names=dv_names,
                aoa_grid=aoa_grid, mach_grid=mach_grid,
                raw_cache=raw_cache, repaired_nodes=repaired_nodes,
            )))
        except Exception as e:
            self._queue.put(("error", str(e)))

    def _poll_build(self):
        try:
            status, payload = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll_build)
            return

        self.build_button.config(state="normal")
        if status == "error":
            self.status_label.config(text=f"Build failed: {payload}", foreground="red")
            messagebox.showerror("Build failed", payload)
            return

        self.decks = payload["decks"]
        self.dv_values = payload["dv_values"]
        self.dv_names = payload["dv_names"]
        self.aoa_grid = payload["aoa_grid"]
        self.mach_grid = payload["mach_grid"]
        self.raw_cache = payload["raw_cache"]
        self.repaired_nodes = payload["repaired_nodes"]

        n_total = len(self.aoa_grid) * len(self.mach_grid)
        n_rep = len(self.repaired_nodes)
        self.status_label.config(
            text=f"Built OK -- {n_total} grid points, {n_rep} repaired "
                 f"(propulsion near-choke/failure).",
            foreground="dark green",
        )

        self.quantity_box.config(state="readonly")
        self.sens_dv_box.config(state="readonly")
        self.plot_button.config(state="normal")

    # ---------------- plot (instant -- reads cached data, no physics) ----------------
    def _on_plot(self):
        if self.decks is None:
            return
        quantity = self.quantity_var.get()
        sens_dv = self.sens_dv_var.get()
        if quantity not in self.decks or sens_dv not in self.dv_names:
            messagebox.showerror("Plot error", "Pick a valid quantity and sensitivity DV.")
            return

        fn = self.decks[quantity]
        dv_idx = self.dv_names.index(sens_dv)
        dv_values = self.dv_values

        mach_fine = np.linspace(min(self.mach_grid), max(self.mach_grid), FINE_N)
        aoa_fine = np.linspace(min(self.aoa_grid), max(self.aoa_grid), FINE_N)
        MACH_MESH, AOA_MESH = np.meshgrid(mach_fine, aoa_fine)
        VALUE_MESH = np.zeros_like(MACH_MESH)
        SENS_MESH = np.zeros_like(MACH_MESH)

        # Exact (not approximate) per module docstring: fn is exactly
        # linear in each DV by construction of _casadi_interpolant_from_df's
        # Taylor correction.
        h = 1e-3 * max(abs(dv_values[dv_idx]), 1.0)
        dv_plus = list(dv_values); dv_plus[dv_idx] += h
        dv_minus = list(dv_values); dv_minus[dv_idx] -= h
        for i in range(MACH_MESH.shape[0]):
            for j in range(MACH_MESH.shape[1]):
                m, a = float(MACH_MESH[i, j]), float(AOA_MESH[i, j])
                VALUE_MESH[i, j] = float(fn(m, a, *dv_values))
                SENS_MESH[i, j] = (float(fn(m, a, *dv_plus)) - float(fn(m, a, *dv_minus))) / (2 * h)

        cache = self.raw_cache[quantity]
        is_propulsion_q = quantity in PROPULSION_QUANTITIES
        direct_m, direct_a, direct_v, direct_s = [], [], [], []
        rep_m, rep_a, rep_v, rep_s = [], [], [], []
        for (aoa, mach), (value, sens) in cache.items():
            bucket = (rep_m, rep_a, rep_v, rep_s) if (is_propulsion_q and (aoa, mach) in self.repaired_nodes) \
                else (direct_m, direct_a, direct_v, direct_s)
            bucket[0].append(mach); bucket[1].append(aoa); bucket[2].append(value); bucket[3].append(sens[dv_idx])

        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(MACH_MESH, AOA_MESH, VALUE_MESH, cmap="viridis", alpha=0.6, edgecolor="none")
        if direct_m:
            ax1.scatter(direct_m, direct_a, direct_v, color="red", s=45, label="training (direct)")
        if rep_m:
            ax1.scatter(rep_m, rep_a, rep_v, color="orange", s=45, label="training (repaired/frozen)")
        ax1.set_xlabel("Mach"); ax1.set_ylabel("alpha [rad]"); ax1.set_zlabel(quantity)
        ax1.set_title(quantity)
        ax1.legend(fontsize=8)

        ax2 = self.fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(MACH_MESH, AOA_MESH, SENS_MESH, cmap="plasma", alpha=0.6, edgecolor="none")
        if direct_m:
            ax2.scatter(direct_m, direct_a, direct_s, color="red", s=45, label="training (direct)")
        if rep_m:
            ax2.scatter(rep_m, rep_a, rep_s, color="orange", s=45, label="training (repaired/frozen)")
        ax2.set_xlabel("Mach"); ax2.set_ylabel("alpha [rad]"); ax2.set_zlabel(f"d{quantity}/d{sens_dv}")
        ax2.set_title(f"d{quantity}/d{sens_dv}")
        ax2.legend(fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    DeckGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
