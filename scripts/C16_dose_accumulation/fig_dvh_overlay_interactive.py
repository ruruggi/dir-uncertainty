"""
fig_dvh_overlay_interactive.py
==============================
Interactive viewer of the DVH overlay figure (Fig 7).

Lets the user choose legend position, ncol, font size, font family, line
width, title on/off and save DPI. Reuses the heavy DVH computation
(CT load + rasterisation + DVH of the K samples), caching it in an .npz so
that subsequent openings are immediate.

Usage:
    python fig_dvh_overlay_interactive.py [--ens DIR] [--recompute]

Shortcuts:
    Ctrl+S  = save PNG with the current settings
    Ctrl+R  = forced redraw
"""
from __future__ import annotations
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Reuse the computation of the non-interactive version
from fig_dvh_overlay import (
    ROOT, RS_PATH, RD_PLAN, STRUCTURES,
    load_pct_post_rigid, rasterize_roi, resample_plan_dose, dvh_cumulative,
)
import pydicom

LEGEND_LOCS = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "center left", "center right", "lower center", "upper center", "center",
    "outside right", "outside bottom", "none",
]
FONT_FAMILIES = ["sans-serif", "serif", "monospace", "DejaVu Sans", "Arial",
                 "Times New Roman", "Calibri", "Helvetica", "Verdana"]


def compute_or_load_cache(ens_dir, bins, dose_max, force_recompute):
    cache = os.path.join(ens_dir, "_dvh_cache.npz")
    if os.path.exists(cache) and not force_recompute:
        print(f"[cache] loading cached DVH from {cache}")
        d = np.load(cache, allow_pickle=True)
        centers = d["centers"]
        names   = list(d["names"])
        colors  = list(d["colors"])
        v_plan = d["v_plan"]
        v_mean = d["v_mean"]
        v_lo   = d["v_lo"]
        v_hi   = d["v_hi"]
        K = int(d["K"])
        results = {}
        for i, name in enumerate(names):
            results[name] = dict(
                color=colors[i],
                v_plan=v_plan[i], v_mean=v_mean[i],
                v_lo=v_lo[i], v_hi=v_hi[i],
            )
        return centers, results, K

    print(f"[compute] cache missing: recomputing DVH (may take a few minutes)...")
    ct, v2w_post, inv_predef = load_pct_post_rigid()
    nz, ny, nx = ct.shape
    rs = pydicom.dcmread(RS_PATH)
    masks = {}
    for name, _ in STRUCTURES:
        m = rasterize_roi(rs, name, ny, nx, nz, inv_predef, v2w_post)
        if m is None or m.sum() == 0:
            print(f"  [warn] {name} not found/empty — skip")
            continue
        masks[name] = m
        print(f"  {name:>15s}: {int(m.sum())} voxels")

    dose_mean = np.load(f"{ens_dir}/dose_accumulated_fc.npy").astype(np.float32)
    dose_per  = np.load(f"{ens_dir}/dose_acc_per_sample_fc.npy", mmap_mode="r")
    if dose_mean.shape != (nz, ny, nx) and dose_mean.shape == (nx, ny, nz):
        dose_mean = np.transpose(dose_mean, (2, 1, 0))
    K = dose_per.shape[0]
    dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)

    edges = np.linspace(0, dose_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    results, names_out, colors_out = {}, [], []
    v_plan_arr, v_mean_arr, v_lo_arr, v_hi_arr = [], [], [], []
    for name, color in STRUCTURES:
        if name not in masks: continue
        m = masks[name]
        v_plan = dvh_cumulative(dose_plan, m, edges)
        V_per  = np.empty((K, len(centers)), dtype=np.float32)
        for k in range(K):
            V_per[k] = dvh_cumulative(np.asarray(dose_per[k]), m, edges)
        v_mean = V_per.mean(axis=0)
        v_lo   = np.percentile(V_per, 2.5,  axis=0)
        v_hi   = np.percentile(V_per, 97.5, axis=0)
        results[name] = dict(color=color, v_plan=v_plan, v_mean=v_mean,
                              v_lo=v_lo, v_hi=v_hi)
        names_out.append(name); colors_out.append(color)
        v_plan_arr.append(v_plan); v_mean_arr.append(v_mean)
        v_lo_arr.append(v_lo);     v_hi_arr.append(v_hi)
        print(f"  computed DVH for {name}")

    np.savez_compressed(
        cache,
        centers=centers,
        names=np.array(names_out, dtype=object),
        colors=np.array(colors_out, dtype=object),
        v_plan=np.array(v_plan_arr),
        v_mean=np.array(v_mean_arr),
        v_lo=np.array(v_lo_arr),
        v_hi=np.array(v_hi_arr),
        K=K,
    )
    print(f"[cache] saved {cache}")
    return centers, results, K


class DVHViewer(tk.Tk):
    def __init__(self, centers, results, K, default_out):
        super().__init__()
        self.title("Fig 7 DVH overlay — interactive")
        self.geometry("1400x900")
        self.centers = centers
        self.results = results
        self.K = K
        self.default_out = default_out

        # --- State variables (with trace for live redraw) ---------------
        # STRUCTURES legend (larger, one colour per ROI)
        self.var_legend_loc   = tk.StringVar(value="upper right")
        self.var_legend_ncol  = tk.IntVar(value=1)
        self.var_legend_fs    = tk.IntVar(value=11)
        # STYLE legend (small, ensemble vs plan)
        self.var_style_show    = tk.BooleanVar(value=True)
        self.var_style_loc     = tk.StringVar(value="lower left")
        self.var_style_fs      = tk.IntVar(value=8)
        # Typography / curves / general
        self.var_title_fs     = tk.IntVar(value=12)
        self.var_label_fs     = tk.IntVar(value=12)
        self.var_tick_fs      = tk.IntVar(value=10)
        self.var_font_family  = tk.StringVar(value="sans-serif")
        self.var_lw_ens       = tk.DoubleVar(value=2.0)
        self.var_lw_plan      = tk.DoubleVar(value=1.5)
        self.var_band_alpha   = tk.DoubleVar(value=0.20)
        self.var_dose_max     = tk.DoubleVar(value=float(centers.max()))
        self.var_show_title   = tk.BooleanVar(value=True)
        self.var_show_grid    = tk.BooleanVar(value=True)
        self.var_show_pdose   = tk.BooleanVar(value=True)
        self.var_save_dpi     = tk.IntVar(value=800)

        all_vars = [
            self.var_legend_loc, self.var_legend_ncol, self.var_legend_fs,
            self.var_style_show, self.var_style_loc, self.var_style_fs,
            self.var_title_fs, self.var_label_fs, self.var_tick_fs,
            self.var_font_family, self.var_lw_ens, self.var_lw_plan,
            self.var_band_alpha, self.var_dose_max, self.var_show_title,
            self.var_show_grid, self.var_show_pdose,
        ]
        for v in all_vars:
            v.trace_add("write", lambda *a: self._schedule_redraw())

        self._build_ui()
        self._redraw_pending = False
        self._redraw()

        self.bind_all("<Control-s>", lambda e: self.on_save())
        self.bind_all("<Control-r>", lambda e: self._redraw())

    def _build_ui(self):
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # === Canvas (left) =====
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left)
        toolbar.update()

        # === Controls (right): tabbed Notebook =====
        ctrl_container = ttk.Frame(main, padding=4)
        ctrl_container.pack(side="right", fill="y")
        nb = ttk.Notebook(ctrl_container, width=360)
        nb.pack(fill="both", expand=True)

        def row_in(parent, lbl, widget):
            f = ttk.Frame(parent); f.pack(fill="x", pady=2)
            ttk.Label(f, text=lbl, width=16).pack(side="left")
            widget.pack(side="left", fill="x", expand=True)

        # --- Tab 1: Legends ("Legende") ---------------------------------
        tab_leg = ttk.Frame(nb, padding=8); nb.add(tab_leg, text="Legende")
        leg_struct_box = ttk.LabelFrame(tab_leg,
            text="LEGENDA STRUTTURE (grande)", padding=6)
        leg_struct_box.pack(fill="x", pady=(0,8))
        row_in(leg_struct_box, "Posizione",
               ttk.Combobox(leg_struct_box, textvariable=self.var_legend_loc,
                             values=LEGEND_LOCS, width=16, state="readonly"))
        row_in(leg_struct_box, "Colonne (ncol)",
               ttk.Spinbox(leg_struct_box, from_=1, to=4,
                            textvariable=self.var_legend_ncol, width=6))
        row_in(leg_struct_box, "Font",
               ttk.Spinbox(leg_struct_box, from_=4, to=36,
                            textvariable=self.var_legend_fs, width=6))

        leg_style_box = ttk.LabelFrame(tab_leg,
            text="LEGENDA STILE (piccola, ens vs plan)", padding=6)
        leg_style_box.pack(fill="x")
        row_in(leg_style_box, "Mostra",
               ttk.Checkbutton(leg_style_box, variable=self.var_style_show))
        row_in(leg_style_box, "Posizione",
               ttk.Combobox(leg_style_box, textvariable=self.var_style_loc,
                             values=LEGEND_LOCS, width=16, state="readonly"))
        row_in(leg_style_box, "Font",
               ttk.Spinbox(leg_style_box, from_=4, to=36,
                            textvariable=self.var_style_fs, width=6))
        ttk.Label(tab_leg,
            text="Suggerimento: imposta font/colonne dal pannello, poi\n"
                 "trascina le legende sul plot. Ctrl+S per salvare.",
            foreground="gray", justify="left").pack(anchor="w", pady=(10,0))

        # --- Tab 2: Typography ("Tipografia") ---------------------------
        tab_typ = ttk.Frame(nb, padding=8); nb.add(tab_typ, text="Tipografia")
        row_in(tab_typ, "Famiglia font",
               ttk.Combobox(tab_typ, textvariable=self.var_font_family,
                             values=FONT_FAMILIES, width=16))
        row_in(tab_typ, "Font titolo",
               ttk.Spinbox(tab_typ, from_=6, to=36,
                            textvariable=self.var_title_fs, width=6))
        row_in(tab_typ, "Font assi",
               ttk.Spinbox(tab_typ, from_=6, to=36,
                            textvariable=self.var_label_fs, width=6))
        row_in(tab_typ, "Font tick",
               ttk.Spinbox(tab_typ, from_=4, to=24,
                            textvariable=self.var_tick_fs, width=6))
        row_in(tab_typ, "Mostra titolo",
               ttk.Checkbutton(tab_typ, variable=self.var_show_title))

        # --- Tab 3: Curve -------------------------------------------------
        tab_curv = ttk.Frame(nb, padding=8); nb.add(tab_curv, text="Curve")
        row_in(tab_curv, "LW ensemble",
               ttk.Spinbox(tab_curv, from_=0.5, to=5.0, increment=0.25,
                            textvariable=self.var_lw_ens, width=6))
        row_in(tab_curv, "LW plan (--)",
               ttk.Spinbox(tab_curv, from_=0.5, to=5.0, increment=0.25,
                            textvariable=self.var_lw_plan, width=6))
        row_in(tab_curv, "Alpha banda",
               ttk.Spinbox(tab_curv, from_=0.0, to=1.0, increment=0.05,
                            textvariable=self.var_band_alpha, width=6))
        row_in(tab_curv, "Dose max [Gy]",
               ttk.Spinbox(tab_curv, from_=10.0, to=200.0, increment=5.0,
                            textvariable=self.var_dose_max, width=6))
        row_in(tab_curv, "Mostra plan (--)",
               ttk.Checkbutton(tab_curv, variable=self.var_show_pdose))
        row_in(tab_curv, "Mostra grid",
               ttk.Checkbutton(tab_curv, variable=self.var_show_grid))

        # --- Tab 4: Export ------------------------------------------------
        tab_exp = ttk.Frame(nb, padding=8); nb.add(tab_exp, text="Export")
        row_in(tab_exp, "DPI salvataggio",
               ttk.Combobox(tab_exp, textvariable=self.var_save_dpi,
                             values=[100, 200, 300, 600, 800, 1200],
                             width=6, state="readonly"))
        ttk.Button(tab_exp, text="Salva PNG... (Ctrl+S)",
                    command=self.on_save).pack(fill="x", pady=(12, 2))
        ttk.Button(tab_exp, text="Redraw (Ctrl+R)",
                    command=self._redraw).pack(fill="x", pady=2)
        ttk.Button(tab_exp, text="Reset defaults",
                    command=self.reset_defaults).pack(fill="x", pady=2)

    def reset_defaults(self):
        self.var_legend_loc.set("upper right")
        self.var_legend_ncol.set(1)
        self.var_legend_fs.set(11)
        self.var_style_show.set(True)
        self.var_style_loc.set("lower left")
        self.var_style_fs.set(8)
        self.var_title_fs.set(12)
        self.var_label_fs.set(12)
        self.var_tick_fs.set(10)
        self.var_font_family.set("sans-serif")
        self.var_lw_ens.set(2.0)
        self.var_lw_plan.set(1.5)
        self.var_band_alpha.set(0.20)
        self.var_show_title.set(True)
        self.var_show_grid.set(True)
        self.var_show_pdose.set(True)

    def _schedule_redraw(self):
        if self._redraw_pending: return
        self._redraw_pending = True
        self.after(80, self._redraw)

    def _redraw(self):
        self._redraw_pending = False
        try:
            self.fig.set_facecolor("white")
            # Font family
            for txt_obj in [self.ax.title, self.ax.xaxis.label, self.ax.yaxis.label]:
                pass  # set below via rcParams-ish per-call
            family = self.var_font_family.get()
            self.ax.clear()

            from matplotlib.lines import Line2D
            struct_handles, struct_labels = [], []
            for name, color in STRUCTURES:
                if name not in self.results: continue
                r = self.results[name]
                self.ax.fill_between(self.centers, r["v_lo"], r["v_hi"],
                                       color=r["color"], alpha=self.var_band_alpha.get())
                self.ax.plot(self.centers, r["v_mean"], color=r["color"],
                              lw=self.var_lw_ens.get())
                if self.var_show_pdose.get():
                    self.ax.plot(self.centers, r["v_plan"], color=r["color"],
                                  lw=self.var_lw_plan.get(), linestyle="--")
                struct_handles.append(Line2D([0], [0], color=r["color"],
                                              lw=self.var_lw_ens.get()))
                struct_labels.append(name)
            self.ax.axvline(54.0, color="gray", linestyle=":", lw=1.0)
            self.ax.set_xlabel("Dose [Gy]",
                                 fontsize=self.var_label_fs.get(), fontfamily=family)
            self.ax.set_ylabel("Volume [%]",
                                 fontsize=self.var_label_fs.get(), fontfamily=family)
            if self.var_show_title.get():
                self.ax.set_title(
                    f"Lung SBRT 54Gy/3fx — DVH overlay  "
                    f"Heuristic-FC ensemble (N={self.K}, band p2.5-p97.5) vs piano (--)",
                    fontsize=self.var_title_fs.get(), fontfamily=family,
                )
            self.ax.tick_params(axis="both", labelsize=self.var_tick_fs.get())
            for t in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                t.set_fontfamily(family)
            self.ax.grid(self.var_show_grid.get(), alpha=0.3)
            self.ax.set_xlim(0, self.var_dose_max.get())
            self.ax.set_ylim(0, 105)

            def make_legend(loc, handles, labels, fontsize, ncol):
                if loc == "none":
                    return None
                if loc == "outside right":
                    return self.ax.legend(handles, labels, loc="center left",
                                            bbox_to_anchor=(1.02, 0.5),
                                            fontsize=fontsize, ncol=ncol)
                if loc == "outside bottom":
                    return self.ax.legend(handles, labels, loc="upper center",
                                            bbox_to_anchor=(0.5, -0.12),
                                            fontsize=fontsize, ncol=ncol)
                return self.ax.legend(handles, labels, loc=loc,
                                        fontsize=fontsize, ncol=ncol)

            # STRUCTURES legend (primary)
            leg_struct = make_legend(
                self.var_legend_loc.get(), struct_handles, struct_labels,
                self.var_legend_fs.get(), self.var_legend_ncol.get(),
            )
            if leg_struct is not None:
                for txt in leg_struct.get_texts(): txt.set_fontfamily(family)
                leg_struct.set_draggable(True)
                self.ax.add_artist(leg_struct)  # keep: the second legend() call would overwrite it

            # STYLE legend (secondary): ensemble (—) vs plan (--)
            if self.var_style_show.get():
                style_handles = [Line2D([0], [0], color="black",
                                          lw=self.var_lw_ens.get(),
                                          label="ensemble mean")]
                style_labels  = ["ensemble mean"]
                if self.var_show_pdose.get():
                    style_handles.append(Line2D([0], [0], color="black",
                                                  lw=self.var_lw_plan.get(),
                                                  linestyle="--"))
                    style_labels.append("original plan")
                leg_style = make_legend(
                    self.var_style_loc.get(), style_handles, style_labels,
                    self.var_style_fs.get(), 1,
                )
                if leg_style is not None:
                    for txt in leg_style.get_texts(): txt.set_fontfamily(family)
                    leg_style.set_draggable(True)

            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            print(f"[redraw err] {e}")

    def on_save(self):
        path = filedialog.asksaveasfilename(
            title="Salva DVH overlay",
            defaultextension=".png",
            initialfile=os.path.basename(self.default_out),
            initialdir=os.path.dirname(self.default_out),
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path: return
        try:
            self.fig.savefig(path, dpi=self.var_save_dpi.get(), bbox_inches="tight")
            messagebox.showinfo("Salvato", f"Figura salvata in:\n{path}\nDPI = {self.var_save_dpi.get()}")
            print(f"[saved] {path} @ {self.var_save_dpi.get()} dpi")
        except Exception as e:
            messagebox.showerror("Errore salvataggio", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", default=f"{ROOT}/ensemble100",
                     help="Ensemble folder (default: %(default)s)")
    ap.add_argument("--out", default=f"{ROOT}/ensemble100/Fig7_DVH_overlay_N100_custom.png",
                     help="Default save path")
    ap.add_argument("--bins", type=int, default=400)
    ap.add_argument("--dose_max", type=float, default=60.0)
    ap.add_argument("--recompute", action="store_true",
                     help="Recompute the DVH ignoring the cache")
    args = ap.parse_args()

    centers, results, K = compute_or_load_cache(
        args.ens, args.bins, args.dose_max, args.recompute,
    )
    app = DVHViewer(centers, results, K, default_out=args.out)
    app.mainloop()


if __name__ == "__main__":
    main()
