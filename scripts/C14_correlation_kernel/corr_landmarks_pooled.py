# -*- coding: utf-8 -*-
"""Extended probe 1: empirical correlogram of the DIR error at the landmarks, DIR-LAB Castillo, 10 cases x 4 engines, pooled per engine.
Same method as corr_landmarks_probe.py: e_i = pred - true (per-case bias removed); r(h) = sum_pairs e_i.e_j / (n_pairs * var_case), the products
of each case normalised by the case variance and then summed over the cases (pooling); classes: all / CC (|dz| >= 2 d_xy) / in-plane (|dz| <= 0.5 d_xy).
Convergence filter per CASE as in Table 2: case RMS TRE <= q = 10 mm. Bootstrap over landmarks within case (B = 300), 68 % bands.
Positive control: raw respiratory motion. Models: ACF along Z of K and of the iso 11.45 (mean of the 10 cases on a common grid, _out_q21_BLv2),
in-plane theory exp(-h^2/4 l^2) with the mean of the applied l.
Output: _out_corr_probe/corr_pooled.{png,json}. Usage (venv): python corr_landmarks_pooled.py"""
from dirunc_paths import DATASETS  # data roots, see README
import os, csv, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "_out_corr_probe"); os.makedirs(OUT, exist_ok=True)
Q21 = os.path.join(HERE, "_out_q21_BLv2"); BASE = f"{DATASETS}/Castillo-DIR-LAB"
ENGINES = [("elastix", "Elastix"), ("convexadam", "ConvexAdam"), ("unigradicon", "uniGradICON"), ("synra", "SyNRA")]
CASES = range(1, 11); Q = 10.0; B = 300
EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 130], float); MID = 0.5 * (EDGES[1:] + EDGES[:-1]); NB = len(MID)
rng = np.random.default_rng(12345)

def load(eng, c):
    z = np.load(os.path.join(BASE, f"Results_{eng}_procrustes", f"case{c}", "landmarks.npz"))
    X = np.asarray(z["lm_fixed_world"], float); T = np.asarray(z["lm_moving_world"], float); P = np.asarray(z["lm_moving_world_pred"], float)
    assert int(np.argmax(np.asarray(z["spacing_fixed"], float))) == 2
    return X, P - T, T - X

def pair_masks(X):
    D = X[:, None, :] - X[None, :, :]; dz = np.abs(D[..., 2]); dxy = np.hypot(D[..., 0], D[..., 1]); h = np.sqrt(dz ** 2 + dxy ** 2)
    cls = {"all": np.ones_like(h, bool), "CC": dz >= 2 * dxy, "in-plane": dz <= 0.5 * dxy}
    masks = {k: [np.triu((h >= EDGES[b]) & (h < EDGES[b + 1]) & m, 1) for b in range(NB)] for k, m in cls.items()}
    return masks

def sums(V, masks, idx=None):
    """per class: (sum of normalised products per bin, pair counts per bin); idx = landmark subset (bootstrap)."""
    if idx is not None:
        V = V[idx]
    V = V - V.mean(0); G = V @ V.T; var = np.mean(np.sum(V ** 2, 1)); out = {}
    for k, ms in masks.items():
        s = np.zeros(NB); c = np.zeros(NB)
        for b, m in enumerate(ms):
            mm = m if idx is None else m[np.ix_(idx, idx)]
            c[b] = mm.sum(); s[b] = G[mm].sum() / var
        out[k] = (s, c)
    return out

res = {"q_mm": Q, "bins_mm": EDGES.tolist(), "engines": {}}
curves = {}
for eng, lbl in ENGINES:
    data = []; excluded = []
    for c in CASES:
        X, E, RAW = load(eng, c); rms = float(np.sqrt(np.mean(np.sum(E ** 2, 1))))
        if rms > Q: excluded.append((c, round(rms, 2))); continue
        data.append((c, X, E, RAW, pair_masks(X), rms))
    def pooled(which, idxs=None):
        S = {k: np.zeros(NB) for k in ("all", "CC", "in-plane")}; C = {k: np.zeros(NB) for k in S}
        for j, (c, X, E, RAW, masks, rms) in enumerate(data):
            o = sums(E if which == "err" else RAW, masks, None if idxs is None else idxs[j])
            for k in S: S[k] += o[k][0]; C[k] += o[k][1]
        return {k: (np.where(C[k] >= 20, S[k] / np.maximum(C[k], 1), np.nan), C[k]) for k in S}
    base = pooled("err"); raw = pooled("raw")
    boots = {k: [] for k in base}
    for _ in range(B):
        idxs = [np.unique(rng.integers(0, len(d[1]), len(d[1]))) for d in data]
        o = pooled("err", idxs)
        for k in boots: boots[k].append(o[k][0])
    band = {k: (np.nanpercentile(np.array(v), 16, 0), np.nanpercentile(np.array(v), 84, 0)) for k, v in boots.items()}
    def crossing(r):
        ok = np.isfinite(r); x, y = MID[ok], r[ok]
        for i in range(1, len(x)):
            if y[i - 1] >= 1 / np.e > y[i]: return float(x[i - 1] + (y[i - 1] - 1 / np.e) / (y[i - 1] - y[i]) * (x[i] - x[i - 1]))
        return None
    res["engines"][lbl] = {"cases_used": [d[0] for d in data], "excluded_q": excluded, "case_rms_mm": [round(d[5], 2) for d in data],
                           "corr": {k: {"r": np.round(base[k][0], 3).tolist(), "lo": np.round(band[k][0], 3).tolist(), "hi": np.round(band[k][1], 3).tolist(),
                                        "n_pairs": base[k][1].astype(int).tolist(), "h_1e_mm": crossing(base[k][0])} for k in base},
                           "raw_motion_r": {k: np.round(raw[k][0], 3).tolist() for k in raw}}
    curves[lbl] = (base, band, raw)
    print(f"{lbl:12s} cases {len(data)} excl {excluded}")
    for k in base: print(f"   {k:9s}", " ".join(f"{v:+.2f}" if np.isfinite(v) else "  nan" for v in base[k][0]), "| 1/e at", res["engines"][lbl]["corr"][k]["h_1e_mm"] and round(res["engines"][lbl]["corr"][k]["h_1e_mm"], 1), "mm")

# model curves: mean ACF along Z over the 10 cases on a common mm grid; applied lengths
L = list(csv.DictReader(open(os.path.join(Q21, "q21_lengths.csv"), encoding="utf-8")))
grid = np.linspace(0, EDGES[-1], 261); acfm = {}
for k in ("aniso", "iso_geo"):
    rows = []
    for c in CASES:
        spz = float([r for r in L if r["case"] == str(c)][0]["spacing_zyx"].split("/")[0]); a = np.load(os.path.join(Q21, f"acfZ_case{c}_{k}.npy"))
        rows.append(np.interp(grid, np.arange(len(a)) * spz, a, right=0.0))
    acfm[k] = np.mean(rows, 0)
lX = np.mean([float(r["applied_X_mm"]) for r in L if r["kernel"] == "aniso"]); lZ = np.mean([float(r["applied_Z_mm"]) for r in L if r["kernel"] == "aniso"]); liso = (15 * 10 * 10) ** (1 / 3)
theo = lambda l: np.exp(-grid ** 2 / (4 * l ** 2))
res["models"] = {"applied_mm": {"aniso_Z": round(lZ, 2), "aniso_X": round(lX, 2), "iso": round(liso, 2)},
                 "at_bin_mid": {"K_Z": np.round(np.interp(MID, grid, acfm["aniso"]), 3).tolist(), "iso_Z": np.round(np.interp(MID, grid, acfm["iso_geo"]), 3).tolist(),
                                "K_X_theory": np.round(np.exp(-MID ** 2 / (4 * lX ** 2)), 3).tolist(), "iso_theory": np.round(np.exp(-MID ** 2 / (4 * liso ** 2)), 3).tolist()},
                 "h_1e_mm": {"K_Z": float(grid[np.argmax(acfm["aniso"] < 1 / np.e)]), "iso_Z": float(grid[np.argmax(acfm["iso_geo"] < 1 / np.e)]), "K_X_theory": 2 * lX, "iso_theory": 2 * liso}}
json.dump(res, open(os.path.join(OUT, "corr_pooled.json"), "w"), indent=1)
print("models 1/e (mm):", res["models"]["h_1e_mm"])

ECOL = {"Elastix": "#1b1b1b", "SyNRA": "#555555", "uniGradICON": "#8c8c8c", "ConvexAdam": "#bdbdbd"}; EMK = {"Elastix": "o", "SyNRA": "s", "uniGradICON": "^", "ConvexAdam": "D"}
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
for ax, cls in zip(axs, ("CC", "in-plane", "all")):
    for lbl in ECOL:
        base, band, raw = curves[lbl]; r = base[cls][0]
        ax.fill_between(MID, band[cls][0], band[cls][1], color=ECOL[lbl], alpha=0.12, lw=0); ax.plot(MID, r, marker=EMK[lbl], color=ECOL[lbl], ms=4, lw=1.2, label=lbl)
    ax.plot(MID, curves["Elastix"][2][cls][0], ":", color="grey", lw=1, label="raw motion (control)")
    if cls == "CC":
        ax.plot(grid, acfm["aniso"], color="#8e1b12", lw=1.8, label="K aniso, ACF along Z (10 cases)"); ax.plot(grid, acfm["iso_geo"], color="#5d8aa8", lw=1.8, ls="--", label="iso 11.5 mm, ACF along Z")
    else:
        l_k = lX if cls == "in-plane" else (lX * lX * lZ) ** (1 / 3)
        ax.plot(grid, theo(l_k), color="#8e1b12", lw=1.8, label=f"K aniso, theory (l = {l_k:.1f} mm)"); ax.plot(grid, theo(liso), color="#5d8aa8", lw=1.8, ls="--", label=f"iso, theory (l = {liso:.1f} mm)")
    cnt = curves["Elastix"][0][cls][1]
    for b in range(NB): ax.annotate(str(int(cnt[b])), (MID[b], -0.28), ha="center", fontsize=6, color="grey")
    ax.axhline(0, color="grey", lw=0.6); ax.axhline(1 / np.e, color="grey", ls=":", lw=0.6); ax.set_xlim(0, EDGES[-1]); ax.set_ylim(-0.35, 1.05)
    ax.set_title(f"{cls} pairs"); ax.set_xlabel("landmark separation h (mm)"); ax.legend(fontsize=6, loc="upper right")
axs[0].set_ylabel("correlation of DIR error vectors")
fig.suptitle(f"Empirical correlogram of the landmark DIR error, DIR-LAB Castillo, cases pooled per engine (q <= {Q:.0f} mm); pair counts in grey", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "corr_pooled.png"), dpi=200); print("[out]", os.path.join(OUT, "corr_pooled.png"))
