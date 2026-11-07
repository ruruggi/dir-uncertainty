"""
fig_dvh_overlay.py
==================
DVH overlay figure (800 dpi): original plan (dashed) vs Heuristic-FC
ensemble (solid line + p2.5-p97.5 band) for 6 structures:
  PTV54, Lung_R_out, Spinal Cord, Esophagus, Bronchus, Chestwall_R.

Usage:
    python fig_dvh_overlay.py [--ens DIR] [--out OUT] [--dpi 800] [--bins N]
Default: ensemble100/, output Fig7_DVH_overlay_N100.png.
"""
from __future__ import annotations
from dirunc_paths import LUNG_SBRT  # data roots, see README
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage.draw import polygon as draw_polygon
from utils.scan import Scan

ROOT     = rf"{LUNG_SBRT}"
PCT_DIR  = f"{ROOT}/Offline_Plan_18Gyx3fx54Gy"
DVF_FX1  = sorted(glob.glob(f"{ROOT}/Fx_1/reg_*.dcm"))[0]
RS_PATH  = sorted(glob.glob(f"{PCT_DIR}/RS*.dcm"))[0]
RD_PLAN  = sorted(glob.glob(f"{PCT_DIR}/RD*.dcm"))[0]

# Structures (names exactly as in the RTSTRUCT) + colour
STRUCTURES = [
    ("PTV54",        "#e41a1c"),  # red
    ("Lung_R_out",   "#377eb8"),  # blue
    ("Spinal Cord",  "#4daf4a"),  # green
    ("Esophagus",    "#ff7f00"),  # orange
    ("Bronchus",     "#984ea3"),  # purple
    ("Chestwall_R",  "#a65628"),  # brown
]


def load_pct_post_rigid():
    pct = Scan(PCT_DIR)
    dcm = pydicom.dcmread(DVF_FX1, stop_before_pixels=True)
    predef = None
    for reg in getattr(dcm, "DeformableRegistrationSequence", []):
        if hasattr(reg, "PreDeformationMatrixRegistrationSequence"):
            predef = np.array(
                reg.PreDeformationMatrixRegistrationSequence[0]
                    .FrameOfReferenceTransformationMatrix
            ).reshape(4, 4)
            break
    inv_predef = np.linalg.inv(predef)
    v2w_post = inv_predef @ pct.getVoxelToWorldMatrix()
    pct._setVoxelToWorldMatrix(v2w_post)
    ct = pct.getPixelArray()
    if hasattr(ct, "cpu"): ct = ct.cpu().numpy()
    ct = np.asarray(ct).squeeze()
    if ct.shape == (512, 512, 90):
        ct = np.transpose(ct, (2, 1, 0))
    return ct, v2w_post, inv_predef


def rasterize_roi(rs, roi_name, ny, nx, nz, inv_predef, v2w_post):
    rno = None
    for s in rs.StructureSetROISequence:
        if s.ROIName == roi_name:
            rno = s.ROINumber; break
    if rno is None:
        return None
    rc = next((r for r in rs.ROIContourSequence
                if r.ReferencedROINumber == rno), None)
    if rc is None or not hasattr(rc, "ContourSequence"):
        return None
    w2v = np.linalg.inv(v2w_post)
    mask = np.zeros((nz, ny, nx), dtype=bool)
    for c in rc.ContourSequence:
        if c.ContourGeometricType != "CLOSED_PLANAR": continue
        pts = np.array(c.ContourData, dtype=np.float64).reshape(-1, 3)
        homog = np.column_stack([pts, np.ones(len(pts))])
        pts_post = (inv_predef @ homog.T).T[:, :3]
        homog_post = np.column_stack([pts_post, np.ones(len(pts_post))])
        vox = (w2v @ homog_post.T).T[:, :3]
        z = int(round(vox[:, 2].mean()))
        if not 0 <= z < nz: continue
        rr, cc = draw_polygon(vox[:, 1], vox[:, 0], shape=(ny, nx))
        mask[z, rr, cc] = True
    return mask


def resample_plan_dose(rd_path, ct_zyx, v2w_post, inv_predef):
    import scipy.ndimage as ndi
    rd = pydicom.dcmread(rd_path)
    dose_arr = rd.pixel_array.astype(np.float32) * float(rd.DoseGridScaling)
    sx = float(rd.PixelSpacing[1])
    sy = float(rd.PixelSpacing[0])
    sz = float(abs(rd.GridFrameOffsetVector[1] - rd.GridFrameOffsetVector[0])) \
         if len(rd.GridFrameOffsetVector) > 1 else 2.0
    ox, oy, oz = [float(v) for v in rd.ImagePositionPatient]
    v2w_dose = np.array([[sx,0,0,ox],[0,sy,0,oy],[0,0,sz,oz],[0,0,0,1]],
                         dtype=np.float64)
    v2w_dose_post = inv_predef @ v2w_dose
    nz, ny, nx = ct_zyx.shape
    Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    homog = np.stack([X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.size)], axis=0)
    world = v2w_post @ homog
    dose_vox = np.linalg.inv(v2w_dose_post) @ world
    coords = np.stack([dose_vox[2], dose_vox[1], dose_vox[0]], axis=0)
    out = ndi.map_coordinates(dose_arr, coords, order=1, mode="constant", cval=0.0)
    return out.reshape(nz, ny, nx).astype(np.float32)


def dvh_cumulative(dose, mask, edges):
    """Cumulative DVH: V(D) = % of mask volume receiving ≥ D."""
    vals = dose[mask]
    hist, _ = np.histogram(vals, bins=edges)
    return 100.0 * np.cumsum(hist[::-1])[::-1] / max(vals.size, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", default=f"{ROOT}/ensemble100")
    ap.add_argument("--out", default=f"{ROOT}/ensemble100/Fig7_DVH_overlay.png")
    ap.add_argument("--dpi", type=int, default=800)
    ap.add_argument("--bins", type=int, default=400)
    ap.add_argument("--dose_max", type=float, default=60.0)
    args = ap.parse_args()

    print(f"Loading pCT + structures...")
    ct, v2w_post, inv_predef = load_pct_post_rigid()
    nz, ny, nx = ct.shape
    rs = pydicom.dcmread(RS_PATH)
    masks = {}
    for name, _ in STRUCTURES:
        m = rasterize_roi(rs, name, ny, nx, nz, inv_predef, v2w_post)
        if m is None:
            print(f"  [warn] {name} not found in RTSTRUCT — skipped")
            continue
        n = int(m.sum())
        if n == 0:
            print(f"  [warn] {name} empty after rasterization — skipped")
            continue
        masks[name] = m
        print(f"  {name:>15s}: {n} voxels")

    print(f"\nLoading ensemble + plan dose from {args.ens}...")
    dose_mean = np.load(f"{args.ens}/dose_accumulated_fc.npy").astype(np.float32)
    dose_per  = np.load(f"{args.ens}/dose_acc_per_sample_fc.npy", mmap_mode="r")
    if dose_mean.shape != (nz, ny, nx) and dose_mean.shape == (nx, ny, nz):
        dose_mean = np.transpose(dose_mean, (2, 1, 0))
    K = dose_per.shape[0]
    print(f"  K (ensemble size) = {K}")
    print(f"  dose_mean range [{dose_mean.min():.2f}, {dose_mean.max():.2f}] Gy")

    dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)
    print(f"  dose_plan range [{dose_plan.min():.2f}, {dose_plan.max():.2f}] Gy")

    # DVH common grid
    edges = np.linspace(0, args.dose_max, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    print(f"\nComputing DVHs (bins={args.bins}, dose_max={args.dose_max} Gy)...")
    results = {}
    for name, color in STRUCTURES:
        if name not in masks:
            continue
        m = masks[name]
        v_plan = dvh_cumulative(dose_plan, m, edges)
        V_per = np.empty((K, len(centers)), dtype=np.float32)
        for k in range(K):
            V_per[k] = dvh_cumulative(np.asarray(dose_per[k]), m, edges)
        v_mean = V_per.mean(axis=0)
        v_lo   = np.percentile(V_per, 2.5,  axis=0)
        v_hi   = np.percentile(V_per, 97.5, axis=0)
        # Key metrics for printout
        d_mean = float((dose_mean[m]).mean())
        d_plan_mean = float((dose_plan[m]).mean())
        print(f"  {name:>15s}  μ_ens={d_mean:5.2f} Gy   μ_plan={d_plan_mean:5.2f} Gy   "
              f"Δ%={100*(d_mean-d_plan_mean)/max(d_plan_mean,1e-3):+5.2f}")
        results[name] = dict(color=color, v_plan=v_plan,
                              v_mean=v_mean, v_lo=v_lo, v_hi=v_hi)

    print(f"\nRendering DVH overlay at {args.dpi} dpi...")
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for name, color in STRUCTURES:
        if name not in results: continue
        r = results[name]
        ax.fill_between(centers, r["v_lo"], r["v_hi"],
                         color=r["color"], alpha=0.20)
        ax.plot(centers, r["v_mean"], color=r["color"], lw=2.0,
                 label=f"{name} (ens)")
        ax.plot(centers, r["v_plan"], color=r["color"], lw=1.5,
                 linestyle="--", label=f"{name} (plan)")

    ax.axvline(54.0, color="gray", linestyle=":", lw=1.0)
    ax.set_xlabel("Dose [Gy]", fontsize=12)
    ax.set_ylabel("Volume [%]", fontsize=12)
    ax.set_title(
        f"Lung SBRT 54Gy/3fx — DVH overlay  "
        f"Heuristic-FC ensemble (N={K}, band p2.5-p97.5) vs piano originale (--)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.dose_max)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] saved {args.out}")


if __name__ == "__main__":
    main()
