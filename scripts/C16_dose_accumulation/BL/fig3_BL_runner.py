# -*- coding: utf-8 -*-
"""
fig3_BL_runner.py -- STEP 8 (Figure 3, Lung SBRT 54 Gy / 3 fx), corrected driver (2026-09-28). Replaces runner.py PASS 1-3 for
the Figure 3 case; PASS 4 (core.dose_propagation.DosePropagator) and the FC accumulation (runner.py --mode dose_accumulation)
are the submitted code, fed with outputs in the production layout. The submitted code (../) is imported, never modified.

Corrections with respect to the submitted Figure 3 (each checked by an assert below or in the tests of step 8):
  N9  mu = the Velocity DICOM field read with pydicom (components x,y,z in mm) and inverted by fixed-point iteration (pull-back
      onto the plan grid, fig3_BL_conventions_test.py); runner.py read it with reversed channels, x2 and the wrong direction
  N3  sigma_h = f * conventions_check.sigma_body_local (body-only clip quantiles, grad-norm, min-max), f = --f (2.23, user 29/9, R43/R44; was 3.86)
      theta of the l-map = median |grad I_f| over the body (runner.py: whole volume)
  N2  K on the right axes: l0 = (15,10,10) mm on (Z,Y,X) arrays with spacing (sz,sy,sx) (runner.py put 15 mm on X); tanh (R31)
  N8  folding of the backstop measured with the DVF in voxels on the body eroded by one voxel (runner.py: mm, whole volume)
  DL  sigma_DL with the corrected wiring (q22_recompute_sigmaDL.infer, variant C + S): network input [dx,dy,dz] in plan voxels,
      stdev x network-grid spacing; RAW, not calibrated (user, 28/9); zero outside the body like sigma_h
Unchanged: N = 100, FC shared noise (seed 12345, eps_k identical across fractions and arms), kernel 61, 4x4x4 blocks, alpha 0.5,
tau0 = 1e-4, adaptive threshold, gain decay 0.85, 20 retries (FC: retries reuse eps), collapse to mu; doses realigned to the online
isocentre (Fx_n/RD.*_isoalign.dcm); Fx_2 with its own DVF (guard check_fraction_registrations.py).

Arms: Heuristic_aniso (K of the manuscript), Heuristic_iso (stationary isotropic K, l = geometric mean of the per-direction lengths
that K applies over the body, as Q2.1 iso_matched; same eps), Smolders (sigma_DL raw, K of the manuscript),
Heuristic_iso_nom (2026-09-30: stationary isotropic K of equal nominal volume, l = (15*10*10)^(1/3) = 11.45 mm, as Figure 2; same eps).
Layout written (= runner.py in shape, NOT in channel order; 2026-09-29 A2): mu_dvf / sigma_dvf (3,X,Y,Z) mm with channels
[dx,dy,dz], i.e. channel c along array axis c, as core.dose_propagation.DosePropagator expects; runner.py instead saves the
channels as [dz,dy,dx] (convention of utils/dicom_vf.py). dvf_ensemble (N,3,X,Y,Z), same [dx,dy,dz] order;
uncertainty_dir (Z,Y,X); pass3_metadata.json with noise_seed (FC cache of runner.py). Internally everything is (Z,Y,X).

Usage (venv):  python fig3_BL_runner.py --arm Heuristic_aniso --gpu 0 [--fracs 1,2,3] [--n 100] [--stages sigma,sample,dose,acc]
               python fig3_BL_runner.py --arm Heuristic_aniso --smoke          (1 fraction x N = 4, self-tests, out ..._smoke)
"""
from dirunc_paths import LIB, MODELS, RESULTS_DATA, script  # data roots, see README
import os, sys, json, time, math, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB
sys.path.insert(0, HERE); sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS, "dl_arch"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, torch
from scipy import ndimage as ndi
import fig3_BL_conventions_test as T
from conventions_check import sigma_body_local, check_kernel_axes, check_body_independence
from runner import _apply_predef_rigid, _estimate_body_mask
from core.grf import adaptive_ell_map, anisotropic_correlation
from core.jacobian import jacobian_determinant
from core.sampling import DVFSampler
from core.dose_propagation import DosePropagator
from utils.scan import Scan
from utils.interpolate import resampleScan
from utils.conversions import toNumpy

ARMS = {"Heuristic_aniso": ("heuristic", "aniso"), "Heuristic_iso": ("heuristic", "iso"), "Smolders": ("smolders", "aniso"),
        "Heuristic_iso_nom": ("heuristic", "iso_nom")}   # 2026-09-30: iso of equal nominal volume, l = (15*10*10)^(1/3) = 11.45 mm (as Figure 2)
L0_ZYX = (19.5, 13.0, 13.0); ALPHA = 0.0; KS = 61     # 2026-09-29 C2: F_H now a CLI argument (--f), default 2.23 (R44)
TAU0, DECAY, RETRIES, SEED = 1e-4, 0.85, 20, 12345
INV_ITERS = 10
RD = {1: "RD_Fx1_isoalign.dcm",
      2: "RD_Fx2_isoalign.dcm",
      3: "RD_Fx3_isoalign.dcm"}
GUARD = script("check_fraction_registrations.py")
MODEL = os.path.join(rf"{MODELS}", "L20S3-4")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--fracs", default="1,2,3")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", default=rf"{RESULTS_DATA}/Fig3_BL_20260929")          # 2026-09-29 C2
ap.add_argument("--f", type=float, default=2.23)   # 2026-09-29 C2: R43/R44 = mean Elastix f' (q<=10) of Lung-DIR-QA and DIR-LAB; was 3.86
ap.add_argument("--stages", default="sigma,sample,dose,acc")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
FRACS = [int(f) for f in args.fracs.split(",")]; N = args.n; OUT = args.out; F_H = args.f
if args.smoke:
    FRACS, N, OUT = FRACS[:1], 4, args.out + "_smoke"
STAGES = set(args.stages.split(","))
METHOD, KERNEL = ARMS[args.arm]
DEV = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(DEV)
ARMDIR = os.path.join(OUT, args.arm); os.makedirs(ARMDIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {args.arm} {msg}", flush=True)


def zyx(a):          # (3,X,Y,Z) channel c along axis c  ->  (3,Z,Y,X) channels [dz,dy,dx]
    return np.ascontiguousarray(a[::-1].transpose(0, 3, 2, 1))


def xyz(a):          # inverse of zyx
    return np.ascontiguousarray(a.transpose(0, 3, 2, 1)[::-1])


def applied_lengths_mm(ell_vox, spacing, body):
    """Per-direction lengths as K applies them (4x4x4 block means with margin, weighted by body voxels of the core block).
    Copy of q21_kernel_metrics_BL.applied_lengths_mm (that script runs on import)."""
    _, Z, Y, X = ell_vox.shape
    tz, ty, tx = (int(np.ceil(s / 4)) for s in (Z, Y, X)); mg = KS // 2
    acc = np.zeros(3); wsum = 0.0
    for iz in range(4):
        for iy in range(4):
            for ix in range(4):
                z0, z1 = iz * tz, min((iz + 1) * tz, Z); y0, y1 = iy * ty, min((iy + 1) * ty, Y); x0, x1 = ix * tx, min((ix + 1) * tx, X)
                if z1 <= z0 or y1 <= y0 or x1 <= x0:
                    continue
                blk = ell_vox[:, max(0, z0 - mg):min(Z, z1 + mg), max(0, y0 - mg):min(Y, y1 + mg), max(0, x0 - mg):min(X, x1 + mg)]
                w = int(body[z0:z1, y0:y1, x0:x1].sum())
                acc += w * np.array([float(blk[d].mean()) * spacing[d] for d in range(3)]); wsum += w
    return tuple(acc / wsum)


def load_fraction(n):
    """Plan CT (v2w with inv(PreD)), sCT on the plan grid, corrected mu -- all (X,Y,Z) as runner.py PASS 1."""
    fixed = Scan(T.PLAN); _apply_predef_rigid(fixed, T.DVF[n])
    moving = Scan(os.path.join(T.DATA, f"Fx_{n}")); shape = fixed.getShape()
    if moving.getShape() != shape:
        moving = resampleScan(moving, fixed.getVoxelToWorldMatrix(), shape, defaultPixelValue=-1000.0); moving.toNumpy()
    fx = T.sq(fixed.getPixelArray()).astype(np.float32); mv = T.sq(moving.getPixelArray()).astype(np.float32)
    vox_raw, _, _ = T.raw_dvf(n, fixed)
    u = T.pullback_inverse(vox_raw, INV_ITERS, DEV).astype(np.float32)          # plan voxels, channel c along axis c
    sp = np.asarray(fixed.getVoxelSpacing(), float)                             # (sx,sy,sz)
    return fixed, fx, mv, u, sp


def stage_sigma(n, d):
    fixed, fx, mv, u, sp = load_fraction(n)
    body_xyz = np.ascontiguousarray(_estimate_body_mask(fx.transpose(2, 1, 0)).transpose(2, 1, 0))
    r = T.ratio(fx, mv, u, body_xyz, DEV)                                         # stored layout must register (N9)
    assert r <= 0.7, f"frac {n}: corrected mu does not register (ratio {r:.3f})"
    mu_xyz = (u * sp[:, None, None, None]).astype(np.float32)
    F_ = torch.from_numpy(np.ascontiguousarray(fx.transpose(2, 1, 0))).to(DEV); M_ = torch.from_numpy(np.ascontiguousarray(mv.transpose(2, 1, 0))).to(DEV)
    B_ = torch.from_numpy(body_xyz.transpose(2, 1, 0).copy()).to(DEV)
    BM_ = torch.from_numpy(_estimate_body_mask(np.ascontiguousarray(mv.transpose(2, 1, 0)))).to(DEV)
    MU_ = torch.from_numpy(zyx(mu_xyz)).to(DEV); sp_zyx = tuple(sp[::-1])
    meta = {"frac": n, "warp_ratio_mu": r, "inv_iters": INV_ITERS, "method": METHOD}
    if METHOD == "heuristic":
        s1 = sigma_body_local(F_, M_, MU_, B_, BM_, sp_zyx)                        # (Z,Y,X), 0 outside the body
        if args.smoke:
            ok, dmax = check_body_independence(lambda f, m, dv, b, bm: sigma_body_local(f, m, dv, b, bm, sp_zyx), F_, M_, MU_, B_, BM_)
            assert ok, f"sigma_body_local depends on outside-body voxels ({dmax:.2e})"; meta["body_independence_maxdiff"] = dmax
        G = (F_H * s1).cpu().numpy().astype(np.float32)
        G3 = np.repeat(G[None], 3, 0); unc = G; meta["f"] = F_H
    else:
        from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer
        from utils.vector_field import VectorField
        reg = FixedMeanPatchRegisterer(MODEL, patchSize=(256, 256, 96), voxelSpacing=(1.9531, 1.9531, 2.0))
        vtw = np.diag([*sp, 1.0]); fs = Scan(array=fx[None], vtwMatrix=vtw); ms = Scan(array=mv[None], vtwMatrix=vtw)
        inp = reg.preProcess(fs, ms, VectorField(array=u, vtwMatrix=vtw, inMM=True))
        res = reg.resampleVectorField(reg.register(inp), fs.getShape())
        s_net = (np.asarray(fs.getShape(), float) - 1) * sp / (np.asarray(inp.shape[-3:], float) - 1)
        stdev = res.stdev.detach().cpu().numpy() if torch.is_tensor(res.stdev) else np.asarray(res.stdev)
        sig_xyz = (stdev.squeeze() * s_net.reshape(3, 1, 1, 1)).astype(np.float32)     # [sx,sy,sz] mm
        G3 = zyx(sig_xyz) * body_xyz.transpose(2, 1, 0)[None]                        # (3,Z,Y,X), raw sigma_DL, 0 outside body
        unc = np.linalg.norm(G3, axis=0); meta["s_net"] = s_net.tolist(); meta["f"] = None
        del reg, inp, res; torch.cuda.empty_cache()
    b = body_xyz.transpose(2, 1, 0).astype(bool)
    meta["G_body"] = {"comp_mean_mm": [float(G3[c][b].mean()) for c in range(3)], "unc_mean_mm": float(unc[b].mean()),
                      "unc_p99_mm": float(np.percentile(unc[b], 99))}
    T_ = lambda a: np.ascontiguousarray(a)
    np.save(os.path.join(d, "mu_dvf.npy"), mu_xyz); np.save(os.path.join(d, "sigma_dvf.npy"), T_(xyz(G3)))
    np.save(os.path.join(d, "uncertainty_dir.npy"), T_(unc.astype(np.float32)))
    np.save(os.path.join(d, "body_mask.npy"), body_xyz.astype(np.uint8))
    np.save(os.path.join(d, "dvf_magnitude.npy"), T_(np.linalg.norm(mu_xyz, axis=0).transpose(2, 1, 0)))
    json.dump(meta, open(os.path.join(d, "sigma_meta.json"), "w"), indent=2)
    log(f"frac {n} sigma: warp ratio {r:.3f}, G body comp {', '.join(f'{v:.2f}' for v in meta['G_body']['comp_mean_mm'])} mm")


def stage_sample(n, d):
    fixed = Scan(T.PLAN); _apply_predef_rigid(fixed, T.DVF[n])
    sp = np.asarray(fixed.getVoxelSpacing(), float); sp_zyx = tuple(sp[::-1])
    ct = np.ascontiguousarray(T.sq(fixed.getPixelArray()).astype(np.float32).transpose(2, 1, 0))       # (Z,Y,X)
    body = np.load(os.path.join(d, "body_mask.npy")).astype(bool).transpose(2, 1, 0)
    keep = torch.from_numpy(ndi.binary_erosion(body, border_value=1)).to(DEV)
    theta = float(np.median(np.sqrt(sum(g ** 2 for g in np.gradient(ct)))[body]))                     # runner estimator, body
    ell = adaptive_ell_map(torch.from_numpy(ct)[None].to(DEV), sp_zyx, l0=L0_ZYX, alpha=ALPHA, theta=theta)
    ok, rK = check_kernel_axes(ell.cpu().numpy(), sp_zyx, cc_axis=0); assert ok, f"K axes: CC/lateral {rK:.3f} != 1.5"
    lz, ly, lx = applied_lengths_mm(ell.cpu().numpy(), sp_zyx, body); l_iso = (lz * ly * lx) ** (1 / 3)
    if KERNEL == "iso_nom":
        l_iso = float(np.prod(L0_ZYX)) ** (1 / 3)
    kern = ell if KERNEL == "aniso" else tuple(l_iso / s for s in sp_zyx)
    mu = torch.from_numpy(zyx(np.load(os.path.join(d, "mu_dvf.npy")))).to(DEV)
    G = torch.from_numpy(zyx(np.load(os.path.join(d, "sigma_dvf.npy")))).to(DEV)
    VOX = torch.tensor(sp_zyx, device=DEV, dtype=torch.float32).view(3, 1, 1, 1)
    fold = lambda z: float(((jacobian_determinant(z / VOX) <= 0) & keep).sum()) / float(keep.sum())
    base = fold(mu); thr = max(TAU0, base + 5e-4)
    Z, Y, X = mu.shape[1:]
    noise = DVFSampler.draw_shared_noise(N, (Z, Y, X), base_seed=SEED)                               # FC, as runner.py
    ens = np.lib.format.open_memmap(os.path.join(d, "dvf_ensemble.npy"), mode="w+", dtype=np.float32, shape=(N, 3, X, Y, Z))
    s2 = torch.zeros_like(mu); st = {"folding_rates": [], "retries": [], "gain": [], "first_folding_vox_body": [],
                                     "first_folding_mm_whole_production": []}
    t0 = time.time()
    for k in range(N):
        eps = noise[k].to(DEV); gain, r, acc = 1.0, 0, None
        while r < RETRIES:
            delta = G * gain * anisotropic_correlation(eps, kern, kernel_size=KS)
            rate = fold(mu + delta)
            if r == 0:
                st["first_folding_vox_body"].append(rate)
                st["first_folding_mm_whole_production"].append(float((jacobian_determinant(mu + delta) <= 0).float().mean()))
            if rate <= thr:
                acc = mu + delta; break
            gain *= DECAY; r += 1
        if acc is None:
            acc = mu; gain = 0.0; delta = torch.zeros_like(mu)
        s2 += delta ** 2
        st["folding_rates"].append(rate); st["retries"].append(r); st["gain"].append(gain)
        ens[k] = xyz(acc.cpu().numpy())
        if k % 10 == 0 or k == N - 1:
            log(f"frac {n} sample {k + 1}/{N}: retries {r} gain {gain:.3f} fold0 {st['first_folding_vox_body'][-1]:.2e} ({time.time() - t0:.0f}s)")
    ens.flush(); del ens
    b = torch.from_numpy(body).to(DEV); Grms = torch.sqrt((G ** 2).mean(0)); gpos = b & (Grms > 1e-6)
    sig_ens = torch.sqrt(s2.mean(0) / N)
    g = np.array(st["gain"])
    st.update({"mean_folding": float(np.mean(st["folding_rates"])), "max_folding": float(np.max(st["folding_rates"])),
               "mean_retries": float(np.mean(st["retries"])), "frac_r0": float(np.mean(np.array(st["retries"]) == 0)),
               "collapsed": int((g == 0).sum()), "gain_mean": float(g.mean()), "gain_rms": float(np.sqrt((g ** 2).mean())),
               "sigma_ens_over_G_body": float((sig_ens[gpos] / Grms[gpos]).mean()), "threshold": thr, "base_rate_mu_vox_body": base,
               "sec": round(time.time() - t0, 1)})
    json.dump(st, open(os.path.join(d, "folding_stats.json"), "w"), indent=2)
    json.dump({"n_samples": N, "l0_zyx": list(L0_ZYX), "alpha": ALPHA, "theta": theta, "kernel": KERNEL, "kernel_size": KS,
               "iso_length_mm": l_iso if KERNEL != "aniso" else None, "applied_lengths_zyx_mm": [lz, ly, lx], "K_axes_ratio": rK,
               "max_retries": RETRIES, "gain_decay": DECAY, "tau0": TAU0, "folding": "voxels, body eroded by 1 voxel",
               "noise_seed": SEED, "uncertainty_method": METHOD, "spacing": list(sp), "arm": args.arm,
               "driver": "fig3_BL_runner.py"}, open(os.path.join(d, "pass3_metadata.json"), "w"), indent=2)
    log(f"frac {n} sampled: theta {theta:.1f} K ratio {rK:.3f} applied (Z,Y,X) ({lz:.1f},{ly:.1f},{lx:.1f}) mm iso {l_iso:.1f}; "
        f"retries {st['mean_retries']:.2f} r0 {st['frac_r0']:.2f} coll {st['collapsed']} gain_rms {st['gain_rms']:.3f} "
        f"sig_ens/G {st['sigma_ens_over_G_body']:.3f} ({st['sec']:.0f}s)")
    if g.any() and abs(st["sigma_ens_over_G_body"] - st["gain_rms"]) > 0.15 * max(st["gain_rms"], 0.05):
        log(f"WARNING sigma_ens/G {st['sigma_ens_over_G_body']:.3f} does not track gain_rms {st['gain_rms']:.3f}")


def stage_dose(n, d):
    """PASS 4 exactly as runner.py lines 1140-1156, with the dose realigned to the online isocentre."""
    fixed = Scan(T.PLAN); _apply_predef_rigid(fixed, T.DVF[n]); spacing = fixed.getVoxelSpacing()
    dose = resampleScan(Scan(os.path.join(T.DATA, f"Fx_{n}", RD[n])), fixed.getVoxelToWorldMatrix(), fixed.getShape())
    dv = toNumpy(dose.getPixelArray()); dv = dv[0] if dv.ndim == 4 else dv
    DosePropagator(output_dir=d).propagate(np.transpose(dv, (2, 1, 0)).astype(np.float32), os.path.join(d, "dvf_ensemble.npy"), tuple(spacing))
    log(f"frac {n} PASS 4 done")


def stage_acc():
    spec = {"fixed": T.PLAN, "m_delivered": 3, "n_total": 3,
            "fractions": [{"id": f"frac_0{n}", "moving": os.path.join(T.DATA, f"Fx_{n}"), "dvf": T.DVF[n],
                           "moving_dose": os.path.join(T.DATA, f"Fx_{n}", RD[n]), "weight": 1.0} for n in FRACS]}
    js = os.path.join(ARMDIR, "fractions_BL.json"); json.dump(spec, open(js, "w"), indent=2)
    # 2026-09-29 A2: guard -- without dose_ensemble.npy in every frac_0N, runner.py --mode dose_accumulation would re-run its own
    # PASS 1-4 (production conventions) and overwrite ARMDIR; the FC accumulation must only consume the step-8 ensembles
    missing = [n for n in FRACS if not os.path.exists(os.path.join(ARMDIR, f"frac_0{n}", "dose_ensemble.npy"))]
    assert not missing, f"dose_ensemble.npy missing in frac_0{missing}: run stages sigma,sample,dose first (runner.py would redo PASS 1-4)"
    cmd = [sys.executable, os.path.join(SCRIPTS, "runner.py"), "--mode", "dose_accumulation", "--fractions", js, "--out_dir", ARMDIR,
           "--accumulation_mode", "fc", "--noise_seed", str(SEED), "--n_samples", str(N)]
    rc = subprocess.call(cmd); assert rc == 0, f"accumulation exit code {rc}"
    log("FC accumulation done")


def main():
    log(f"out {ARMDIR}, fractions {FRACS}, N {N}, stages {sorted(STAGES)}")
    rc = subprocess.call([sys.executable, GUARD] + sum([["--dvf", f"Fx_{n}={T.DVF[n]}"] for n in (1, 2, 3)], []))
    assert rc == 0, "check_fraction_registrations.py failed"
    for n in FRACS:
        d = os.path.join(ARMDIR, f"frac_0{n}"); os.makedirs(d, exist_ok=True)
        if "sigma" in STAGES and not os.path.exists(os.path.join(d, "sigma_meta.json")):
            stage_sigma(n, d)
        if "sample" in STAGES and not os.path.exists(os.path.join(d, "folding_stats.json")):
            stage_sample(n, d)
        if "dose" in STAGES and not os.path.exists(os.path.join(d, "dose_std.npy")):
            stage_dose(n, d)
        torch.cuda.empty_cache()
    if "acc" in STAGES and len(FRACS) == 3:
        stage_acc()
    log("DONE")


if __name__ == "__main__":
    main()
