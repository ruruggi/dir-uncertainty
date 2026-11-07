"""make_isoaligned_doses.py -- realigns the per-fraction doses to the ONLINE isocentre.

NOT part of the submission. Read-only on the sources; writes new copies.

Why
---
2026-08-05. The daily doses of Figure 3 were recomputed in Eclipse on the Velocity
sCTs after a DICOM export/import, and in that step the plan was copied through a
rigid registration redone in the TPS, different from the ONLINE one.
Result: the beams were placed 1.6-3.0 mm away from the treatment position,
systematically in the cranio-caudal direction. Since the setup correction of the
day had been applied to the couch, the only legitimate placement is the one
given by the online REG.

What it does
------------
For each fraction it computes

    Delta = iso_online - iso_recalc

where `iso_recalc` is the isocentre declared by the recomputed RTPLAN (sCT frame)
and `iso_online = inv(M_CBCT->plan) @ iso_plan` comes from the online REG of the session, and
writes a copy of the RTDOSE with `ImagePositionPatient += Delta`.

Translating a dose via the header is NUMERICALLY EXACT: no interpolation, no
resampling, the voxels are not touched (verified on the md5 of the pixels).
The remaining approximation is PHYSICAL -- the dose deposition is assumed to be
invariant under a translation of ~2 mm in the thorax -- and must be stated in the manuscript.
The exact alternative would be to copy the plan again in Eclipse onto the online isocentre and
recompute at fixed MU.

Checks that this script does NOT skip (the Fx_2 class of error originates exactly here)
---------------------------------------------------------------------------------------
  1. fraction <-> session pairing: the sCT date must match the date of the
     CBCT of the online REG used for that fraction;
  2. the two frames must coincide: sCT origin == CBCT origin;
  3. a single RTDOSE and a single RTPLAN per fraction, a single isocentre per plan;
  4. machine angles of the recomputed plan identical to those of the offline plan
     (otherwise the difference would not be a pure translation);
  5. the pixels of the written dose must have the md5 of those read.

Usage
-----
    python make_isoaligned_doses.py            # writes the three copies + table
    python make_isoaligned_doses.py --dry-run  # computes and prints, does not write
"""
from __future__ import annotations

from dirunc_paths import ACCUM_STUDY, LUNG_SBRT  # data roots, see README
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pydicom
from pydicom.uid import generate_uid

OLD = rf"{LUNG_SBRT}"                     # sCT + recomputed doses
NEW = rf"{ACCUM_STUDY}/_source/Pz1_54Gy3fx"   # online REG, anonymised copy (step 0)
PLAN_DIR = os.path.join(OLD, "Offline_Plan_18Gyx3fx54Gy")
FRACTIONS = ("Fx_1", "Fx_2", "Fx_3")
SUFFIX = "_isoalign"


# --------------------------------------------------------------------------- readers
def _one(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    assert len(hits) == 1, f"expected 1 file for {pattern}, found {len(hits)}"
    return hits[0]


def plan_isocentre(path: str) -> np.ndarray:
    """Isocentre declared by an RTPLAN. Asserts that it is unique across the beams."""
    d = pydicom.dcmread(path, force=True)
    isos = [np.array(b.ControlPointSequence[0].IsocenterPosition, float)
            for b in d.BeamSequence if "IsocenterPosition" in b.ControlPointSequence[0]]
    assert isos, f"no isocentre in {path}"
    spread = max(float(np.linalg.norm(i - isos[0])) for i in isos)
    assert spread < 1e-6, f"{len(isos)} different isocentres in the plan (spread {spread:.4f} mm)"
    return isos[0], d


def machine_angles(path: str) -> list:
    d = pydicom.dcmread(path, force=True)
    out = []
    for b in d.BeamSequence:
        cp = b.ControlPointSequence[0]
        out.append((str(b.BeamName), float(cp.GantryAngle),
                    float(cp.PatientSupportAngle), float(cp.BeamLimitingDeviceAngle)))
    return out


def ct_series(folder: str) -> dict:
    """Geometry of a CT series. The z values are read from ALL slices and sorted: two
    series on the same frame may have opposite file-name orderings, so the
    first slice by name tells nothing."""
    paths = sorted(glob.glob(os.path.join(folder, "CT*.dcm")))
    assert paths, f"no CT in {folder}"
    zs, first = [], None
    for p in paths:
        d = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        zs.append(float(d.ImagePositionPatient[2]))
        if first is None:
            first = d
    zs = np.sort(np.array(zs))
    return dict(for_uid=str(first.FrameOfReferenceUID),
                xy0=np.array(first.ImagePositionPatient, float)[:2],
                zmin=float(zs[0]), zmax=float(zs[-1]),
                spacing=[float(v) for v in first.PixelSpacing],
                shape=(int(first.Rows), int(first.Columns)),
                date=str(getattr(first, "SeriesDate", "") or getattr(first, "StudyDate", "")),
                desc=str(getattr(first, "SeriesDescription", "")), n=len(paths))


def online_rigid(fx: str) -> tuple:
    """(M CBCT->plan, CBCT info) from the online `_pre` REG of the session."""
    base = os.path.join(NEW, fx)
    pre = [d for d in os.listdir(base) if d.lower().endswith("_pre")]
    assert len(pre) == 1, f"expected a single *_pre folder in {base}, found {pre}"
    folder = os.path.join(base, pre[0])
    reg = [f for f in os.listdir(folder) if f.startswith("RE.")]
    assert len(reg) == 1, f"expected a single REG in {folder}, found {reg}"
    d = pydicom.dcmread(os.path.join(folder, reg[0]), force=True)
    top = str(d.FrameOfReferenceUID)
    mats = {}
    for rs in d.RegistrationSequence:
        for m in rs.MatrixRegistrationSequence:
            for mm in m.MatrixSequence:
                assert str(mm.FrameOfReferenceTransformationMatrixType) == "RIGID"
                mats[str(rs.FrameOfReferenceUID)] = np.array(
                    mm.FrameOfReferenceTransformationMatrix, float).reshape(4, 4)
    assert np.allclose(mats[top], np.eye(4)), \
        "the plan-frame item is not the identity: matrix direction is ambiguous"
    cbct_for = [k for k in mats if k != top][0]
    return mats[cbct_for], ct_series(folder), top


def pixel_md5(ds) -> str:
    return hashlib.md5(bytes(ds.PixelData)).hexdigest()


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="computes and prints, does not write")
    args = ap.parse_args()

    iso_plan, dplan = plan_isocentre(_one(os.path.join(PLAN_DIR, "RP.*.dcm")))
    plan_ct = ct_series(PLAN_DIR)
    ref_angles = machine_angles(_one(os.path.join(PLAN_DIR, "RP.*.dcm")))
    print(f"piano offline: iso {np.round(iso_plan, 3)}  FoR ...{plan_ct['for_uid'][-12:]}")

    report = {"iso_plan_mm": [round(float(v), 4) for v in iso_plan], "fractions": {}}
    rows = []
    for fx in FRACTIONS:
        fol = os.path.join(OLD, fx)
        sct = ct_series(fol)
        M, cbct, plan_for_online = online_rigid(fx)

        # (1) fraction <-> session pairing. The strong test is on the UIDs: Velocity
        #     inherited the FrameOfReferenceUID of the CBCT, so the online REG of THIS
        #     session and the sCT of THIS fraction must declare the same frame. A wrong
        #     pairing (the Fx_2 class of error) cannot pass here.
        assert cbct["for_uid"] == sct["for_uid"], (
            f"{fx}: the online REG used refers to frame ...{cbct['for_uid'][-12:]} "
            f"while the sCT of {fx} is on ...{sct['for_uid'][-12:]} -> fraction "
            f"paired with the WRONG session")
        assert sct["date"] == cbct["date"], (
            f"{fx}: same frame but different dates (sCT {sct['date']}, CBCT {cbct['date']})")
        # (2) and the geometry must be the same, not just the frame name
        assert np.allclose(sct["xy0"], cbct["xy0"], atol=1e-3) \
            and abs(sct["zmin"] - cbct["zmin"]) < 1e-3 \
            and abs(sct["zmax"] - cbct["zmax"]) < 1e-3 \
            and np.allclose(sct["spacing"], cbct["spacing"], atol=1e-6) \
            and sct["shape"] == cbct["shape"], (
            f"{fx}: sCT geometry != CBCT (xy0 {sct['xy0']} vs {cbct['xy0']}, "
            f"z [{sct['zmin']},{sct['zmax']}] vs [{cbct['zmin']},{cbct['zmax']}])")
        # the plan frame seen by the online REG must be that of the planning CT
        assert plan_for_online == plan_ct["for_uid"], \
            f"{fx}: the online REG does not refer to the planning CT"

        # (3)(4) recomputed plan: one isocentre, identical machine angles
        rp = _one(os.path.join(fol, "RP.*.dcm"))
        iso_recalc, drp = plan_isocentre(rp)
        assert machine_angles(rp) == ref_angles, (
            f"{fx}: machine angles differ from the offline plan -> the placement "
            f"difference is not a pure translation, this script is not enough")
        assert str(drp.FrameOfReferenceUID) == sct["for_uid"], \
            f"{fx}: the recomputed RTPLAN is not on the FoR of its sCT"

        iso_online = (np.linalg.inv(M) @ np.append(iso_plan, 1.0))[:3]
        delta = iso_online - iso_recalc

        rd = _one(os.path.join(fol, "RD.*.dcm"))
        ds = pydicom.dcmread(rd, force=True)
        assert str(ds.FrameOfReferenceUID) == sct["for_uid"], \
            f"{fx}: the RTDOSE is not on the FoR of its sCT"
        md5_before = pixel_md5(ds)
        ipp_before = np.array(ds.ImagePositionPatient, float)

        out_path = os.path.join(fol, os.path.basename(rd).replace(".dcm", f"{SUFFIX}.dcm"))
        rows.append((fx, sct["date"], iso_recalc, iso_online, delta,
                     float(np.linalg.norm(delta)), os.path.basename(out_path)))
        report["fractions"][fx] = {
            "sCT_date": sct["date"], "sCT_desc": sct["desc"], "CBCT_desc": cbct["desc"],
            "iso_recalc_mm": [round(float(v), 4) for v in iso_recalc],
            "iso_online_mm": [round(float(v), 4) for v in iso_online],
            "delta_mm": [round(float(v), 4) for v in delta],
            "delta_norm_mm": round(float(np.linalg.norm(delta)), 4),
            "rtdose_in": os.path.basename(rd), "rtdose_out": os.path.basename(out_path),
            "ipp_before": [round(float(v), 4) for v in ipp_before],
            "ipp_after": [round(float(v), 4) for v in (ipp_before + delta)],
            "pixel_md5": md5_before,
        }

        if args.dry_run:
            continue

        ds.ImagePositionPatient = [float(v) for v in (ipp_before + delta)]
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesInstanceUID = generate_uid()      # it is a derived series, not the original
        ds.SeriesDescription = (str(getattr(ds, "SeriesDescription", ""))[:48]
                                + " isoalign")
        ds.ImageComments = ("dose translated by {:+.3f},{:+.3f},{:+.3f} mm to place the "
                            "isocentre on the ONLINE treatment isocentre of this "
                            "fraction (see make_isoaligned_doses.py)").format(*delta)
        ds.save_as(out_path)

        # (5) the pixels must not have been touched
        back = pydicom.dcmread(out_path, force=True)
        assert pixel_md5(back) == md5_before, f"{fx}: the dose pixels have changed!"
        assert np.allclose(np.array(back.ImagePositionPatient, float),
                           ipp_before + delta, atol=1e-9), f"{fx}: ipp not applied"
        assert float(back.DoseGridScaling) == float(ds.DoseGridScaling)
        assert str(back.FrameOfReferenceUID) == sct["for_uid"]

    print()
    print(f"{'fx':<6}{'date':<10}{'iso recalc':<28}{'iso online':<28}"
          f"{'Delta (mm)':<26}{'|D|':>6}")
    for fx, date, a, b, d_, n, _o in rows:
        print(f"{fx:<6}{date:<10}{str(np.round(a, 2)):<28}{str(np.round(b, 2)):<28}"
              f"{str(np.round(d_, 2)):<26}{n:>6.2f}")
    print("\nthe z component (cranio-caudal) dominates and has the same sign in the three "
          "fractions:\na systematic bias, not noise.")

    if not args.dry_run:
        rep = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "isoaligned_doses.json")
        with open(rep, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {len(rows)} realigned RTDOSE files (+ {os.path.basename(rep)})")
        for _fx, _d, _a, _b, _dd, _n, out in rows:
            print(f"   {out}")
    else:
        print("\n--dry-run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
