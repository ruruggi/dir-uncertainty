"""check_fraction_registrations.py -- READ-ONLY guard, NOT part of the submission.

Checks that every fraction uses its OWN deformable registration.

Origin
------
2026-07-31. In the Lung SBRT 54Gy/3fx case (manuscript Figure 3) the Fx_2 folder
contained a bit-for-bit duplicate of the Fx_3 registration (same md5, same
SOPInstanceUID, same deformable grid), exported under a different file name.
The Fx_2 dose was therefore deformed onto the planning CT with the Fx_3 field.
The defect was invisible downstream: no pipeline step compares the
FrameOfReferenceUID of the registration with that of the CT it claims to register.

Run this guard BEFORE any accumulation, on any multi-fraction case.
It takes seconds; the recomputation it protects against takes hours.

Two modes, same invariants (`audit`), different objects:

    --base   DICOM registrations from a commercial engine (the Velocity / Figure 3 case);
    --accum  the in-house sub-project, where the "registration" of a fraction is the
             fields inside `step4.npz` (phi^-1 and the readers' psi) and the dose is the
             RTDOSE recomputed in Eclipse on the sCT of THAT fraction. The pipeline
             steps call it from code with `guard_accum_fractions(root, FRACTIONS)`.

Usage
-----
    python check_fraction_registrations.py                       # Lung case, default
    python check_fraction_registrations.py --base <dir> --plan <sub> --fx Fx_1 Fx_2 Fx_3
    python check_fraction_registrations.py --dvf Fx_2=<path>      # explicit override
    python check_fraction_registrations.py --accum <root>         # sub-project gate
    python check_fraction_registrations.py --selftest             # logic test

Exit code 0 = all fractions sound, 1 = at least one problem.
"""
from __future__ import annotations

from dirunc_paths import ACCUM_STUDY, LUNG_SBRT  # data roots, see README
import argparse
import glob
import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Pure logic (no DICOM I/O): testable with --selftest
# ---------------------------------------------------------------------------
@dataclass
class RegRecord:
    """Metadata extracted from a registration + the CT it should register."""
    fx: str                         # fraction label, e.g. "Fx_2"
    ct_for: str = ""                # FrameOfReferenceUID of the fraction CT ("" = n/a)
    reg_path: str = ""
    reg_for: str = ""               # FrameOfReferenceUID declared by the registration
    source_for: str = ""            # SourceFrameOfReferenceUID of the pre-deformation item
    sop_uid: str = ""
    md5: str = ""
    grid_md5: str = ""
    missing: bool = False
    nan_frac: float = 0.0           # fraction of non-finite vectors in the grid
    n_vectors: int = 0


def audit(records: List[RegRecord], plan_for: str,
          what: str = "la registrazione") -> Dict[str, List[str]]:
    """Returns {fx: [problems]}. A fraction absent from the dict is sound.

    Four invariants, derived from the sound fractions observed (Fx_1, Fx_3):
      1. the registration declares the FrameOfReferenceUID of the fraction CT;
      2. the pre-deformation item has the planning CT as its source;
      3. no SOPInstanceUID is shared between fractions;
      4. no file (md5) nor deformable grid is shared between fractions.

    `what` is only the object label used in the messages: the same logic runs on
    DICOM registrations, on RTDOSE files and on the sub-project fields.
    """
    problems: Dict[str, List[str]] = {r.fx: [] for r in records}

    by_sop: Dict[str, List[str]] = {}
    by_md5: Dict[str, List[str]] = {}
    by_grid: Dict[str, List[str]] = {}
    by_for: Dict[str, str] = {r.ct_for: r.fx for r in records if r.ct_for}

    for r in records:
        if r.missing:
            problems[r.fx].append(f"{what} non c'e'")
            continue

        if r.reg_for and r.ct_for and r.reg_for != r.ct_for:
            owner = by_for.get(r.reg_for)
            whose = f" (e' la CT di {owner})" if owner and owner != r.fx else ""
            problems[r.fx].append(
                f"{what} dichiara FrameOfReferenceUID {r.reg_for}{whose}, "
                f"non quello della CT di {r.fx} ({r.ct_for})")

        if plan_for and r.source_for and r.source_for != plan_for:
            problems[r.fx].append(
                f"SourceFrameOfReferenceUID {r.source_for} != planning CT ({plan_for})")

        # A file with the right geometry and size may still contain no valid
        # vectors: ConvexAdam_computed_dvf.dcm was 100% NaN over 70,778,880 float32.
        if r.nan_frac > 0.001:
            problems[r.fx].append(
                f"griglia deformabile con {100*r.nan_frac:.1f}% di valori non finiti "
                f"su {r.n_vectors} ({'inutilizzabile' if r.nan_frac > 0.999 else 'parzialmente corrotta'})")

        if r.sop_uid:
            by_sop.setdefault(r.sop_uid, []).append(r.fx)
        if r.md5:
            by_md5.setdefault(r.md5, []).append(r.fx)
        if r.grid_md5:
            by_grid.setdefault(r.grid_md5, []).append(r.fx)

    for label, table in (("SOPInstanceUID", by_sop), ("file (md5)", by_md5),
                         ("griglia deformabile", by_grid)):
        for key, fxs in table.items():
            if len(fxs) > 1:
                for fx in fxs:
                    others = [f for f in fxs if f != fx]
                    problems[fx].append(
                        f"{label} condiviso con {', '.join(others)} ({key[:16]}...): "
                        f"e' lo stesso oggetto, non due oggetti distinti")

    return {fx: p for fx, p in problems.items() if p}


# ---------------------------------------------------------------------------
# DICOM extraction
# ---------------------------------------------------------------------------
def ct_frame_of_reference(folder: str) -> str:
    import pydicom
    # Naming schemes encountered: "CT.<uid>.dcm" (lung), "CT<uid>.dcm" (prostate),
    # "CT.0001.dcm" (the sub-project sCTs).
    paths = sorted(glob.glob(os.path.join(folder, "CT*.dcm")))
    if not paths:
        return ""
    return str(pydicom.dcmread(paths[0], stop_before_pixels=True).FrameOfReferenceUID)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_reg(path: str) -> dict:
    import pydicom
    r = pydicom.dcmread(path)
    out = {
        "reg_for": str(getattr(r, "FrameOfReferenceUID", "")),
        "sop_uid": str(getattr(r, "SOPInstanceUID", "")),
        "source_for": "",
        "grid_md5": "",
    }
    out["md5"] = file_md5(path)

    for it in getattr(r, "DeformableRegistrationSequence", []):
        if getattr(it, "PreDeformationMatrixRegistrationSequence", None) is None:
            continue
        out["source_for"] = str(getattr(it, "SourceFrameOfReferenceUID", ""))
        grid = getattr(it, "DeformableRegistrationGridSequence", None)
        if grid is not None and "VectorGridData" in grid[0]:
            raw = bytes(grid[0].VectorGridData)
            out["grid_md5"] = hashlib.md5(raw).hexdigest()
            import numpy as _np
            v = _np.frombuffer(raw, dtype=_np.float32)
            out["n_vectors"] = int(v.size)
            out["nan_frac"] = float((~_np.isfinite(v)).mean()) if v.size else 1.0
        break
    return out


def collect(base: str, plan: str, fx_names: List[str],
            dvf_overrides: Optional[Dict[str, str]] = None,
            ct_overrides: Optional[Dict[str, str]] = None) -> tuple:
    dvf_overrides = dvf_overrides or {}
    # ct_overrides: needed for structures that do not follow the layout of the old study
    # (`<base>/Fx_n` with the CTs at the first level). E.g. row (e) of Figure 3, where the
    # moving image is `_accum_study/<pz>/Fx_n/sCT90_dicom`. Without this the guard would compare
    # our REG with the Velocity SynCT and would reject correct files.
    ct_overrides = ct_overrides or {}
    plan_for = ct_frame_of_reference(os.path.join(base, plan))

    records = []
    for fx in fx_names:
        folder = ct_overrides.get(fx) or os.path.join(base, fx)
        rec = RegRecord(fx=fx, ct_for=ct_frame_of_reference(folder))

        path = dvf_overrides.get(fx)
        if path is None:
            cands = sorted(glob.glob(os.path.join(folder, "reg_name*.dcm")))
            if len(cands) > 1:
                print(f"  [WARN] {fx}: {len(cands)} reg_name*.dcm files present; "
                      f"using {os.path.basename(cands[0])}. Pass --dvf {fx}=<path> "
                      f"to be explicit.")
            path = cands[0] if cands else None

        if not path or not os.path.exists(path):
            rec.missing = True
        else:
            rec.reg_path = path
            for k, v in _read_reg(path).items():
                setattr(rec, k, v)
        records.append(rec)

    return records, plan_for


# ---------------------------------------------------------------------------
# Gate for the in-house sub-project: same invariants, our own objects
# ---------------------------------------------------------------------------
ACCUM_FIELDS = ("phi_inv", "psi_elastix", "psi_unigradicon")


def array_md5(a) -> str:
    """md5 of the bytes of an array: two fractions with the same field have the same digest."""
    import numpy as np
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def guard_accum_fractions(root: str, fractions, fields=ACCUM_FIELDS,
                          plan_dir: str = "", verbose: bool = True) -> Dict[str, List[str]]:
    """PRE-ACCUMULATION gate of the sub-project. Call it and forget it: if something is
    wrong it prints why and raises SystemExit(1), so the accumulation does not start.

    root       patient folder, e.g. <ACCUM_STUDY>\\Pz1_54Gy3fx
    fractions  {"Fx_1": "rePlan1", ...} to ALSO check the Eclipse RTDOSE files,
               or ("Fx_1", "Fx_2", ...) to check only the fields.
    fields      keys of step4.npz to compare across fractions.

    What it rejects, i.e. the two defects that actually occurred (2026-07-31) translated
    into the objects of this pipeline:
      - a fraction that uses the field or dose of another one (shared digest /
        SOPInstanceUID, or an RTDOSE that declares the FrameOfReferenceUID of another
        fraction's sCT);
      - a field that is present but unusable because it is non-finite (it was ConvexAdam, 100 % NaN).

    It costs ~30 s (re-reads the fields from step4.npz and digests them): negligible next to
    the runtime of each step, and the only way to be independent of the caller.
    """
    import numpy as np

    fx_names = list(fractions)
    subdirs = dict(fractions) if isinstance(fractions, dict) else {}
    plan_for = ct_frame_of_reference(plan_dir) if plan_dir else ""
    problems: Dict[str, List[str]] = {}

    if verbose:
        print(f"[GUARD] {len(fx_names)} fractions in {root}")

    # 1) The doses re-exported from Eclipse: each must declare the FoR of its OWN sCT
    #    (the sCTs have a new, distinct FrameOfReferenceUID per fraction), and the three
    #    must be three genuinely distinct files.
    if subdirs:
        import pydicom
        recs = []
        for fx in fx_names:
            rec = RegRecord(fx=fx, ct_for=ct_frame_of_reference(
                os.path.join(root, fx, "sCT_dicom")))
            cands = sorted(glob.glob(os.path.join(root, fx, subdirs[fx], "RD.*.dcm")))
            if not cands:
                rec.missing = True
            else:
                d = pydicom.dcmread(cands[0], stop_before_pixels=True)
                rec.reg_path = cands[0]
                rec.reg_for = str(getattr(d, "FrameOfReferenceUID", ""))
                rec.sop_uid = str(getattr(d, "SOPInstanceUID", ""))
                rec.md5 = file_md5(cands[0])
            recs.append(rec)
        for fx, p in audit(recs, plan_for, what="la dose").items():
            problems[f"{fx} [dose]"] = p

    # 2) The fields, one pass per key: two fractions cannot share the same
    #    field, and a non-finite field is unusable.
    for key in fields:
        recs = []
        for fx in fx_names:
            npz = os.path.join(root, fx, "step4.npz")
            rec = RegRecord(fx=fx, reg_path=f"{key} in step4.npz")
            if not os.path.exists(npz):
                rec.missing = True
            else:
                with np.load(npz) as z:
                    if key not in z.files:
                        rec.missing = True
                    else:
                        a = z[key]
                        rec.grid_md5 = array_md5(a)
                        rec.n_vectors = int(a.size)
                        rec.nan_frac = float(1.0 - np.isfinite(a).mean())
            recs.append(rec)
        for fx, p in audit(recs, "", what=f"il campo {key}").items():
            problems[f"{fx} [{key}]"] = p

    if problems:
        print(f"\n[GUARD] FAIL — {len(problems)} objects with problems:")
        for label, msgs in problems.items():
            print(f"  {label}:")
            for m in msgs:
                print(f"    - {m}")
        print("\nDO NOT accumulate: the dose would be deformed with the wrong field.")
        raise SystemExit(1)

    if verbose:
        print(f"[GUARD] ok: dose and fields ({', '.join(fields)}) distinct and finite, "
              f"each belonging to its own fraction.")
    return problems


# ---------------------------------------------------------------------------
def _selftest() -> None:
    """The audit logic recognises the sound case and the duplicate that actually occurred."""
    PLAN = "FOR.PLAN"
    good = [
        RegRecord("Fx_1", "FOR.1", "a.dcm", "FOR.1", PLAN, "SOP.1", "md5a", "g1"),
        RegRecord("Fx_2", "FOR.2", "b.dcm", "FOR.2", PLAN, "SOP.2", "md5b", "g2"),
        RegRecord("Fx_3", "FOR.3", "c.dcm", "FOR.3", PLAN, "SOP.3", "md5c", "g3"),
    ]
    assert audit(good, PLAN) == {}, audit(good, PLAN)

    # The real bug: Fx_2 contains the Fx_3 file.
    bug = [
        RegRecord("Fx_1", "FOR.1", "a.dcm", "FOR.1", PLAN, "SOP.1", "md5a", "g1"),
        RegRecord("Fx_2", "FOR.2", "b.dcm", "FOR.3", PLAN, "SOP.3", "md5c", "g3"),
        RegRecord("Fx_3", "FOR.3", "c.dcm", "FOR.3", PLAN, "SOP.3", "md5c", "g3"),
    ]
    got = audit(bug, PLAN)
    assert set(got) == {"Fx_2", "Fx_3"}, got
    assert any("non quello della CT di Fx_2" in m for m in got["Fx_2"]), got["Fx_2"]
    assert any("SOPInstanceUID condiviso" in m for m in got["Fx_2"]), got["Fx_2"]
    assert any("file (md5) condiviso" in m for m in got["Fx_2"]), got["Fx_2"]
    assert "Fx_1" not in got, got

    # Missing registration, and non-planning source.
    other = [
        RegRecord("Fx_1", "FOR.1", missing=True),
        RegRecord("Fx_2", "FOR.2", "b.dcm", "FOR.2", "FOR.X", "SOP.2", "md5b", "g2"),
    ]
    got = audit(other, PLAN)
    assert got["Fx_1"] == ["la registrazione non c'e'"], got
    assert any("!= planning CT" in m for m in got["Fx_2"]), got

    # Syntactically valid grid but without vectors (the ConvexAdam case).
    empty = [RegRecord("Fx_1", "FOR.1", "a.dcm", "FOR.1", PLAN, "SOP.1", "md5a", "g1",
                       nan_frac=1.0, n_vectors=70778880)]
    got = audit(empty, PLAN)
    assert any("non finiti" in m and "inutilizzabile" in m for m in got["Fx_1"]), got
    ok = [RegRecord("Fx_1", "FOR.1", "a.dcm", "FOR.1", PLAN, "SOP.1", "md5a", "g1",
                    nan_frac=0.0, n_vectors=100)]
    assert audit(ok, PLAN) == {}, audit(ok, PLAN)

    # Shape of the sub-project records (guard_accum_fractions, field pass):
    # no FoR, no SOP, only the field digest -> the duplicate is still detected.
    import numpy as np
    a, b = np.zeros((3, 4, 4, 4), np.float32), np.ones((3, 4, 4, 4), np.float32)
    dup = [RegRecord(fx, grid_md5=array_md5(arr), n_vectors=arr.size)
           for fx, arr in (("Fx_1", a), ("Fx_2", b), ("Fx_3", b))]
    got = audit(dup, "", what="il campo psi_elastix")
    assert set(got) == {"Fx_2", "Fx_3"}, got
    assert any("griglia deformabile condiviso con Fx_3" in m for m in got["Fx_2"]), got
    nan = np.full((3, 4, 4, 4), np.nan, np.float32)
    got = audit([RegRecord("Fx_1", grid_md5=array_md5(nan), n_vectors=nan.size,
                           nan_frac=float(1.0 - np.isfinite(nan).mean()))], "")
    assert any("inutilizzabile" in m for m in got["Fx_1"]), got

    print("[OK] selftest audit()")


def main() -> int:
    BASE = rf"{LUNG_SBRT}"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--plan", default="Offline_Plan_18Gyx3fx54Gy")
    ap.add_argument("--fx", nargs="+", default=["Fx_1", "Fx_2", "Fx_3"])
    ap.add_argument("--dvf", action="append", default=[], metavar="FX=PATH",
                    help="Override the registration path, e.g. --dvf Fx_2=C:/new.dcm")
    ap.add_argument("--ct", action="append", default=[], metavar="FX=DIR",
                    help="Override the CT folder of the fraction (the moving image), for structures "
                         "that do not follow the <base>/Fx_n layout. E.g. --ct "
                         f"Fx_1={ACCUM_STUDY}/Pz1_54Gy3fx/Fx_1/sCT90_dicom")
    ap.add_argument("--accum", metavar="ROOT", help="sub-project gate: audits the fields "
                    "of step4.npz and the re-exported RTDOSE files under ROOT (patient folder)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return 0

    if args.accum:
        # Same fractions/subfolders as accum_step5_accumulate.FRACTIONS. Intentional
        # duplicate: the guard must be able to run standalone, without importing the pipeline.
        guard_accum_fractions(args.accum, {"Fx_1": "rePlan1", "Fx_2": "rePlan2",
                                           "Fx_3": "rePlan3"})
        return 0

    def _pairs(specs, flag):
        out = {}
        for spec in specs:
            if "=" not in spec:
                ap.error(f"{flag} expects FX=PATH, got '{spec}'")
            fx, path = spec.split("=", 1)
            out[fx.strip()] = path.strip()
        return out

    overrides = _pairs(args.dvf, "--dvf")
    ct_overrides = _pairs(args.ct, "--ct")

    print(f"Base : {args.base}")
    print(f"Plan : {args.plan}")
    for fx, p in ct_overrides.items():
        print(f"CT   : {fx} -> {p}")
    records, plan_for = collect(args.base, args.plan, args.fx, overrides, ct_overrides)
    print(f"Planning CT FrameOfReferenceUID: {plan_for or '(not found)'}\n")

    for r in records:
        print(f"{r.fx}")
        print(f"  CT  FoR   : {r.ct_for or '-'}")
        if r.missing:
            print("  reg       : MISSING")
            continue
        print(f"  reg file  : {os.path.basename(r.reg_path)}  md5={r.md5[:16]}")
        print(f"  reg FoR   : {r.reg_for or '-'}")
        print(f"  sourceFoR : {r.source_for or '-'}")
        print(f"  grid md5  : {r.grid_md5[:16] or '-'}")

    problems = audit(records, plan_for)
    print("\n" + "=" * 70)
    if not problems:
        print(f"[OK] {len(records)} fractions: every registration belongs to its own CT.")
        return 0
    print(f"[FAIL] {len(problems)} fractions with problems:")
    for fx, msgs in problems.items():
        print(f"\n  {fx}:")
        for m in msgs:
            print(f"    - {m}")
    print("\nDO NOT accumulate: the dose would be deformed with the wrong field.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
