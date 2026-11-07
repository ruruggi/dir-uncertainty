"""Data and code roots used by the analysis scripts.

Every root can be overridden with an environment variable; the defaults
assume the layout described in README.md (``data/``, ``models/`` and
``results/`` next to ``lib/`` and ``scripts/``).
Run the analysis scripts from the repository root with ``PYTHONPATH=lib``.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = str(REPO / "lib").replace("\\", "/")


def _root(var, default):
    return os.environ.get(var, str(default)).replace("\\", "/").rstrip("/")


# Public landmark datasets (Lung-, Abdominal-, Liver-DIR-QA, DIR-LAB) and the
# per-engine registration outputs written next to them.
DATASETS = _root("DIRUNC_DATASETS", REPO / "data" / "Datasets")
# Institutional lung SBRT case of the dose-accumulation example (not redistributed).
LUNG_SBRT = _root("DIRUNC_LUNG_SBRT", REPO / "data" / "Lung_SBRT_54Gy3fx")
ACCUM_STUDY = _root("DIRUNC_ACCUM_STUDY", REPO / "data" / "accum_study")
# Large outputs of the dose-accumulation ensembles.
RESULTS_DATA = _root("DIRUNC_RESULTS_DATA", REPO / "data" / "results_large")
# Weights of sigma_DL (Smolders et al. 2023), not redistributed.
MODELS = _root("DIRUNC_MODELS", REPO / "models")
# Outputs of the analysis scripts.
RESULTS = _root("DIRUNC_RESULTS", REPO / "results")

# The analysis scripts import one another by module name: every script
# folder goes on the path, after lib/ so that no library module is shadowed.
for _d in sorted({str(p.parent) for p in (REPO / "scripts").rglob("*.py")}):
    if _d not in sys.path:
        sys.path.append(_d)


def script(name):
    """Path of an analysis script, by file name (names are unique under scripts/)."""
    hits = sorted((REPO / "scripts").rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: {len(hits)} matches under scripts/")
    return str(hits[0]).replace("\\", "/")
