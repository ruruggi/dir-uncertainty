# dir-uncertainty

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24031479.svg)](https://doi.org/10.5281/zenodo.24031479)

Code accompanying the article

> R. Ruggieri, N. Bianchi, S. Naccarato, R. Ruggieri.
> **Closed-form, engine-portable uncertainty prediction of deformable image registration for dose accumulation in adaptive radiotherapy.**
> *Physics in Medicine and Biology* (2026, accepted). DOI: to be added.

The repository contains

* `lib/`: the library: wrappers of the four DIR engines with Procrustes pre-alignment, the heuristic-residual
  uncertainty predictor σ<sub>h</sub>, the anisotropic correlation kernel *K*, diffeomorphism-controlled DVF sampling,
  sample-level dose accumulation, LECE estimation and the one-scalar calibration of *f*;
* `scripts/`: the analysis scripts that produced the tables, figures and numbers of the article;
* `tests/`: consistency checks of the σ<sub>DL</sub> comparison.

## Installation

Python 3.11 and an NVIDIA GPU. The published results were computed with Python 3.11.2 and PyTorch 2.0.1 (CUDA 11.7 build).

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117   # or the build matching your driver
pip install -r requirements.txt
```

Run everything from the repository root with the library on the path:

```bash
export PYTHONPATH=lib          # Windows PowerShell: $env:PYTHONPATH = "lib"
```

## Data, model weights and output folders

All locations are set in `lib/dirunc_paths.py` and can be overridden with environment variables:

| Variable | Default | Content |
|---|---|---|
| `DIRUNC_DATASETS` | `data/Datasets` | the four landmark datasets and the registration outputs written next to them |
| `DIRUNC_MODELS` | `models` | σ<sub>DL</sub> weights (`models/L20S3-4`) |
| `DIRUNC_RESULTS` | `results` | outputs of the analysis scripts |
| `DIRUNC_RESULTS_DATA` | `data/results_large` | DVF/dose ensembles of the dose-accumulation example (hundreds of GB) |
| `DIRUNC_LUNG_SBRT`, `DIRUNC_ACCUM_STUDY` | `data/...` | institutional lung SBRT case (not redistributed, see below) |

**Datasets** (public, landmark-annotated CT–CT pairs), placed under `DIRUNC_DATASETS`:

| Folder | Dataset | Reference |
|---|---|---|
| `Lung-DIR-QA/` | 30 thoracic 4D-CT pairs | Criscuolo et al. 2024, doi:10.1002/mp.17026 |
| `AbdominalDIR-QA/` | 30 abdominal CT pairs | Criscuolo et al. 2025, doi:10.1002/mp.17907 |
| `Liver-DIR-QA/` | 30 of the 40 published hepatic CT pairs | Zhang et al. 2025, doi:10.1002/mp.17507 |
| `Castillo-DIR-LAB/` | DIR-LAB, 10 thoracic 4D-CT patients | Castillo et al. 2009, 2010 (dir-lab.com) |

`scripts/C00_data_conversion/convert_castillo_to_nifti.py` converts the DIR-LAB release to NIfTI. The registration
scripts (`scripts/C01_registration/`) write one folder per engine next to each dataset
(`Results_<engine>_procrustes/<case>/`), which is the layout read by `lib/validation/build_v11_data.py`.

**Model weights** are not redistributed:

* σ<sub>DL</sub>: model `L20S3-4` of Smolders et al. (2023), from https://github.com/AndreasSmolders/DIRUncertainty,
  copied to `models/L20S3-4`;
* uniGradICON (Tian et al. 2024): downloaded automatically by the `unigradicon` package on first use.

**Institutional case.** The lung SBRT case of the dose-accumulation example cannot be redistributed. The scripts expect
the fraction folders `Fx_1..3` with the registrations named `reg_Fx<n>.dcm` and the re-aligned doses named
`RD_Fx<n>_isoalign.dcm` (the original DICOM file names were replaced by these neutral names).

## Registration settings used in the article

| Engine | Settings |
|---|---|
| ANTs | `--type_of_transform SyNRA --prealign procrustes` (note: the library default is `SyN` without pre-alignment) |
| ConvexAdam | package defaults (MIND-SSC), `--prealign procrustes` |
| uniGradICON | package defaults, `--prealign procrustes` |
| Elastix | default B-spline parameter map of `itk-elastix`, `--prealign procrustes` |

Procrustes pre-alignment is a rigid Kabsch fit on the annotated landmarks (`lib/core/rigid_align.py`).

## Where each part of the article is implemented

| Article | Implementation |
|---|---|
| Algorithm 1, σ<sub>h</sub> | `sigma_body_local` in `scripts/C04_sigma_h/conventions_check.py` |
| Algorithm 2, kernel *K* | `lib/core/grf.py` (`anisotropic_correlation`) |
| Algorithm 3, diffeomorphism-controlled sampling | `lib/core/sampling.py` (`DVFSampler`); screening as used for the results in `scripts/C16_dose_accumulation/BL/fig3_BL_runner.py` and `scripts/C14_correlation_kernel/q21_kernel_metrics_BLv2.py` |
| Algorithm 4, calibration of *f*, LECE | `lib/validation/calibrate.py`, `lib/validation/lece.py` |
| Table 1 (RMSE TRE per dataset and engine) | `scripts/C06_registration_accuracy/` |
| Table 2 (calibration and discrimination) | `scripts/C07_calibration_lece/q22_table23_v2.py`, `scripts/C23_revision_analyses/q26c_rho_q10.py`, `q31_lece_perdir.py` |
| Figure 1 (effect of the weight *w*) | `scripts/C23_revision_analyses/q22_wsweep_5fold.py` → `q26d_figure4_q10.py` |
| Figure 2 (reliability diagrams) | `scripts/C23_revision_analyses/q22_fig1_v4.py` (drawing code in `scripts/C07_calibration_lece/make_figure1_4x4.py`) |
| Figure 3 (evaluation of *K*) | `scripts/C23_revision_analyses/q25_const_lengths.py`, `q25_const_lengths_synra298.py`, `q32_figure2_trunc.py`, `q35_figure3_3panels.py` (kernel metrics in `scripts/C14_correlation_kernel/`) |
| Figure 4 (dose accumulation) | `scripts/C23_revision_analyses/q25_fig3_const.py` (ensembles via `fig3_BL_runner.py`) → `q36_figure4_panel_b.py`; dose bands `q25_oar_bands_A2.py` |
| §3.1 constant-σ baseline and log-likelihood | `scripts/C23_revision_analyses/q33_const_baseline.py`, `q34_nll_baseline.py` |
| §3.1 between-patient variability, σ<sub>DL</sub> on DIR-LAB | `q26c_between_patients.py`, `q26b_dirlab_q10.py` |
| §3.2 per-axis RMSE of the landmark error, truncation ratio | `q37_directional_tre_q10.py`, `q33_trunc_ratio.py` |
| Appendix C (σ<sub>h</sub> vs σ<sub>DL</sub>, Figure C1) | `scripts/C10_sigma_dl_comparison/`, `scripts/C11_sigma_dl_resolution/` |

Folders `C12`, `C13`, `C17` and `C22` hold supporting analyses quoted in the text or in the review.

## Notes

* The scripts are the record of the analyses, kept as they were run. Several of them read intermediate outputs of
  earlier steps from `results/` (for example `results/_out_q22_step6/cache.npz`), and some execute another script of
  the repository with a documented string patch. They are not a one-command pipeline.
* `lib/core/jacobian.py` (`folding_rate`, `check_diffeomorphism`) measures folding over the whole volume with the
  displacement in mm. The screening of the article (Eq. 13, inside the body mask, voxel units) is computed in the
  analysis scripts listed above.

## License and citation

Apache License 2.0 (`LICENSE`); third-party code and its MIT license are listed in `NOTICE`.
If you use this code, please cite the article above (`CITATION.cff`). The code is archived on Zenodo:
doi:[10.5281/zenodo.24031479](https://doi.org/10.5281/zenodo.24031479) (all versions).
