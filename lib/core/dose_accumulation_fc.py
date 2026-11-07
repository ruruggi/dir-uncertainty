"""
core/dose_accumulation_fc.py — Multi-fraction dose accumulation with sample-level
uncertainty propagation; supports both the 'fully correlated' (FC) and the
'not correlated' (NC) cases of Smolders et al. (2023, §2.7.3).

Difference from the `dose_accumulation.py` module
=================================================
The legacy module aggregates only per-fraction statistics (mean_i, std_i) and
computes Var_acc = sum_i w_i^2 * Var_i. This is exact for NC (independent
fractions) but wrong for FC (the inter-fraction covariance terms are missing).

In FC, the same white noise eps_k is shared by all fractions of the
treatment (Smolders 2023). This introduces positive covariances between the
doses of the different fractions; computing them in closed form requires the
inter-fraction DVF correlation matrix, which is generally not available.

Solution valid for both FC and NC: aggregate the dose AT SAMPLE LEVEL.
For each index k = 1..K (global sample), compose:

    D_acc_k(x) = sum_i w_i * D_{i,k}(x)             (1)

where D_{i,k}(x) is the dose of fraction i propagated through sample k of the
DVF ensemble of that fraction. If the K eps_k are shared (FC), the D_{i,k}
are correlated; if they are independent (NC), the D_{i,k} are independent.
The final distribution of D_acc is then characterised by mean and standard
deviation OVER THE INDEX k:

    mean_acc(x) = (1/K) * sum_k D_acc_k(x)         (2)
    std_acc(x)  = sqrt( (1/(K-1)) * sum_k (D_acc_k(x) - mean_acc(x))^2 )

This formulation is valid for any inter-fraction correlation model
(FC, NC, or any partial correlation): it is enough that the samples
{eps_k} were generated with the desired statistics.

Calling convention
==================
For each fraction i, the caller must provide the dose-on-fixed
FOR EVERY sample k (shape (K, Z, Y, X), typically a memmap on disk) +
the weight w_i. The accumulator keeps a running sum_i w_i*D_{i,k} for each k
in a separate memmap of shape (K, Z, Y, X).

Usage example (pseudocode):

    fixed_shape = (Z, Y, X)
    K = 100
    acc = SampleEnsembleDoseAccumulator(fixed_shape, n_samples=K,
                                         out_dir="path/")

    # In FC mode, generate the shared epsilon ONCE:
    shared_eps = DVFSampler.draw_shared_noise(K, fixed_shape)

    for i, frac in enumerate(fractions):
        # (i) generate the DVF ensemble for fraction i
        sampler.generate(mu_i, sigma_i, ell_i, out_i,
                         shared_white_noise=shared_eps)  # FC mode

        # (ii) propagate the dose for each sample k of the ensemble
        dose_per_sample_path = warp_dose_per_sample(dose_i, out_i)
        # shape (K, Z, Y, X) memmap

        # (iii) add to the accumulator
        acc.add_per_sample(dose_per_sample_path, weight=frac.weight,
                           frac_id=frac.id)

    result = acc.finalize()  # mean, std, predicted (n/m extrapolation), metadata
    acc.save(out_dir)

References
===========
[Smolders 2023] §2.7.3, p. 7:
    "...additional assumptions need to be taken about the temporal
    correlation between these DVF samples [...]. For simplicity, only two
    cases are considered: fully correlated (FC) and not correlated (NC).
    In the case of full correlation, the same eps_k is used for each DVF
    sample of the different CTs (equation (4)). Without correlation, eps_k
    is sampled independently for each DVF."
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np


class SampleEnsembleDoseAccumulator:
    """Multi-fraction dose accumulator with sample-level propagation.

    Works for any inter-fraction correlation mode (FC, NC, or
    intermediate): it is enough that the caller generates the dose_per_sample_i
    with the desired statistics for the samples.

    Storage
    -------
    out_dir/dose_acc_per_sample.npy : memmap (K, Z, Y, X), running sum
        sum_i w_i * D_{i,k}.
    finalize() loads the memmap and computes mean/std along the k axis.
    """

    def __init__(self,
                 fixed_shape: Tuple[int, int, int],
                 n_samples: int,
                 out_dir: str,
                 dtype=np.float32,
                 n_total: Optional[int] = None,
                 m_delivered: Optional[int] = None,
                 correlation_mode: str = "user-supplied"):
        """
        Parameters
        ----------
        fixed_shape : (Z, Y, X) of the fixed CT.
        n_samples   : K, number of global samples.
        out_dir     : directory where the running memmap is saved.
        n_total/m_delivered : optional, enables adaptive-RT extrapolation.
        correlation_mode : informative string for the metadata; typically
            'FC' (fully correlated, shared epsilon), 'NC' (not correlated,
            independent epsilon per fraction), or 'user-supplied'.
        """
        self.fixed_shape = tuple(int(s) for s in fixed_shape)
        self.n_samples = int(n_samples)
        self.dtype = dtype
        self.out_dir = out_dir
        self.n_total = int(n_total) if n_total is not None else None
        self.m_delivered = int(m_delivered) if m_delivered is not None else None
        self.correlation_mode = str(correlation_mode)

        os.makedirs(out_dir, exist_ok=True)
        # Filename with mode suffix (fc/nc): allows writing NC and FC in
        # parallel in the same out_dir without conflicts.
        suffix = self.correlation_mode.lower()
        # Sanitize: accepts "FC"/"NC"/"fc"/"nc"; for other modes uses the raw
        # value (lowercased) as suffix.
        self._running_path = os.path.join(
            out_dir, f"dose_acc_per_sample_{suffix}.npy")
        # Memmap allocation; explicit zeroing (open_memmap mode='w+' does
        # not guarantee zeros on all filesystems).
        self._running = np.lib.format.open_memmap(
            self._running_path, mode="w+", dtype=dtype,
            shape=(self.n_samples,) + self.fixed_shape,
        )
        self._running[:] = 0.0
        self._running.flush()
        self.entries: List[dict] = []

    # -----------------------------------------------------------------
    def add_per_sample(self,
                       dose_per_sample: Union[str, np.ndarray, np.memmap],
                       weight: float, frac_id: str,
                       permute_seed: Optional[int] = None) -> None:
        """Adds w_i * D_{i,k} to the running sum for every k.

        Parameters
        ----------
        dose_per_sample : path to the (K, Z, Y, X) memmap, or equivalent array.
        weight          : w_i (e.g. 1.0 for a direct sum, or the dosimetric
                          fractions of the plan).
        frac_id         : fraction label, written to the metadata.
        permute_seed    : if None, uses the original sample order (FC mode:
            the correlation shared across fractions is preserved, because
            all fractions access the same index k, which corresponds
            to the same shared epsilon_k). If an integer, applies a seeded
            deterministic permutation of the k axis: this breaks the
            correlation (NC mode with shared-noise reuse), since
            the index k of the running sum no longer corresponds to the same
            epsilon shared across fractions.
        """
        if isinstance(dose_per_sample, str):
            arr = np.load(dose_per_sample, mmap_mode="r")
        else:
            arr = dose_per_sample
        if arr.shape != (self.n_samples,) + self.fixed_shape:
            raise ValueError(
                f"Fraction '{frac_id}': dose_per_sample shape {arr.shape} "
                f"!= expected ({self.n_samples}, *{self.fixed_shape}).")
        w = float(weight)
        K = self.n_samples
        if permute_seed is not None:
            rng = np.random.default_rng(int(permute_seed))
            perm = rng.permutation(K)
        else:
            perm = None
        # Streaming chunk-by-chunk to limit RAM on large volumes.
        # One pass over the k axis.
        for k in range(K):
            src_k = int(perm[k]) if perm is not None else k
            self._running[k] += w * arr[src_k].astype(self.dtype, copy=False)
        self._running.flush()
        self.entries.append({
            "id": str(frac_id),
            "weight": w,
            "permute_seed": permute_seed,
            "k_min": int(arr.shape[0]),
            "voxel_min": float(arr.min()) if arr.size > 0 else 0.0,
            "voxel_max": float(arr.max()) if arr.size > 0 else 0.0,
        })

    # -----------------------------------------------------------------
    def finalize(self) -> dict:
        """Computes mean and std along the k axis. The statistics are correct
        also in FC (covariance terms included automatically by the sample-level
        propagation). Returns a dict with mean/std/metadata; optionally
        also predicted_mean/predicted_std if n_total and m_delivered are given.
        """
        # Load the finalized memmap for reading (reopen the same file).
        running = np.load(self._running_path, mmap_mode="r")
        mean = running.mean(axis=0).astype(self.dtype)
        # unbiased std; for K=1, std=0 by convention.
        if self.n_samples > 1:
            std = running.std(axis=0, ddof=1).astype(self.dtype)
        else:
            std = np.zeros_like(mean)
        m_eff = self.m_delivered if self.m_delivered is not None else len(self.entries)
        n_eff = self.n_total
        result = {
            "mean": mean,
            "std": std,
            "metadata": {
                "model": "sample_level_propagation",
                "correlation_mode": self.correlation_mode,
                "n_fractions": len(self.entries),
                "n_samples": self.n_samples,
                "fractions": self.entries,
                "total": {
                    "mean_min": float(mean.min()),
                    "mean_max": float(mean.max()),
                    "std_min": float(std.min()),
                    "std_max": float(std.max()),
                },
            },
        }
        # Extrapolation adaptive RT (n/m).
        if n_eff is not None and m_eff is not None and m_eff > 0 and n_eff != m_eff:
            scale = float(n_eff) / float(m_eff)
            predicted_mean = (scale * mean).astype(self.dtype)
            predicted_std = (scale * std).astype(self.dtype)
            result["predicted_mean"] = predicted_mean
            result["predicted_std"] = predicted_std
            result["metadata"]["extrapolation"] = {
                "n_total": int(n_eff),
                "m_delivered": int(m_eff),
                "scale_factor": scale,
                "predicted": {
                    "mean_min": float(predicted_mean.min()),
                    "mean_max": float(predicted_mean.max()),
                    "std_min": float(predicted_std.min()),
                    "std_max": float(predicted_std.max()),
                },
            }
        return result

    # -----------------------------------------------------------------
    def close(self) -> None:
        """Releases the running memmap handle. Useful on Windows, where the
        file stays locked until the memmap object is destroyed."""
        try:
            del self._running
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -----------------------------------------------------------------
    def save(self, out_dir: Optional[str] = None) -> dict:
        out_dir = out_dir or self.out_dir
        os.makedirs(out_dir, exist_ok=True)
        result = self.finalize()
        np.save(os.path.join(out_dir, "dose_accumulated.npy"), result["mean"])
        np.save(os.path.join(out_dir, "dose_accumulated_std.npy"), result["std"])
        if "predicted_mean" in result:
            np.save(os.path.join(out_dir, "dose_predicted_full.npy"),
                    result["predicted_mean"])
            np.save(os.path.join(out_dir, "dose_predicted_full_std.npy"),
                    result["predicted_std"])
        with open(os.path.join(out_dir, "accumulation_metadata.json"), "w") as f:
            json.dump(result["metadata"], f, indent=2)
        return result


# ---------------------------------------------------------------------
# Utility: convert a DVF ensemble + fraction dose into dose_per_sample
# ---------------------------------------------------------------------
def warp_dose_per_sample(dose_volume: np.ndarray,
                          dvf_ensemble_path: str,
                          out_path: str,
                          warp_fn,
                          dtype=np.float32) -> str:
    """For each sample k of the DVF ensemble, warps the dose volume
    and writes it to a memmap (K, Z, Y, X).

    Parameters
    ----------
    dose_volume : (Z, Y, X) fraction dose (planning CT space).
    dvf_ensemble_path : path to the (K, 3, Z, Y, X) memmap produced by DVFSampler.
    out_path : where to save the output memmap.
    warp_fn : callable(dose_arr, dvf_kxyz) -> warped_dose_arr. The caller
              provides the warping routine (from utils.warp.warp_torch or similar).

    Returns
    -------
    out_path itself, after the memmap has been written and flushed.
    """
    dvf_ens = np.load(dvf_ensemble_path, mmap_mode="r")
    K = int(dvf_ens.shape[0])
    Z, Y, X = dose_volume.shape
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype,
                                     shape=(K, Z, Y, X))
    for k in range(K):
        warped = warp_fn(dose_volume, dvf_ens[k])
        out[k] = np.asarray(warped, dtype=dtype)
    out.flush()
    return out_path
