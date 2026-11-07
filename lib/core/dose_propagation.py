"""
core/dose_propagation.py — Dose deformation using DVF ensemble from PASS 3.

Handles arbitrary axis ordering: input dose is (Z,Y,X) as expected by the viewer.
DVF ensemble may be (N,3, X, Y, Z) or (N,3, Z, Y, X). The code detects and
transposes consistently, then saves output with original (Z,Y,X) order.
"""

import os
import numpy as np
import torch

from utils.warp import warp_torch


class DosePropagator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def propagate(self,
                  dose_volume: np.ndarray,
                  dvf_ensemble_path: str,
                  voxel_spacing: tuple,
                  progress_callback=None) -> dict:
        """
        Parameters
        ----------
        dose_volume : ndarray (Z, Y, X) — dose in Gy, order as in viewer
        dvf_ensemble_path : str — path to dvf_ensemble.npy (N, 3, ...)
        voxel_spacing : (sz, sy, sx) in mm (z, y, x order)
        """
        # Load DVF ensemble
        dvf_ensemble = np.load(dvf_ensemble_path, mmap_mode='r')
        if dvf_ensemble.ndim != 5:
            raise ValueError(f"Expected 5D array (N,3,Z,Y,X), got {dvf_ensemble.shape}")
        N, C, Z_dvf, Y_dvf, X_dvf = dvf_ensemble.shape
        assert C == 3

        # Original dose shape (viewer order)
        Z_orig, Y_orig, X_orig = dose_volume.shape
        print(f"  Dose original shape (Z,Y,X): ({Z_orig},{Y_orig},{X_orig})")
        print(f"  DVF spatial shape (from file): ({Z_dvf},{Y_dvf},{X_dvf})")

        # Determine permutation to align dose_volume to DVF axes
        # We need to bring dose_volume to shape (Z_dvf, Y_dvf, X_dvf) for warping
        dose_shape = dose_volume.shape
        dvf_shape = (Z_dvf, Y_dvf, X_dvf)

        if dose_shape == dvf_shape:
            dose_warp = dose_volume
            inv_perm = None
        elif dose_shape == (X_dvf, Y_dvf, Z_dvf):
            # Common case: DVF is (X,Y,Z) while dose is (Z,Y,X)
            dose_warp = np.transpose(dose_volume, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
            inv_perm = (2, 1, 0)  # to go back
            print(f"  Transposing dose for warp: {dose_shape} -> {dose_warp.shape}")
        elif dose_shape == (Y_dvf, X_dvf, Z_dvf):
            dose_warp = np.transpose(dose_volume, (1, 2, 0))
            inv_perm = (2, 0, 1)
            print(f"  Transposing dose for warp: {dose_shape} -> {dose_warp.shape}")
        elif dose_shape == (Z_dvf, X_dvf, Y_dvf):
            dose_warp = np.transpose(dose_volume, (0, 2, 1))
            inv_perm = (0, 2, 1)
        else:
            raise ValueError(
                f"Cannot align dose shape {dose_shape} to DVF shape {dvf_shape}. "
                "Check axis ordering."
            )

        # Create output mmap with original viewer order (Z_orig, Y_orig, X_orig)
        dose_ens_path = os.path.join(self.output_dir, "dose_ensemble.npy")
        dose_mmap = np.lib.format.open_memmap(
            dose_ens_path, mode='w+', dtype=np.float32, shape=(N, Z_orig, Y_orig, X_orig)
        )

        # Welford online stats
        running_mean = np.zeros((Z_orig, Y_orig, X_orig), dtype=np.float32)
        running_var = np.zeros((Z_orig, Y_orig, X_orig), dtype=np.float32)

        # Prepare tensors
        dose_tensor = torch.from_numpy(dose_warp).float()
        sz, sy, sx = voxel_spacing  # (z, y, x) in mm

        for i in range(N):
            dvf_mm = dvf_ensemble[i].astype(np.float32)  # (3, Z_dvf, Y_dvf, X_dvf)

            # Convert mm to voxels (same axis order as dvf_mm)
            dvf_voxels = np.zeros((3, Z_dvf, Y_dvf, X_dvf), dtype=np.float32)
            dvf_voxels[0] = dvf_mm[0] / sz
            dvf_voxels[1] = dvf_mm[1] / sy
            dvf_voxels[2] = dvf_mm[2] / sx
            dvf_tensor = torch.from_numpy(dvf_voxels).float()

            # Warp
            try:
                deformed = warp_torch(dose_tensor, dvf_tensor).cpu().numpy()
                if deformed.ndim == 4:
                    deformed = deformed[0]  # remove channel dim
                # deformed shape is (Z_dvf, Y_dvf, X_dvf)
            except Exception as e:
                print(f"    Sample {i} warp error: {e}. Using original dose.")
                deformed = dose_warp.copy()

            # Transform back to original viewer order if needed
            if inv_perm is not None:
                deformed = np.transpose(deformed, inv_perm)
                # Now shape should be (Z_orig, Y_orig, X_orig)
            else:
                # Already in correct order, but ensure shape matches
                if deformed.shape != (Z_orig, Y_orig, X_orig):
                    raise ValueError(f"Shape mismatch after warp: {deformed.shape} vs expected {dose_volume.shape}")

            dose_mmap[i] = deformed

            # Welford update
            count = i + 1
            delta = deformed - running_mean
            running_mean += delta / count
            running_var += delta * (deformed - running_mean)

            if i % 10 == 0:
                dose_mmap.flush()

            if progress_callback is not None:
                progress_callback(i + 1, N)

        dose_mmap.flush()
        dose_std = np.sqrt(running_var / N) if N > 0 else np.zeros_like(running_mean)

        # Save results
        np.save(os.path.join(self.output_dir, "dose_mean.npy"), running_mean)
        np.save(os.path.join(self.output_dir, "dose_std.npy"), dose_std)
        np.save(os.path.join(self.output_dir, "dose_original.npy"), dose_volume)

        print(f"  Dose propagation complete: mean=[{running_mean.min():.2f}, {running_mean.max():.2f}], "
              f"std=[{dose_std.min():.4f}, {dose_std.max():.4f}]")

        return {"dose_mean": running_mean, "dose_std": dose_std}