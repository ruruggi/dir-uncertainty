"""
core/sampling.py — DVF ensemble sampler.

Supports two sampling modes for the white noise epsilon (Smolders
et al. 2023 §2.7.3):

  - NC ("not correlated"): epsilon is independent for each sample k and for
    each fraction i. This is the default (shared_white_noise=None).
  - FC ("fully correlated"): epsilon_k is the same across the different
    fractions of a treatment; only the sample index k varies. It is enabled
    by passing to generate() a shared_white_noise argument pre-generated via
    DVFSampler.draw_shared_noise().

In FC mode, the noise structure (spatial pattern) is identical across
fractions; only the anisotropic modulation K_i(x) (which depends on the
map l_d(x) of fraction i, derived from the gradient of the fixed CT_i) and
the amplitude G_i(x) change per fraction. This reproduces Smolders'
"fully correlated" mode extended to our spatially adaptive framework.
"""

import os
import json
import threading
from typing import List, Optional, Union
import numpy as np
import torch
from core.grf import anisotropic_correlation
from core.jacobian import check_diffeomorphism


class _LazyFCNoise:
    """RAM-O(1) lazy provider of shared FC white-noise tensors.

    See ``DVFSampler.draw_shared_noise`` for context. Each call to
    ``self[k]`` regenerates epsilon_k deterministically from ``base_seed + k``,
    so the FC semantics (same epsilon_k across all fractions) is preserved
    without ever materialising the full list in memory.
    """

    def __init__(self, n_samples: int, shape_zyx: tuple, base_seed: int,
                 dtype: torch.dtype = torch.float32):
        self.n_samples = int(n_samples)
        self.shape_zyx = tuple(shape_zyx)
        self.base_seed = int(base_seed)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, k: int) -> torch.Tensor:
        if not 0 <= int(k) < self.n_samples:
            raise IndexError(f"Sample index k={k} out of range "
                             f"[0, {self.n_samples}).")
        Z, Y, X = self.shape_zyx
        # Per-k Generator: same `base_seed + k` everywhere this provider is
        # consumed -> identical epsilon_k across all fractions of the same
        # treatment plan.
        gen = torch.Generator()
        gen.manual_seed(self.base_seed + int(k))
        eps = torch.empty(3, Z, Y, X, dtype=self.dtype)
        eps.normal_(0.0, 1.0, generator=gen)
        return eps


class DVFSampler:
    def __init__(self,
                 n_samples: int = 100,
                 kernel_size: int = 61,
                 variability_gain: float = 1.0,
                 max_retries: int = 20,
                 folding_threshold: float = 0.0001,
                 gain_decay: float = 0.85,
                 exclude_gpus: list = None):
        self.n_samples = n_samples
        self.kernel_size = kernel_size
        self.variability_gain = variability_gain
        self.max_retries = max_retries
        self.folding_threshold = folding_threshold
        self.gain_decay = gain_decay
        self.exclude_gpus = exclude_gpus or []

    def get_valid_gpus(self):
        n = torch.cuda.device_count()
        return [i for i in range(n) if i not in self.exclude_gpus]

    @staticmethod
    def draw_shared_noise(n_samples: int, shape_zyx: tuple,
                          generator: Optional[torch.Generator] = None,
                          dtype: torch.dtype = torch.float32,
                          base_seed: Optional[int] = None):
        """Lazy provider of N white-noise tensors shared across fractions
        in FC mode ("fully correlated" Smolders 2023 §2.7.3).

        Previous implementation (< v7, deprecated): materialised the N
        tensors in a CPU list. For (3, 90, 512, 512) f32 -> 283 MB/sample ->
        ~141 GB at n_samples=500: infeasible. It crashed in `torch.empty(3,
        Z, Y, X)` (CPU allocator OOM).

        Current implementation: returns an object that generates epsilon_k
        on demand from `base_seed + k` when indexed with []. The FC semantics
        are preserved (the same epsilon_k runs through all fractions thanks
        to the deterministic seed) but the peak RAM drops to O(1 tensor)
        = ~283 MB regardless of n_samples.

        Parameters
        ----------
        n_samples : number of global samples (k = 0..N-1).
        shape_zyx : (Z, Y, X) of the fixed CT.
        generator : optional torch.Generator; if provided, `initial_seed()`
                    is used as `base_seed` for backward compatibility.
        base_seed : optional int; if not None, it overrides `generator`. If
                    both are None, default = 12345.
        dtype     : default float32.

        Returns
        -------
        _LazyFCNoise: object with len() and __getitem__ that deterministically
            regenerates epsilon_k as a CPU torch.Tensor of shape
            (3, Z, Y, X) at every access.
        """
        if base_seed is None:
            if generator is not None:
                # Extract the initial seed from the Generator (backward compatible
                # with runner.py, which passes `gen = torch.Generator().manual_seed(args.noise_seed)`).
                try:
                    base_seed = int(generator.initial_seed())
                except Exception:
                    base_seed = 12345
            else:
                base_seed = 12345
        return _LazyFCNoise(int(n_samples), tuple(shape_zyx),
                            int(base_seed), dtype)

    def generate(self, mu_dvf, sigma_map, ell_map, output_dir,
                 progress_callback=None,
                 shared_white_noise: Optional[Union[List[torch.Tensor], _LazyFCNoise]] = None):
        os.makedirs(output_dir, exist_ok=True)
        if mu_dvf.dim() == 5:
            mu_dvf = mu_dvf.squeeze(0)
        if sigma_map.dim() == 5:
            sigma_map = sigma_map.squeeze(0)
        C, Z, Y, X = mu_dvf.shape
        assert C == 3
        dtype = np.float32
        N = self.n_samples

        # Validation of shared_white_noise (FC mode).
        # Accepts either a legacy list (backward compatible for small N) or a
        # _LazyFCNoise provider (default for N >= ~150, where the list blows up
        # CPU RAM). In both cases we check len() and the shape of the
        # first element — iterating over all elements would defeat the lazy
        # provider (it would regenerate N tensors just for the check).
        if shared_white_noise is not None:
            if len(shared_white_noise) != N:
                raise ValueError(
                    f"shared_white_noise has {len(shared_white_noise)} entries, "
                    f"expected n_samples={N}.")
            if N > 0:
                _eps0 = shared_white_noise[0]
                if tuple(_eps0.shape) != (3, Z, Y, X):
                    raise ValueError(
                        f"shared_white_noise[0] has shape {tuple(_eps0.shape)}, "
                        f"expected (3, {Z}, {Y}, {X}).")
                del _eps0
            kind = type(shared_white_noise).__name__
            print(f"  [FC] Sampling with shared white noise across fractions "
                  f"(N={N}, source={kind}). Retries reuse the same epsilon "
                  f"(only G amplitude decays).")

        ensemble_path = os.path.join(output_dir, "dvf_ensemble.npy")
        mag_path = os.path.join(output_dir, "dvf_ensemble_mag.npy")
        ds = 8
        Zd, Yd, Xd = (Z+ds-1)//ds, (Y+ds-1)//ds, (X+ds-1)//ds
        vec_path = os.path.join(output_dir, "dvf_vectors_down.npy")

        # Pre-cleanup: on Windows / NTFS, np.lib.format.open_memmap(mode='w+')
        # on a pre-existing multi-GB file occasionally fails with
        # `OSError: [Errno 22] Invalid argument` if a previous run left the
        # file in a sparse / mapped state that the OS cannot truncate
        # cleanly. Removing the file first sidesteps the issue.
        for _p in (ensemble_path, mag_path, vec_path):
            if os.path.exists(_p):
                try:
                    os.remove(_p)
                except OSError as _err:
                    raise OSError(
                        f"Cannot remove stale memmap file {_p}: {_err}. "
                        f"Likely a leftover handle from a killed run; "
                        f"close any Python process holding it and retry.") from _err

        ensemble_mmap = np.lib.format.open_memmap(ensemble_path, mode='w+', dtype=dtype, shape=(N, 3, Z, Y, X))
        mag_mmap = np.lib.format.open_memmap(mag_path, mode='w+', dtype=dtype, shape=(N, Z, Y, X))
        vec_mmap = np.lib.format.open_memmap(vec_path, mode='w+', dtype=dtype, shape=(N, 3, Zd, Yd, Xd))

        valid_gpus = self.get_valid_gpus()
        n_gpus = len(valid_gpus)
        if n_gpus == 0:
            return self._generate_on_device(mu_dvf, sigma_map, ell_map, output_dir,
                                            ensemble_path, mag_path, vec_path,
                                            device='cpu',
                                            progress_callback=progress_callback,
                                            shared_white_noise=shared_white_noise)
        print(f"  Using {n_gpus} GPU(s): {valid_gpus}")
        sample_indices = np.array_split(np.arange(N), n_gpus)
        threads = []
        folding_rates_all = []
        retry_counts_all = []
        for gpu_idx, indices in enumerate(sample_indices):
            if len(indices) == 0:
                continue
            gpu_id = valid_gpus[gpu_idx]
            t = threading.Thread(
                target=self._generate_on_gpu,
                args=(gpu_id, indices, mu_dvf, sigma_map, ell_map,
                      ensemble_path, mag_path, vec_path,
                      folding_rates_all, retry_counts_all,
                      shared_white_noise),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        mag_mmap_global = np.load(mag_path, mmap_mode='r')
        running_mean = np.mean(mag_mmap_global, axis=0).astype(dtype)
        running_var = np.var(mag_mmap_global, axis=0).astype(dtype)
        std_mag = np.sqrt(running_var)
        np.save(os.path.join(output_dir, "dvf_mean_mag.npy"), running_mean)
        np.save(os.path.join(output_dir, "dvf_std_mag.npy"), std_mag)

        folding_stats = {
            "folding_rates": folding_rates_all,
            "retries": retry_counts_all,
            "mean_folding": float(np.mean(folding_rates_all)) if folding_rates_all else 0.0,
            "max_folding": float(np.max(folding_rates_all)) if folding_rates_all else 0.0,
            "mean_retries": float(np.mean(retry_counts_all)) if retry_counts_all else 0.0,
        }
        with open(os.path.join(output_dir, "folding_stats.json"), 'w') as f:
            json.dump(folding_stats, f, indent=4)
        metadata = {
            "n_samples": N,
            "shape_zyx": [Z, Y, X],
            "kernel_size": self.kernel_size,
            "variability_gain": self.variability_gain,
            "folding_threshold": self.folding_threshold,
            "mean_folding_rate": folding_stats["mean_folding"],
        }
        with open(os.path.join(output_dir, "ensemble_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  Ensemble complete: {N} samples")
        return {"mean_mag": running_mean, "std_mag": std_mag, "folding_stats": folding_stats}

    def _generate_on_gpu(self, gpu_id, sample_indices, mu_dvf, sigma_map, ell_map,
                         ensemble_path, mag_path, vec_path,
                         folding_rates_list, retry_counts_list,
                         shared_white_noise: Optional[Union[List[torch.Tensor], _LazyFCNoise]] = None):
        # NOTE: this method is invoked once per GPU from `generate()` via Python
        # threading. Each thread is assigned a disjoint slice of sample_indices
        # (see `np.array_split` upstream) and writes to non-overlapping rows of
        # the memmap files (ensemble_mmap[k], mag_mmap[k], vec_mmap[k] for k in
        # this slice). No two threads ever touch the same memmap row, so the
        # writes are race-free by index partitioning even without explicit locks.
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        mu_gpu = mu_dvf.float().to(device)
        sigma_gpu = sigma_map.float().to(device)
        # Safety warning: if sigma_map is identically zero (e.g. caller passed
        # an empty mask or a misconfigured σ pipeline), all retries collapse to
        # μ and the "ensemble" is degenerate. Warn loudly rather than failing
        # silently.
        if float(sigma_gpu.max().item()) < 1e-6:
            print(f"  [WARNING gpu{gpu_id}] sigma_map.max() < 1e-6; "
                  f"ensemble will collapse to deterministic warp μ (zero variability).")
        ell_gpu = ell_map.float().to(device) if isinstance(ell_map, torch.Tensor) else ell_map
        ensemble_mmap = np.lib.format.open_memmap(ensemble_path, mode='r+', dtype=np.float32)
        mag_mmap = np.lib.format.open_memmap(mag_path, mode='r+', dtype=np.float32)
        vec_mmap = np.lib.format.open_memmap(vec_path, mode='r+', dtype=np.float32)

        # Compute the adaptive threshold (if the threshold is 1.0, the check is disabled)
        if self.folding_threshold >= 1.0:
            adaptive_threshold = 1.0  # disables the check
        else:
            _, base_rate = check_diffeomorphism(mu_gpu, threshold=1.0)
            adaptive_threshold = max(self.folding_threshold, base_rate + 0.0005)

        local_folding = []
        local_retries = []
        for local_idx, k in enumerate(sample_indices):
            retry = 0
            local_gain = self.variability_gain
            accepted_dvf = None
            f_rate = 0.0
            # In FC mode the noise is fixed for sample k (taken from
            # shared_white_noise[k]); the retry only decays the gain.
            fixed_eps = None
            if shared_white_noise is not None:
                fixed_eps = shared_white_noise[k].to(device, non_blocking=True)
            while retry < self.max_retries:
                epsilon = fixed_eps if fixed_eps is not None else torch.randn_like(mu_gpu)
                eps_corr = anisotropic_correlation(epsilon, ell_gpu, kernel_size=self.kernel_size)
                delta = sigma_gpu * local_gain * eps_corr
                z_k = mu_gpu + delta
                if adaptive_threshold >= 1.0:
                    # Check disabled
                    accepted_dvf = z_k
                    f_rate = 0.0
                    break
                accepted_dvf, f_rate = check_diffeomorphism(z_k, threshold=adaptive_threshold)
                if accepted_dvf is not None:
                    break
                local_gain *= self.gain_decay
                retry += 1
            local_folding.append(f_rate)
            local_retries.append(retry)
            if accepted_dvf is None:
                accepted_dvf = mu_gpu
            sample_np = accepted_dvf.cpu().numpy().astype(np.float32)
            ensemble_mmap[k] = sample_np
            mag = np.linalg.norm(sample_np, axis=0)
            mag_mmap[k] = mag
            vec_mmap[k] = sample_np[:, ::8, ::8, ::8]
        folding_rates_list.extend(local_folding)
        retry_counts_list.extend(local_retries)

    def _generate_on_device(self, mu_dvf, sigma_map, ell_map, output_dir,
                            ensemble_path, mag_path, vec_path, device='cpu',
                            progress_callback=None,
                            shared_white_noise: Optional[Union[List[torch.Tensor], _LazyFCNoise]] = None):
        """CPU fallback: sequential loop without threading."""
        mu_dev = mu_dvf.float().to(device)
        sigma_dev = sigma_map.float().to(device)
        ell_dev = ell_map.float().to(device) if isinstance(ell_map, torch.Tensor) else ell_map

        ensemble_mmap = np.lib.format.open_memmap(ensemble_path, mode='r+', dtype=np.float32)
        mag_mmap = np.lib.format.open_memmap(mag_path, mode='r+', dtype=np.float32)
        vec_mmap = np.lib.format.open_memmap(vec_path, mode='r+', dtype=np.float32)

        if self.folding_threshold >= 1.0:
            adaptive_threshold = 1.0
        else:
            _, base_rate = check_diffeomorphism(mu_dev, threshold=1.0)
            adaptive_threshold = max(self.folding_threshold, base_rate + 0.0005)

        folding_rates, retry_counts = [], []
        N = self.n_samples
        for k in range(N):
            retry = 0
            local_gain = self.variability_gain
            accepted_dvf = None
            f_rate = 0.0
            # In FC mode the noise is fixed for sample k.
            fixed_eps = None
            if shared_white_noise is not None:
                fixed_eps = shared_white_noise[k].to(device, non_blocking=True)
            while retry < self.max_retries:
                epsilon = fixed_eps if fixed_eps is not None else torch.randn_like(mu_dev)
                eps_corr = anisotropic_correlation(epsilon, ell_dev, kernel_size=self.kernel_size)
                delta = sigma_dev * local_gain * eps_corr
                z_k = mu_dev + delta
                if adaptive_threshold >= 1.0:
                    accepted_dvf = z_k
                    f_rate = 0.0
                    break
                accepted_dvf, f_rate = check_diffeomorphism(z_k, threshold=adaptive_threshold)
                if accepted_dvf is not None:
                    break
                local_gain *= self.gain_decay
                retry += 1
            folding_rates.append(f_rate)
            retry_counts.append(retry)
            if accepted_dvf is None:
                accepted_dvf = mu_dev
            sample_np = accepted_dvf.cpu().numpy().astype(np.float32)
            ensemble_mmap[k] = sample_np
            mag_mmap[k] = np.linalg.norm(sample_np, axis=0)
            vec_mmap[k] = sample_np[:, ::8, ::8, ::8]
            if progress_callback:
                progress_callback(int((k + 1) / N * 100))
        return {
            "folding_rates": folding_rates,
            "retry_counts": retry_counts,
            "mean_folding": float(np.mean(folding_rates)) if folding_rates else 0.0,
        }