import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np
import os
import glob
from dl_arch.Model.gaussianBlur import GaussianBlur
from utils.constants import EPS
from utils.interpolate import interpolateArray
from utils.miscellaneous import roundUpToOdd
from utils.grf import adaptive_ell_map, anisotropic_correlation, jacobian_check

PROBABLITY_MODEL_DIAGONAL = "diagonal"
PROBABLITY_MODEL_BLOCK_DIAGONAL = "blockDiagonal"


class NormalSamplingLayer(nn.Module):
    def __init__(self, blurringLayer=None, nbSamples=1, downSampleFactor=1):
        super().__init__()
        self.fullResolutionBlurringLayer = blurringLayer
        self._setSampleDownSamplingFactor(downSampleFactor)
        self.nbSamples = nbSamples

    def forward(self, mean, stdev):
        if not self.training:
            return mean
        return mean + stdev * self.getStandardSample(mean.shape, mean.device)

    def getStandardSample(self, shape, device):
        dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        targetShape = self.getSampleShape(shape)
        eps1 = dist.rsample(targetShape)[..., 0].to(device)
        if self.blurringLayer is not None:
            eps1 = self.blurringLayer(eps1)
        eps1 = interpolateArray(eps1, shape[-3:])
        return eps1

    def getSampleShape(self, shape):
        sampleShape = torch.tensor(shape)
        sampleShape[-3:] = torch.div(
            sampleShape[-3:], self.sampleDownSamplingFactor, rounding_mode="floor"
        )
        sampleShape[0] = sampleShape[0] * self.nbSamples
        if self.blurringLayer is not None and self.blurringLayer.padding == 0:
            sampleShape[-3:] = (
                sampleShape[-3:] + torch.tensor(self.blurringLayer.kernel_size) - 1
            )
        return sampleShape

    def _setSampleDownSamplingFactor(self, newDownSamplingFactor):
        self.sampleDownSamplingFactor = newDownSamplingFactor
        if newDownSamplingFactor == 1:
            self.blurringLayer = self.fullResolutionBlurringLayer
        else:
            if self.fullResolutionBlurringLayer is not None:
                assert self.fullResolutionBlurringLayer.rescale
                kernelSize = tuple(
                    roundUpToOdd(
                        np.array(self.fullResolutionBlurringLayer.kernel_size)
                        // newDownSamplingFactor
                    ).astype(np.int16)
                )
                smoothingSigma = tuple(
                    np.array(self.fullResolutionBlurringLayer.sigma) / newDownSamplingFactor
                )
                self.blurringLayer = GaussianBlur(
                    kernelSize,
                    smoothingSigma,
                    padding=self.fullResolutionBlurringLayer.padding,
                    rescale=self.fullResolutionBlurringLayer.rescale,
                )
            else:
                self.blurringLayer = None
                self.downSampleConversionFactor = 1


class AnisotropicSamplingLayer(NormalSamplingLayer):
    """
    Extends NormalSamplingLayer to support anisotropic GRF correlation (Smolders 2.1).
    Supports saving directly to disk to save RAM.
    Computes mean and std incrementally.
    At the end, concatenates all samples into a single ensemble file and deletes the individual files.
    """
    def __init__(self, blurringLayer=None, nbSamples=50, downSampleFactor=1, 
                 base_ell=(10, 10, 15), kernel_size=31, 
                 adaptation_gain=0.5, variability_gain=1.0):
        super().__init__(blurringLayer, nbSamples, downSampleFactor)
        self.base_ell = base_ell
        self.kernel_size = kernel_size
        self.adaptation_gain = adaptation_gain
        self.variability_gain = variability_gain

    def _normalize_tensor(self, tensor, name="tensor"):
        """Normalise the tensor dimensions to (C, Z, Y, X)"""
        if tensor is None:
            return None
        
        original_shape = tensor.shape
        tensor_dim = tensor.dim()
        
        if tensor_dim == 5:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                print(f"  {name}: removed batch dimension from {original_shape} to {tensor.shape}")
            else:
                print(f"  {name}: 5D shape but batch>1, shape {original_shape}")
        
        if tensor.dim() == 5:
            tensor = tensor[0]
            print(f"  {name}: took first element, new shape {tensor.shape}")
        
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D (C, Z, Y, X), but has shape {tensor.shape}, dim={tensor.dim()}")
        
        return tensor

    def _compute_magnitude_zyx(self, sample_np):
        """
        Compute the DVF magnitude and transpose it to (D, H, W).
        sample_np shape: (3, X, Y, Z)
        returns shape: (Z, Y, X)
        """
        # Norm along the channels (axis=0) -> (X, Y, Z)
        mag = np.linalg.norm(sample_np, axis=0)
        # Transpose from (X, Y, Z) to (Z, Y, X)
        mag_zyx = np.transpose(mag, (2, 1, 0))
        return mag_zyx

    def _save_sample_and_update_stats(self, sample, sample_idx, output_dir, 
                                       running_mean, running_var, count, dtype=np.float32, 
                                       ensemble_mmap=None, vec_mmap=None):
        """
        Save a single sample to disk and update the incremental statistics.
        Uses Welford's algorithm for the online variance.
        """
        # Convert to numpy and compute the magnitude
        sample_np = sample.cpu().numpy().astype(dtype)
        mag_zyx = self._compute_magnitude_zyx(sample_np)
        
        # DEBUG: print the shapes
        if sample_idx == 0:
            print(f"    DEBUG: mag_zyx shape: {mag_zyx.shape}")
            print(f"    DEBUG: running_mean shape: {running_mean.shape}")
        
        # Save the sample to disk (memmap if available, otherwise a single file)
        if ensemble_mmap is not None:
            ensemble_mmap[sample_idx] = mag_zyx
            ensemble_mmap.flush()
        elif output_dir is not None:
            sample_file = os.path.join(output_dir, f"dvf_sample_{sample_idx:04d}.npy")
            np.save(sample_file, mag_zyx)
        
        # Save downsampled 3D vectors for the quiver plot (if vec_mmap is available)
        if vec_mmap is not None:
            ds = 8
            # Standardise to (3, Z_ds, Y_ds, X_ds) for the viewer
            # sample_np is (3, X, Y, Z)
            vec_ds = np.transpose(sample_np[:, ::ds, ::ds, ::ds], (0, 3, 2, 1)) 
            vec_mmap[sample_idx] = vec_ds
            if sample_idx % 50 == 0:
                vec_mmap.flush()
        
        # Update the incremental statistics (Welford algorithm)
        count_new = count + 1
        delta = mag_zyx - running_mean
        running_mean_new = running_mean + delta / count_new
        delta2 = mag_zyx - running_mean_new
        running_var_new = running_var + delta * delta2
        
        return running_mean_new, running_var_new, count_new

    def _generate_samples_on_gpu(self, mu_dvf, sigma_map, ct_gradients, n_samples, 
                                  gpu_id, ell_map, output_dir, start_idx,
                                  running_mean, running_var, count, dtype=np.float32, 
                                  ensemble_mmap=None, vec_mmap=None):
        """
        Generate samples on one specific GPU and save them to disk.
        Updates the incremental statistics.
        """
        mu_local = mu_dvf.cuda(gpu_id)
        sigma_local = sigma_map.cuda(gpu_id)
        
        if ct_gradients is not None:
            grads_local = ct_gradients.cuda(gpu_id)
        else:
            grads_local = None
        
        if isinstance(ell_map, torch.Tensor):
            ell_map_local = ell_map.cuda(gpu_id)
        else:
            ell_map_local = ell_map
        
        # The target shape is (D, H, W) = (191, 512, 512)
        target_shape = (mu_dvf.shape[3], mu_dvf.shape[1], mu_dvf.shape[2])
        
        # If running_mean has shape (X, Y, Z), transpose to (Z, Y, X)
        if running_mean.shape == (mu_dvf.shape[1], mu_dvf.shape[2], mu_dvf.shape[3]):
            print(f"    GPU {gpu_id}: transposing running_mean from {running_mean.shape} to {target_shape}")
            running_mean = np.transpose(running_mean, (2, 1, 0))
            running_var = np.transpose(running_var, (2, 1, 0))
        
        local_running_mean = running_mean.copy()
        local_running_var = running_var.copy()
        local_count = count
        
        total_folding_rates = []
        total_retries = []
        
        # Compute the adaptive threshold: tolerate the original rate + 0.05%
        _, base_rate = jacobian_check(mu_local, threshold=1.0)
        adaptive_threshold = max(0.0001, base_rate + 0.0005)
        
        for k in range(n_samples):
            retry = 0
            max_retries = 20
            phi_k_checked = None
            f_rate = 0.0
            local_gain = self.variability_gain
            
            while retry < max_retries:
                # Epsilon and Correlated now stay on GPU
                epsilon = torch.randn_like(mu_local)
                correlated = anisotropic_correlation(epsilon, ell_map_local, kernel_size=self.kernel_size)

                # FIX: guarantees K → I when ell → 0
                if isinstance(ell_map_local, torch.Tensor):
                    ell_min = ell_map_local.min()
                    if ell_min < 1e-3:  # threshold to be tuned (in voxels)
                        correlated = epsilon
                
                # Smolders formula: delta = sigma * gain * correlated
                # Casting local_gain to input dtype for safety in FP16/Half mode
                gain_tensor = torch.tensor(local_gain, dtype=mu_local.dtype, device=mu_local.device)
                delta = sigma_local * gain_tensor * correlated
                phi_k = mu_local + delta
                
                # Check Jacobian consistency on GPU (no toNumpy() here!)
                phi_k_checked, f_rate = jacobian_check(phi_k, threshold=adaptive_threshold)
                if phi_k_checked is not None:
                    if retry > 0:
                        print(f"    GPU {gpu_id}: Sample {k} accepted after {retry} attempts (rate: {f_rate:.6f}, gain reduced to {local_gain:.3f}).")
                    break
                
                # DYNAMIC DECAY: Reduce perturbation strength by 15% to help pass Jacobian check
                local_gain *= 0.85
                retry += 1
            
            total_folding_rates.append(f_rate)
            total_retries.append(retry)
                
            if phi_k_checked is None:
                print(f"    GPU {gpu_id}: [WARNING] Sample {k} failed after {max_retries} attempts (rate: {f_rate:.6f}). Using the mean.")
                phi_k_checked = mu_local
            
            local_running_mean, local_running_var, local_count = self._save_sample_and_update_stats(
                phi_k_checked, start_idx + k, output_dir,
                local_running_mean, local_running_var, local_count, dtype, ensemble_mmap, vec_mmap
            )
            
            # Memory Cleanup
            del epsilon, correlated, delta, phi_k
            if k % 20 == 0:
                torch.cuda.empty_cache()
            
        # Save the folding statistics to disk if output_dir is given
        if output_dir is not None:
            import json
            stats_path = os.path.join(output_dir, f"folding_stats_gpu_{gpu_id}.json")
            with open(stats_path, "w") as f:
                json.dump({
                    "folding_rates": total_folding_rates,
                    "retries": total_retries,
                    "mean_folding": float(np.mean(total_folding_rates)),
                    "max_folding": float(np.max(total_folding_rates)),
                    "retry_rate": float(np.mean([1 if r > 0 else 0 for r in total_retries]))
                }, f, indent=4)
        
        return local_running_mean, local_running_var, local_count

    def _concat_and_cleanup(self, output_dir, count):
        """
        Concatenate all individual samples into a single ensemble file
        and delete the individual files to save space.
        """
        if output_dir is None or count == 0:
            return
        
        # Find all the sample files
        sample_files = sorted(glob.glob(os.path.join(output_dir, "dvf_sample_*.npy")))
        
        if len(sample_files) != count:
            print(f"  Warning: found {len(sample_files)} files, expected {count}")
            return
        
        print(f"  Concatenating {len(sample_files)} samples into dvf_sampled_ensemble.npy...")
        
        # Load and concatenate one sample at a time (memory-optimised)
        ensemble_list = []
        for sf in sample_files:
            ensemble_list.append(np.load(sf))
        
        ensemble_full = np.stack(ensemble_list)
        ensemble_path = os.path.join(output_dir, "dvf_sampled_ensemble.npy")
        np.save(ensemble_path, ensemble_full)
        print(f"  Ensemble saved: {ensemble_full.shape} ({ensemble_full.nbytes / 1e9:.2f} GB)")
        
        # Delete the individual files
        for sf in sample_files:
            os.remove(sf)
        print(f"  Individual files deleted ({len(sample_files)} files removed)")

    def forward(self, mu_dvf, sigma_map, ct_gradients=None, nb_samples=None, output_dir=None):
        """
        Generates an ensemble of DVF samples.
        mu_dvf: (3, Z, Y, X) Mean DVF
        sigma_map: (3, Z, Y, X) Voxel-wise standard deviation (Uncertainty)
        ct_gradients: (3, Z, Y, X) Image gradients for adaptive correlation.
        nb_samples: Optional override for the number of samples to generate.
        output_dir: Directory in which to save the samples on disk (if None, nothing is saved)
        """
        n_samples = nb_samples if nb_samples is not None else self.nbSamples
        
        # Normalise tensors
        mu_dvf = self._normalize_tensor(mu_dvf, "mu_dvf")
        sigma_map = self._normalize_tensor(sigma_map, "sigma_map")
        
        if ct_gradients is not None:
            ct_gradients = self._normalize_tensor(ct_gradients, "ct_gradients")
        
        if mu_dvf.shape[0] != 3:
            raise ValueError(f"mu_dvf must have 3 channels (X,Y,Z), but has shape {mu_dvf.shape}")
        
        print(f"  Normalised input shapes: mu_dvf={mu_dvf.shape}, sigma_map={sigma_map.shape}")
        
        # Compute ell_map
        if ct_gradients is not None:
            ell_map = adaptive_ell_map(ct_gradients, base_ell=self.base_ell, adaptation_gain=self.adaptation_gain)
        else:
            ell_map = torch.tensor(list(self.base_ell), device=mu_dvf.device).view(3, 1, 1, 1)
        
        # Shape of the magnitude (Z, Y, X)
        # NOTE: mu_dvf is (3, X, Y, Z) after normalisation
        mag_shape = (mu_dvf.shape[3], mu_dvf.shape[2], mu_dvf.shape[1])  # (Z, Y, X)
        
        print(f"  mag_shape (Z, Y, X): {mag_shape}")
        
        # Initialise the incremental statistics
        dtype = np.float32
        running_mean = np.zeros(mag_shape, dtype=dtype)
        running_var = np.zeros(mag_shape, dtype=dtype)
        count = 0
        
        # Single-sample case
        if not self.training and n_samples == 1:
            mu_local = mu_dvf.cuda()
            sigma_local = sigma_map.cuda()
            
            if ct_gradients is not None:
                grads_local = ct_gradients.cuda()
                ell_map_local = adaptive_ell_map(grads_local, base_ell=self.base_ell, adaptation_gain=self.adaptation_gain)
            else:
                ell_map_local = ell_map.cuda() if isinstance(ell_map, torch.Tensor) else ell_map
            
            retry = 0
            while retry < 20:
                epsilon = torch.randn_like(mu_local)
                correlated = anisotropic_correlation(epsilon, ell_map_local, kernel_size=self.kernel_size)

                # FIX: guarantees K → I when ell → 0
                if isinstance(ell_map_local, torch.Tensor):
                    ell_min = ell_map_local.min()
                    if ell_min < 1e-3:  # threshold to be tuned (in voxels)
                        correlated = epsilon
                phi_k = mu_local + (sigma_local * self.variability_gain * correlated)
                phi_k_checked, f_rate = jacobian_check(phi_k)
                if phi_k_checked is not None:
                    return phi_k_checked
                retry += 1
            return mu_local # Fallback

        # Multiple samples
        _raw_gpus = torch.cuda.device_count()
        valid_gpus = [i for i in range(_raw_gpus) if i != 2] # Exclude the Quadro P620
        n_gpus = len(valid_gpus)
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"  Saving samples to: {output_dir}")
            ensemble_path = os.path.join(output_dir, "dvf_sampled_ensemble.npy")
            # Initialise an empty memmap for all the samples (magnitudes)
            ensemble_mmap = np.lib.format.open_memmap(
                ensemble_path, mode='w+', dtype=dtype, 
                shape=(n_samples, mag_shape[0], mag_shape[1], mag_shape[2])
            )
            # Initialise a memmap for the downsampled 3D vectors (for the quiver plot)
            ds = 8
            # mu_dvf is (3, X, Y, Z) -> viewer standardisation requires (3, Z_ds, Y_ds, X_ds)
            vec_shape_ds = (
                3,
                (mu_dvf.shape[3] + ds - 1) // ds, # Z
                (mu_dvf.shape[2] + ds - 1) // ds, # Y
                (mu_dvf.shape[1] + ds - 1) // ds  # X
            )
            vec_path = os.path.join(output_dir, "dvf_sampled_vectors_down.npy")
            try:
                vec_mmap = np.lib.format.open_memmap(
                    vec_path, mode='w+', dtype=dtype,
                    shape=(n_samples, vec_shape_ds[0], vec_shape_ds[1], vec_shape_ds[2], vec_shape_ds[3])
                )
            except OSError as e:
                if "[Errno 22]" in str(e):
                    print(f"  CRITICAL ERROR: Access denied to {vec_path}. The file might be locked by another process (e.g. the GUI Viewer).")
                    print("  Please close any open Viewer windows or switch the overlay to 'None' before re-running sampling.")
                raise e
            print(f"  Vec memmap shape: {vec_mmap.shape}")
        else:
            ensemble_mmap = None
            vec_mmap = None
        
        # Distribution over GPUs
        if n_gpus <= 1 or n_samples < n_gpus * 2:
            gpu_id = valid_gpus[0] if n_gpus > 0 else 0
            print(f"  Generating {n_samples} samples on a single GPU (GPU {gpu_id})...")
            
            mu_local = mu_dvf.cuda(gpu_id)
            sigma_local = sigma_map.cuda(gpu_id)
            
            if ct_gradients is not None:
                grads_local = ct_gradients.cuda(gpu_id)
                ell_map_local = adaptive_ell_map(grads_local, base_ell=self.base_ell, adaptation_gain=self.adaptation_gain)
            else:
                ell_map_local = ell_map.cuda(gpu_id) if isinstance(ell_map, torch.Tensor) else ell_map
            
            total_folding_rates = []
            total_retries = []
            
            # Compute the adaptive threshold: tolerate the original rate + 0.05%
            _, base_rate = jacobian_check(mu_local, threshold=1.0)
            adaptive_threshold = max(0.0001, base_rate + 0.0005)
            
            for k in range(n_samples):
                retry = 0
                local_gain = self.variability_gain
                while retry < 20:
                    epsilon = torch.randn_like(mu_local)
                    correlated = anisotropic_correlation(epsilon, ell_map_local, kernel_size=self.kernel_size)

                    # FIX: guarantees K → I when ell → 0
                    if isinstance(ell_map_local, torch.Tensor):
                        ell_min = ell_map_local.min()
                        if ell_min < 1e-3:  # threshold to be tuned (in voxels)
                            correlated = epsilon
                    
                    gain_tensor = torch.tensor(local_gain, dtype=mu_local.dtype, device=mu_local.device)
                    delta = sigma_local * gain_tensor * correlated
                    phi_k = mu_local + delta
                    
                    phi_k_checked, f_rate = jacobian_check(phi_k, threshold=adaptive_threshold)
                    if phi_k_checked is not None:
                        if retry > 0:
                            print(f"    GPU {gpu_id}: Sample {k} accepted after {retry} attempts (gain reduced to {local_gain:.3f}).")
                        break
                    
                    # DYNAMIC DECAY
                    local_gain *= 0.85
                    retry += 1
                
                total_folding_rates.append(f_rate)
                total_retries.append(retry)
                
                if phi_k_checked is None:
                    phi_k_checked = mu_local
                
                # Compute the magnitude and transpose
                sample_np = phi_k_checked.cpu().numpy().astype(dtype)
                mag_zyx = self._compute_magnitude_zyx(sample_np)
                
                # Save the individual sample (magnitude)
                if ensemble_mmap is not None:
                    ensemble_mmap[k] = mag_zyx
                    ensemble_mmap.flush()
                elif output_dir is not None:
                    sample_file = os.path.join(output_dir, f"dvf_sample_{k:04d}.npy")
                    np.save(sample_file, mag_zyx)
                
                # Save downsampled 3D vectors for the quiver plot
                if vec_mmap is not None:
                    ds = 8
                    # Standardise to (3, Z_ds, Y_ds, X_ds) for the viewer
                    vec_ds = np.transpose(sample_np[:, ::ds, ::ds, ::ds], (0, 3, 2, 1))
                    vec_mmap[k] = vec_ds
                    if k % 50 == 0:
                        vec_mmap.flush()
                
                # Update the incremental statistics
                count += 1
                delta_stat = mag_zyx - running_mean
                running_mean += delta_stat / count
                delta2 = mag_zyx - running_mean
                running_var += delta_stat * delta2
                
                if k % 10 == 0:
                    torch.cuda.empty_cache()
            
            std = np.sqrt(running_var / count) if count > 0 else np.zeros_like(running_mean)
            
            # Save mean and std
            if output_dir is not None:
                np.save(os.path.join(output_dir, "dvf_sampled_mean_mag.npy"), running_mean)
                np.save(os.path.join(output_dir, "dvf_sampled_std_mag.npy"), std)
            
            # Save the folding statistics (single-GPU path)
            if output_dir is not None:
                import json
                stats_path = os.path.join(output_dir, f"folding_stats_gpu_{gpu_id}.json")
                with open(stats_path, "w") as f:
                    json.dump({
                        "folding_rates": total_folding_rates,
                        "retries": total_retries,
                        "mean_folding": float(np.mean(total_folding_rates)),
                        "max_folding": float(np.max(total_folding_rates)),
                        "retry_rate": float(np.mean([1 if r > 0 else 0 for r in total_retries]))
                    }, f, indent=4)
            
            # Final flush of vec_mmap
            if vec_mmap is not None:
                vec_mmap.flush()
                print(f"  Vec memmap saved: {vec_mmap.shape}")
            
            return running_mean, std
        
        # Distribution over multiple GPUs
        samples_per_gpu = n_samples // n_gpus
        remainder = n_samples % n_gpus
        
        print(f"  Distributing {n_samples} samples over {n_gpus} valid GPUs:")
        for idx in range(n_gpus):
            gpu_id = valid_gpus[idx]
            n_for_this_gpu = samples_per_gpu + (1 if idx < remainder else 0)
            print(f"    GPU {gpu_id}: {n_for_this_gpu} samples")
        
        # Prepare ell_map on the CPU
        if isinstance(ell_map, torch.Tensor):
            ell_map_cpu = ell_map.cpu()
        else:
            ell_map_cpu = ell_map
        
        # Prepare tensors on the CPU
        mu_cpu = mu_dvf.cpu()
        sigma_cpu = sigma_map.cpu()
        grads_cpu = ct_gradients.cpu() if ct_gradients is not None else None
        
        # Start one thread per GPU
        import threading
        results_per_gpu = [None] * _raw_gpus
        
        def run_on_gpu(gpu_id, n_for_this_gpu, start_idx, init_mean, init_var, init_count):
            try:
                final_mean, final_var, final_count = self._generate_samples_on_gpu(
                    mu_cpu, sigma_cpu, grads_cpu,
                    n_for_this_gpu, gpu_id, ell_map_cpu,
                    output_dir, start_idx,
                    init_mean, init_var, init_count, dtype, ensemble_mmap, vec_mmap
                )
                results_per_gpu[gpu_id] = (final_mean, final_var, final_count)
            except Exception as e:
                print(f"  Error on GPU {gpu_id}: {e}")
                results_per_gpu[gpu_id] = None
        
        threads = []
        current_idx = 0
        for idx in range(n_gpus):
            gpu_id = valid_gpus[idx]
            n_for_this_gpu = samples_per_gpu + (1 if idx < remainder else 0)
            if n_for_this_gpu > 0:
                thread = threading.Thread(target=run_on_gpu, args=(gpu_id, n_for_this_gpu, current_idx,
                                                                   running_mean, running_var, count))
                thread.start()
                threads.append(thread)
                current_idx += n_for_this_gpu
        
        for thread in threads:
            thread.join()
        
        # Combine the results from the different GPUs (Welford merge)
        for idx in range(n_gpus):
            gpu_id = valid_gpus[idx]
            if results_per_gpu[gpu_id] is not None:
                mean_gpu, var_gpu, count_gpu = results_per_gpu[gpu_id]
                if count_gpu > 0:
                    total_count = count + count_gpu
                    delta = mean_gpu - running_mean
                    running_mean = running_mean + delta * count_gpu / total_count
                    running_var = running_var + var_gpu + delta * delta * count * count_gpu / total_count
                    count = total_count
        
        std = np.sqrt(running_var / count) if count > 0 else np.zeros_like(running_mean)
        
        # Save mean and std
        if output_dir is not None:
            np.save(os.path.join(output_dir, "dvf_sampled_mean_mag.npy"), running_mean)
            np.save(os.path.join(output_dir, "dvf_sampled_std_mag.npy"), std)
        
        # Final flush of vec_mmap
        if vec_mmap is not None:
            vec_mmap.flush()
            print(f"  Vec memmap saved: {vec_mmap.shape}")
        
        print(f"  Ensemble completed: {count} samples generated (memmap ready)")
        return running_mean, std
