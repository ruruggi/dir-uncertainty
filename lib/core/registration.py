"""
core/registration.py — DIR engines for computing the initial DVF.

Implements the common API `BaseRegisterer.register(fixed, moving) -> VectorField`.
Implemented engines:
    - `AntsSyNRegisterer`  : ANTs SyN/SyNRA (gold standard, Avants 2008).
    - `ConvexAdamRegisterer`: MIND-SSC + coupled convex + Adam IO (Heinrich 2014;
                              Heinrich/Hansen 2022, Learn2Reg winner). GPU.

Project-internal conventions:
    - Scan.pixelArray has shape (C, X, Y, Z), C=1 for images, C=3 for VFs.
    - VectorField in project convention: shape (3, X, Y, Z), channels in order
      [dz, dy, dx] (cf. utils/dicom_vf.readDicomVectorField, where the DICOM
      (x,y,z) components are reversed on reading).
    - Displacements in mm (inMM=True) by default, conversion to voxels via toVoxels().

ANTs/ITK convention:
    - numpy array (X, Y, Z) in LPS physical space.
    - Warp (.nii.gz output): shape (X, Y, Z, 3), components (dx, dy, dz) in mm.

ConvexAdam convention:
    - Input array (H, W, D), any layout (here (X, Y, Z) taken from Scan).
    - Output `disp` (H, W, D, 3), components `[d_axis0, d_axis1, d_axis2]` in
      VOXEL units (summed with the identity grid `np.arange`).
    - Direction: pull-back fixed-grid -> moving-coords (same as ANTs).
"""

import os
import numpy as np
from typing import Optional, Callable

from utils.scan import Scan
from utils.vector_field import VectorField


class BaseRegisterer:
    """Common API for the DIR engines. The subclass implements register()."""

    name: str = "base"

    def register(
        self,
        fixed_scan: Scan,
        moving_scan: Scan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VectorField:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        """Parameters used, to be serialised in pass3_metadata.json or in the DICOM."""
        return {"method": self.name}


class AntsSyNRegisterer(BaseRegisterer):
    """
    ANTs Symmetric Normalization (SyN) — Avants et al. 2008.

    By default uses `type_of_transform='SyN'` (no rigid/affine prefix), so the
    .nii.gz produced is already the complete displacement field in the physical
    space of the fixed scan and no composition is needed. For images with gross
    misalignment 'SyNRA' or 'SyNCC' can be passed, but in that case the warp is
    only the elastic component and the affine stays in a separate .mat file;
    this case is documented but not automated.

    References:
        Avants BB, Epstein CL, Grossman M, Gee JC. Symmetric diffeomorphic
        image registration with cross-correlation: Evaluating automated
        labeling of elderly and neurodegenerative brain. Med Image Anal,
        12(1):26-41, 2008. DOI: 10.1016/j.media.2007.06.004

        Avants BB, Tustison NJ, Song G, Cook PA, Klein A, Gee JC. A
        reproducible evaluation of ANTs similarity metric performance in
        brain image registration. NeuroImage, 54(3):2033-2044, 2011.

        Tustison NJ et al. The ANTsX ecosystem for quantitative biological
        and medical imaging. Sci Rep 11:9068, 2021.
    """

    name = "ants_syn"

    def __init__(
        self,
        type_of_transform: str = "SyN",
        reg_iterations=(100, 70, 50, 20),
        syn_metric: str = "CC",
        syn_sampling: int = 4,
        flow_sigma: float = 3.0,
        total_sigma: float = 0.0,
        grad_step: float = 0.25,
        verbose: bool = False,
    ):
        self.type_of_transform = type_of_transform
        self.reg_iterations = tuple(reg_iterations)
        self.syn_metric = syn_metric
        self.syn_sampling = syn_sampling
        self.flow_sigma = flow_sigma
        self.total_sigma = total_sigma
        self.grad_step = grad_step
        self.verbose = verbose

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "type_of_transform": self.type_of_transform,
            "reg_iterations": list(self.reg_iterations),
            "syn_metric": self.syn_metric,
            "syn_sampling": self.syn_sampling,
            "flow_sigma": self.flow_sigma,
            "total_sigma": self.total_sigma,
            "grad_step": self.grad_step,
        }

    def register(
        self,
        fixed_scan: Scan,
        moving_scan: Scan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VectorField:
        try:
            import ants
        except ImportError as e:
            raise ImportError(
                "antspyx not installed. Run:\n"
                "    venv/Scripts/python.exe -m pip install antspyx"
            ) from e

        def _log(msg: str, frac: float = -1.0):
            if progress_callback is not None:
                progress_callback(msg, frac)

        # --- 1) Extract (X,Y,Z) arrays and geometry from fixed/moving -----------
        fixed_arr = self._scan_to_xyz(fixed_scan)
        moving_arr = self._scan_to_xyz(moving_scan)

        spacing_fixed = tuple(float(s) for s in fixed_scan.getVoxelSpacing())
        spacing_moving = tuple(float(s) for s in moving_scan.getVoxelSpacing())
        origin_fixed = tuple(float(o) for o in fixed_scan.getOrigin())
        origin_moving = tuple(float(o) for o in moving_scan.getOrigin())

        _log(f"Fixed shape {fixed_arr.shape} spacing {spacing_fixed}", 0.05)
        _log(f"Moving shape {moving_arr.shape} spacing {spacing_moving}", 0.10)

        # ANTs assumes canonical LPS orientation from the direction matrices; for
        # axially aligned clinical CTs the rotation is the identity.
        fixed = ants.from_numpy(
            fixed_arr.astype(np.float32),
            spacing=spacing_fixed,
            origin=origin_fixed,
        )
        moving = ants.from_numpy(
            moving_arr.astype(np.float32),
            spacing=spacing_moving,
            origin=origin_moving,
        )

        # --- 2) SyN registration ---------------------------------------------------
        _log(f"Starting ANTs {self.type_of_transform}...", 0.15)
        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=self.type_of_transform,
            reg_iterations=self.reg_iterations,
            syn_metric=self.syn_metric,
            syn_sampling=self.syn_sampling,
            flow_sigma=self.flow_sigma,
            total_sigma=self.total_sigma,
            grad_step=self.grad_step,
            verbose=self.verbose,
        )
        _log("Registration completed.", 0.85)

        # --- 3) Warp extraction -----------------------------------------------------
        # SyN-only returns a single .nii.gz warp (pure deformable).
        # SyNRA/SyNCC/SyNAggro return warp + GenericAffine.mat.
        # To obtain a single dense displacement field capturing both the
        # rigid+affine and the deformable part, we call apply_transforms
        # with compose=<path>: ANTs writes the composite to a .nii.gz and
        # returns the path. We read it directly, without having to handle
        # the composition manually.
        warp_paths = [t for t in result["fwdtransforms"] if t.endswith(".nii.gz")]
        affine_paths = [t for t in result["fwdtransforms"] if t.endswith(".mat")]
        if not warp_paths:
            raise RuntimeError(
                f"ANTs did not produce a deformable warp (.nii.gz). "
                f"fwdtransforms={result['fwdtransforms']}"
            )

        if affine_paths:
            # Compose affine + warp -> single .nii.gz
            # NB: ants.apply_transforms(compose=prefix) wants a path prefix
            # (existing directory + file-name prefix). ANTs writes
            # `<prefix>comptx.nii.gz` and returns the full path.
            import tempfile
            tmpdir = tempfile.mkdtemp(prefix="ants_comp_")
            try:
                prefix = os.path.join(tmpdir, "comp_")
                _log(
                    f"Composing affine ({len(affine_paths)}) + warp -> "
                    f"single displacement field...", 0.87,
                )
                composed_path = ants.apply_transforms(
                    fixed=fixed,
                    moving=fixed,
                    transformlist=result["fwdtransforms"],
                    compose=prefix,
                )
                if not (isinstance(composed_path, str) and os.path.isfile(composed_path)):
                    raise RuntimeError(
                        f"apply_transforms compose returned {composed_path!r}, "
                        "expected path to an existing .nii.gz."
                    )
                warp_img = ants.image_read(composed_path)
            finally:
                # Cleanup tmpdir
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            warp_img = ants.image_read(warp_paths[0])

        warp_xyz_dxyz = warp_img.numpy()  # shape (X, Y, Z, 3), components (dx,dy,dz) mm
        if warp_xyz_dxyz.ndim != 4 or warp_xyz_dxyz.shape[-1] != 3:
            raise RuntimeError(
                f"ANTs warp has unexpected shape {warp_xyz_dxyz.shape}, "
                "expected (X, Y, Z, 3)"
            )

        # --- 4) Conversion to the project convention --------------------------
        # ANTs:    (X, Y, Z, 3) with channels [dx, dy, dz]
        # Project: (3, X, Y, Z) with channels [dz, dy, dx]   (cf. dicom_vf.py)
        warp_3xyz_dxyz = np.transpose(warp_xyz_dxyz, (3, 0, 1, 2))  # (3, X, Y, Z), [dx,dy,dz]
        warp_project = warp_3xyz_dxyz[::-1].copy()                  # (3, X, Y, Z), [dz,dy,dx]

        # --- 5) Wrap in VectorField, fixed geometry --------------------------
        vf = VectorField(array=warp_project.astype(np.float32),
                         vtwMatrix=np.array(fixed_scan.getVoxelToWorldMatrix(), dtype=np.float64),
                         inMM=True)
        _log(f"DVF produced: shape {vf.getPixelArray().shape}, "
             f"range [{warp_project.min():.2f}, {warp_project.max():.2f}] mm", 0.95)

        return vf

    @staticmethod
    def _scan_to_xyz(scan: Scan) -> np.ndarray:
        """Extract the (X, Y, Z) float32 array from the Scan, handling the leading channel."""
        arr = scan.getPixelArray()
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(
                    f"Multi-channel Scan not supported for registration: "
                    f"shape {arr.shape}"
                )
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array after squeeze, got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)


class ConvexAdamRegisterer(BaseRegisterer):
    """
    ConvexAdam — Heinrich/Hansen 2022 (Learn2Reg winner).

    Two-stage pipeline without supervised training:
      1) MIND-SSC features (Heinrich 2012/2013) -> SSD correlation volume
         on a sparse grid (`grid_sp`, default 6) -> coupled convex optimisation
         for the initial solution (Heinrich/Hansen 2019).
      2) Adam instance optimisation with diffusion regularisation and
         B-spline smoothing on grid `grid_sp_adam` (default 2).

    Advantages: GPU-only, ~5-30 s/case at full resolution on GV100, robust to
    cross-modal pairs thanks to MIND-SSC (modality-invariant). No training required.

    References:
        Heinrich MP, Jenkinson M, Bhushan M, et al. MIND: Modality independent
        neighbourhood descriptor for multi-modal deformable registration.
        Med Image Anal 16(7):1423-1435, 2012. DOI: 10.1016/j.media.2012.05.008

        Heinrich MP, Jenkinson M, Papiez BW, et al. Towards realtime multimodal
        fusion for image-guided interventions using self-similarities. MICCAI
        2013, LNCS 8149:187-194. DOI: 10.1007/978-3-642-40811-3_24

        Heinrich MP, Hansen L. Voxelmorph++ Going beyond the cranial vault with
        keypoint supervision and multi-channel instance optimisation.
        WBIR 2022, LNCS 13386:85-95. DOI: 10.1007/978-3-031-11203-4_10

        Siebert H, Hansen L, Heinrich MP. Fast 3D registration with accurate
        optimisation and little learning for Learn2Reg 2021. MICCAI WBIR 2022.

    Implementation notes:
        - ConvexAdam output is in VOXEL units, with channels ordered like the
          axes of the input array. We convert to mm by multiplying by the
          per-axis spacing and reverse the channel order to the project
          convention [dz, dy, dx].
        - Only `fixed_scan.spacing` is used for the conversion: the moving
          image must first have been brought onto the fixed grid (resampling);
          the caller (run_pair.py / GUI) guarantees this precondition.
    """

    name = "convex_adam"

    def __init__(
        self,
        mind_r: int = 1,
        mind_d: int = 2,
        lambda_weight: float = 1.25,
        grid_sp: int = 6,
        disp_hw: int = 4,
        selected_niter: int = 80,
        selected_smooth: int = 0,
        grid_sp_adam: int = 2,
        ic: bool = True,
        device: str = "cuda:0",
        verbose: bool = False,
    ):
        self.mind_r = mind_r
        self.mind_d = mind_d
        self.lambda_weight = lambda_weight
        self.grid_sp = grid_sp
        self.disp_hw = disp_hw
        self.selected_niter = selected_niter
        self.selected_smooth = selected_smooth
        self.grid_sp_adam = grid_sp_adam
        self.ic = ic
        self.device = device
        self.verbose = verbose

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "mind_r": self.mind_r,
            "mind_d": self.mind_d,
            "lambda_weight": self.lambda_weight,
            "grid_sp": self.grid_sp,
            "disp_hw": self.disp_hw,
            "selected_niter": self.selected_niter,
            "selected_smooth": self.selected_smooth,
            "grid_sp_adam": self.grid_sp_adam,
            "ic": self.ic,
            "device": self.device,
        }

    def register(
        self,
        fixed_scan: Scan,
        moving_scan: Scan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VectorField:
        try:
            import torch
            from convexAdam.convex_adam_MIND import convex_adam_pt
        except ImportError as e:
            raise ImportError(
                "convexAdam not installed. Run:\n"
                "    venv/Scripts/python.exe -m pip install convexAdam"
            ) from e

        def _log(msg: str, frac: float = -1.0):
            if progress_callback is not None:
                progress_callback(msg, frac)

        # --- 1) Extract (X,Y,Z) arrays and geometry from fixed/moving -----------
        fixed_arr = AntsSyNRegisterer._scan_to_xyz(fixed_scan)
        moving_arr = AntsSyNRegisterer._scan_to_xyz(moving_scan)

        if fixed_arr.shape != moving_arr.shape:
            raise ValueError(
                f"ConvexAdam requires fixed and moving on the same grid: "
                f"fixed {fixed_arr.shape} != moving {moving_arr.shape}. "
                "First resample the moving image onto the fixed grid."
            )

        spacing_fixed = tuple(float(s) for s in fixed_scan.getVoxelSpacing())
        _log(f"Fixed/moving shape {fixed_arr.shape} spacing {spacing_fixed}", 0.05)

        # --- 2) Device selection ----------------------------------------------------
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            _log("CUDA not available, falling back to CPU.", 0.08)
            device = torch.device("cpu")
        else:
            device = torch.device(self.device)
        _log(f"Device: {device}", 0.10)

        # --- 3) Run ConvexAdam ----------------------------------------------
        _log("Starting ConvexAdam (MIND-SSC + coupled convex + Adam IO)...", 0.15)
        disp_voxel = convex_adam_pt(
            img_fixed=fixed_arr,
            img_moving=moving_arr,
            mind_r=self.mind_r,
            mind_d=self.mind_d,
            lambda_weight=self.lambda_weight,
            grid_sp=self.grid_sp,
            disp_hw=self.disp_hw,
            selected_niter=self.selected_niter,
            selected_smooth=self.selected_smooth,
            grid_sp_adam=self.grid_sp_adam,
            ic=self.ic,
            use_mask=False,
            device=device,
            verbose=self.verbose,
        )
        _log("Registration completed.", 0.85)

        if disp_voxel.ndim != 4 or disp_voxel.shape[-1] != 3 or \
                disp_voxel.shape[:3] != fixed_arr.shape:
            raise RuntimeError(
                f"Unexpected ConvexAdam disp shape {disp_voxel.shape}, "
                f"expected ({fixed_arr.shape}, 3)."
            )

        # --- 4) Voxel -> mm conversion --------------------------------------------
        # The k-th component is the displacement along the k-th axis of the
        # input volume (checked in a smoke test: a 5-voxel shift along axis 0
        # is recovered in disp[..., 0]).
        sx, sy, sz = spacing_fixed
        disp_mm_dxyz = np.empty_like(disp_voxel, dtype=np.float32)
        disp_mm_dxyz[..., 0] = disp_voxel[..., 0] * sx
        disp_mm_dxyz[..., 1] = disp_voxel[..., 1] * sy
        disp_mm_dxyz[..., 2] = disp_voxel[..., 2] * sz

        # --- 5) Conversion to the project convention --------------------------
        # ConvexAdam: (X, Y, Z, 3) with channels [dx, dy, dz]
        # Project:    (3, X, Y, Z) with channels [dz, dy, dx]
        warp_3xyz_dxyz = np.transpose(disp_mm_dxyz, (3, 0, 1, 2))  # (3, X, Y, Z), [dx,dy,dz]
        warp_project = warp_3xyz_dxyz[::-1].copy()                  # (3, X, Y, Z), [dz,dy,dx]

        # --- 6) Wrap in VectorField ------------------------------------------------
        vf = VectorField(
            array=warp_project.astype(np.float32),
            vtwMatrix=np.array(fixed_scan.getVoxelToWorldMatrix(), dtype=np.float64),
            inMM=True,
        )
        _log(
            f"DVF produced: shape {vf.getPixelArray().shape}, "
            f"range [{warp_project.min():.2f}, {warp_project.max():.2f}] mm",
            0.95,
        )

        return vf


# ----------------------------------------------------------------------
# Helpers in-memory ITK image construction (uniGradICON + Elastix)
# ----------------------------------------------------------------------
def _numpy_xyz_to_itk(arr_xyz: np.ndarray, spacing, origin):
    """Build an in-memory itk.Image from a numpy array (X, Y, Z).

    ITK numpy convention: an image of size (X, Y, Z) corresponds to a numpy
    ndarray of shape (Z, Y, X). We transpose + .copy() for Python ownership and
    contiguity. Spacing and origin in ITK convention (X, Y, Z) world (mm).
    """
    import itk
    arr_zyx = np.ascontiguousarray(arr_xyz.transpose(2, 1, 0).astype(np.float32, copy=True))
    img = itk.image_view_from_array(arr_zyx)
    img.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])
    img.SetOrigin([float(origin[0]), float(origin[1]), float(origin[2])])
    return img, arr_zyx  # also return arr_zyx to avoid gc of the buffer view


def _itk_disp_to_project_vf(dvf_itk, vtw_matrix: np.ndarray) -> VectorField:
    """Convert an itk.Image[Vector[float, 3], 3] to a project-convention VectorField.

    Pipeline:
        ITK output (Z, Y, X, 3) [dx, dy, dz] mm world ->
        explicit copy (no view, avoids SIGSEGV after gc) ->
        transpose to (X, Y, Z, 3) ->
        permute to (3, X, Y, Z) [dx, dy, dz] ->
        reverse channels to [dz, dy, dx] = project convention
    """
    import itk
    # EXPLICIT COPY: array_view_from_image returns a view into the C++ buffer
    # of dvf_itk; as soon as dvf_itk goes out of scope the buffer is freed and
    # any access to the view causes a SIGSEGV.
    dvf_zyxv = np.array(itk.array_view_from_image(dvf_itk),
                         dtype=np.float32, copy=True)
    if dvf_zyxv.ndim != 4 or dvf_zyxv.shape[-1] != 3:
        raise ValueError(f"Expected (Z, Y, X, 3), got {dvf_zyxv.shape}")
    dvf_xyzv = np.ascontiguousarray(np.transpose(dvf_zyxv, (2, 1, 0, 3)))
    warp_3xyz_dxyz = np.transpose(dvf_xyzv, (3, 0, 1, 2))   # (3, X, Y, Z), [dx,dy,dz]
    warp_project = warp_3xyz_dxyz[::-1].copy()              # [dz,dy,dx]
    return VectorField(array=warp_project.astype(np.float32),
                        vtwMatrix=vtw_matrix, inMM=True)


class UniGradICONRegisterer(BaseRegisterer):
    """
    uniGradICON — Tian et al. 2024 MICCAI.

    Pretrained DL foundation model for medical image registration, zero-shot
    across anatomies / modalities. The model works internally at 175^3
    and produces an itk.CompositeTransform that incorporates the rescaling. The
    dense DVF is obtained via itk.transform_to_displacement_field_filter on the
    fixed grid.

    Advantages:
        - GPU-accelerated (~90 s/case 512^3 on GV100), 3-4x faster than
          CPU Elastix and 10x faster than cluster SyNRA
        - Robust to extreme misalignment: in the worst LIM cases (Liver case19,
          Abdomen case16) it does not diverge where SyNRA with paper-default
          settings produced TRE > 100 mm
        - CT-specific loss (HU clamp [-1000, 1000], shift+scale [0, 1])

    Limitations:
        - Requires ~80 MB of weights downloaded on first run (urlretrieve from
          GitHub releases, cached in <project>/network_weights/unigradicon1.0/)
        - Does not rescue the residual rigid Procrustes offset in LIM cases
          (Δ vs rigid typically +3 mm); adds a small overhead but does not degrade

    References:
        Tian L, Greer H, Vialard FX, Kwitt R, Estepar RSJ, Rushmore RJ,
        Makris N, Bouix S, Niethammer M. uniGradICON: A Foundation Model
        for Medical Image Registration. MICCAI 2024, LNCS 15006, Springer.
        DOI: 10.1007/978-3-031-72069-7_70

        Tian L, Greer H, et al. GradICON: Approximate Diffeomorphisms via
        Gradient Inverse Consistency. CVPR 2023.
    """

    name = "unigradicon"

    def __init__(
        self,
        model_name: str = "unigradicon",
        finetune_steps: int = 50,
        io_sim: str = "lncc",
        weights_path: Optional[str] = None,
        device: str = "cuda:0",
        verbose: bool = False,
    ):
        if model_name not in ("unigradicon", "multigradicon"):
            raise ValueError(f"model_name '{model_name}' not recognised. "
                             "Use 'unigradicon' or 'multigradicon'.")
        if io_sim not in ("lncc", "lncc2", "mind"):
            raise ValueError(f"io_sim '{io_sim}' not recognised.")
        self.model_name = model_name
        self.finetune_steps = int(finetune_steps)
        self.io_sim = io_sim
        self.weights_path = weights_path
        self.device = device
        self.verbose = verbose

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "model_name": self.model_name,
            "finetune_steps": self.finetune_steps,
            "io_sim": self.io_sim,
            "weights_path": self.weights_path,
            "device": self.device,
            "internal_shape": "175x175x175",
        }

    def register(
        self,
        fixed_scan: Scan,
        moving_scan: Scan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VectorField:
        try:
            import itk
            import unigradicon as ug
            import icon_registration.itk_wrapper as iw
        except ImportError as e:
            raise ImportError(
                "unigradicon not installed. Run:\n"
                "    venv/Scripts/python.exe -m pip install unigradicon"
            ) from e

        def _log(msg: str, frac: float = -1.0):
            if progress_callback is not None:
                progress_callback(msg, frac)

        fixed_arr = AntsSyNRegisterer._scan_to_xyz(fixed_scan)
        moving_arr = AntsSyNRegisterer._scan_to_xyz(moving_scan)
        if fixed_arr.shape != moving_arr.shape:
            raise ValueError(
                f"uniGradICON requires fixed and moving on the same grid: "
                f"fixed {fixed_arr.shape} != moving {moving_arr.shape}. "
                "First resample the moving image onto the fixed grid."
            )

        spacing = tuple(float(s) for s in fixed_scan.getVoxelSpacing())
        origin = tuple(float(o) for o in fixed_scan.getOrigin())
        _log(f"Fixed/moving shape {fixed_arr.shape} spacing {spacing}", 0.05)

        # itk.Image in-memory (no temp file)
        fixed_itk, _f_buf = _numpy_xyz_to_itk(fixed_arr, spacing, origin)
        moving_itk, _m_buf = _numpy_xyz_to_itk(moving_arr, spacing, origin)

        # Weights: default cache in <project>/network_weights/unigradicon1.0/
        weights_path = self.weights_path
        if weights_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_dir = os.path.join(
                project_root, "network_weights", f"{self.model_name}1.0")
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, "Step_2_final.trch")
            if not os.path.exists(weights_path):
                _log(f"Downloading weights {self.model_name} -> {weights_path}", 0.10)
                import urllib.request
                url = (f"https://github.com/uncbiag/uniGradICON/releases/download/"
                       f"{self.model_name}_weights/Step_2_final.trch")
                urllib.request.urlretrieve(url, weights_path)

        _log(f"Loading model {self.model_name}...", 0.15)
        if self.model_name == "unigradicon":
            net = ug.get_unigradicon(loss_fn=ug.make_sim(self.io_sim),
                                      weights_location=weights_path)
        else:
            net = ug.get_multigradicon(loss_fn=ug.make_sim(self.io_sim),
                                        weights_location=weights_path)

        _log(f"Starting uniGradICON (finetune {self.finetune_steps} iter)...", 0.20)
        # NB: register_pair(net, image_A=moving, image_B=fixed) by uniGradICON
        # convention; phi_AB is the fixed->moving map used by resample.
        phi_AB, _phi_BA = iw.register_pair(
            net,
            ug.preprocess(moving_itk, modality="ct"),
            ug.preprocess(fixed_itk, modality="ct"),
            finetune_steps=self.finetune_steps,
        )
        _log("Registration completed.", 0.85)

        # Dense DVF on the fixed grid
        dvf_itk = itk.transform_to_displacement_field_filter(
            phi_AB, reference_image=fixed_itk, use_reference_image=True,
        )

        vtw = np.array(fixed_scan.getVoxelToWorldMatrix(), dtype=np.float64)
        vf = _itk_disp_to_project_vf(dvf_itk, vtw)
        arr = vf.getPixelArray()
        _log(f"DVF produced: shape {arr.shape}, "
             f"range [{arr.min():.2f}, {arr.max():.2f}] mm", 0.95)
        return vf


class ElastixRegisterer(BaseRegisterer):
    """
    Elastix — Klein et al. 2010 IEEE TMI.

    Parametric multi-resolution B-spline DIR. Default: parameter map "bspline"
    (AdvancedMattesMutualInformation, 3 resolutions, 256-bin histogram).
    The canonical parameter files from the SlicerElastix database (Par0011
    lung CT, Par0049 abdominal CT) can be used via parameter_files=.

    Pipeline:
        1. Parameter object: default bspline or file-based (multi-stage)
        2. itk.elastix_registration_method(fixed, moving, parameter_object=po)
        3. itk.transformix_deformation_field(moving, result_params) -> DVF
        4. Wrap in VectorField project (3, X, Y, Z) [dz, dy, dx] mm

    References:
        Klein S, Staring M, Murphy K, Viergever MA, Pluim JPW. elastix:
        A Toolbox for Intensity-Based Medical Image Registration. IEEE
        Trans Med Imaging 29(1):196-205, 2010.
        DOI: 10.1109/TMI.2009.2035616

        Shamonin DP, Bron EE, Lelieveldt BPF, Smits M, Klein S, Staring M.
        Fast Parallel Image Registration on CPU and GPU for Diagnostic
        Classification of Alzheimer's Disease. Front Neuroinform 7:50, 2014.
        DOI: 10.3389/fninf.2013.00050

    Implementation notes:
        - Procrustes prealignment already applied by the caller (run_pair.py /
          GUI) -> rigid/affine skipped in elastix, bspline only.
        - Slow on CPU (~5-7 min/case 512^3); vs GPU-based uniGradICON ~90 s.
        - Output deformation field via Transformix on the fixed grid.
    """

    name = "elastix"

    def __init__(
        self,
        parameter_files: Optional[list] = None,
        log_to_console: bool = False,
        verbose: bool = False,
    ):
        self.parameter_files = list(parameter_files) if parameter_files else None
        self.log_to_console = log_to_console
        self.verbose = verbose

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "parameter_files": self.parameter_files or ["default_bspline"],
            "log_to_console": self.log_to_console,
        }

    def register(
        self,
        fixed_scan: Scan,
        moving_scan: Scan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VectorField:
        try:
            import itk
        except ImportError as e:
            raise ImportError(
                "itk-elastix not installed. Run:\n"
                "    venv/Scripts/python.exe -m pip install itk-elastix"
            ) from e

        def _log(msg: str, frac: float = -1.0):
            if progress_callback is not None:
                progress_callback(msg, frac)

        fixed_arr = AntsSyNRegisterer._scan_to_xyz(fixed_scan)
        moving_arr = AntsSyNRegisterer._scan_to_xyz(moving_scan)
        if fixed_arr.shape != moving_arr.shape:
            raise ValueError(
                f"Elastix accepts different shapes, but for consistency with the "
                f"pipeline (Procrustes prealign) we require the same grid: "
                f"fixed {fixed_arr.shape} != moving {moving_arr.shape}.")

        spacing = tuple(float(s) for s in fixed_scan.getVoxelSpacing())
        origin = tuple(float(o) for o in fixed_scan.getOrigin())
        _log(f"Fixed/moving shape {fixed_arr.shape} spacing {spacing}", 0.05)

        fixed_itk, _f_buf = _numpy_xyz_to_itk(fixed_arr, spacing, origin)
        moving_itk, _m_buf = _numpy_xyz_to_itk(moving_arr, spacing, origin)

        # Parameter object
        po = itk.ParameterObject.New()
        if self.parameter_files:
            for pf in self.parameter_files:
                po.AddParameterFile(pf)
            _log(f"Parameter files: {self.parameter_files}", 0.10)
        else:
            po.AddParameterMap(po.GetDefaultParameterMap("bspline"))
            _log("Parameter map: bspline default (Procrustes-prealigned input)", 0.10)
        po.SetParameter("WriteResultImage", "false")
        po.SetParameter("WriteIterationInfo", "false")

        _log("Starting Elastix...", 0.15)
        try:
            _, result_params = itk.elastix_registration_method(
                fixed_itk, moving_itk, parameter_object=po,
                log_to_console=self.log_to_console,
            )
        except TypeError:
            _, result_params = itk.elastix_registration_method(
                fixed_itk, moving_itk, parameter_object=po,
            )
        _log("Registration completed.", 0.85)

        # Dense DVF via Transformix (returns ONLY the deformation field;
        # cf. itk-elastix 0.25.x ElastixPython.py:233)
        try:
            dvf_itk = itk.transformix_deformation_field(
                moving_itk, result_params, log_to_console=self.log_to_console,
            )
        except TypeError:
            dvf_itk = itk.transformix_deformation_field(moving_itk, result_params)

        vtw = np.array(fixed_scan.getVoxelToWorldMatrix(), dtype=np.float64)
        vf = _itk_disp_to_project_vf(dvf_itk, vtw)
        arr = vf.getPixelArray()
        _log(f"DVF produced: shape {arr.shape}, "
             f"range [{arr.min():.2f}, {arr.max():.2f}] mm", 0.95)
        return vf
