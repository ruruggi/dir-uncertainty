"""
core/dvf_dicom.py — Reading/writing a DVF as DICOM Deformable Spatial Registration.

Reference storage object:
    SOP Class: Deformable Spatial Registration Storage
    UID:       1.2.840.10008.5.1.4.1.1.66.3
    Standard:  DICOM PS3.3 (IOD), introduced via Supplement 73 (NEMA, 2004).

Preserved conventions:
    The project reader utils/dicom_vf.readDicomVectorField interprets the
    VectorGridData blob as float32 with shape (Z, Y, X, 3) and components
    (dx, dy, dz) in mm — then reverses the 3 channels to obtain (dz, dy, dx).
    The writer below must produce the *same* on-disk layout.

Quality preservation:
    The DVF components are stored as **float32 IEEE-754 little-endian**
    (no quantisation): bit-exact roundtrip with the reader.
    VR used: OF (Other Float) or equivalent.

References:
    NEMA. DICOM PS3 Supplement 73: Spatial Registration Storage SOP Classes.
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.20.html

    Whittle JM, Tonks A et al. Quality assurance of deformable image
    registration in radiation oncology. Phys Med Biol 60:R1-R39, 2015.
"""

from __future__ import annotations

import os
import datetime
from typing import Optional, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

from utils.vector_field import VectorField
from utils.io_utils import isDicom, listDirAbsolute


# DSR SOP class
DSR_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.3"

# RegistrationTypeCodeSequence values (cf. DCM 125021 = identity)
REG_TYPE_CODE_IDENTITY = "125021"
REG_TYPE_CODE_DEFORMABLE = "125022"  # "Deformable Image Registration"


def _read_first_dicom(folder_or_file: str) -> Optional[pydicom.Dataset]:
    """Read the first valid .dcm to extract the Patient/Study/FoR UIDs."""
    if not folder_or_file or not os.path.exists(folder_or_file):
        return None
    if os.path.isfile(folder_or_file) and isDicom(folder_or_file):
        try:
            return pydicom.dcmread(folder_or_file, stop_before_pixels=True)
        except Exception:
            return None
    if os.path.isdir(folder_or_file):
        for f in sorted(listDirAbsolute(folder_or_file)):
            if not isDicom(f):
                continue
            try:
                return pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:
                continue
    return None


def _identity_matrix_str_4x4() -> list:
    """Identity 4x4 row-major flatten per FrameOfReferenceTransformationMatrix."""
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def write_dvf_dicom(
    vector_field: VectorField,
    output_path: str,
    fixed_dicom_dir: Optional[str] = None,
    moving_dicom_dir: Optional[str] = None,
    method_metadata: Optional[dict] = None,
) -> str:
    """
    Save a VectorField as DICOM Deformable Spatial Registration (Sup 73).

    Parameters
    ----------
    vector_field : VectorField
        Project convention: shape (3, X, Y, Z), channels [dz, dy, dx], units mm.
        The DVF grid is that of the fixed scan (consistent voxelToWorldMatrix).
    output_path : str
        Destination .dcm path.
    fixed_dicom_dir, moving_dicom_dir : str, optional
        DICOM (CT) folders from which to inherit PatientID, StudyInstanceUID,
        FrameOfReferenceUID. If None, new UIDs are generated (self-consistent
        file but not linkable to a patient study).
    method_metadata : dict, optional
        DIR method + parameters (e.g. from AntsSyNRegisterer.get_metadata()):
        stored in `ContentDescription` for traceability.

    Returns
    -------
    str : path of the saved file.
    """
    if not vector_field.inMM:
        raise ValueError(
            "VectorField not in mm. Call .toMM() before saving."
        )
    arr = vector_field.getPixelArray()
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"Expected VectorField (3,X,Y,Z), got shape {arr.shape}")

    # --- Grid geometry (in fixed coordinates) -------------------------------
    vtw = np.asarray(vector_field.getVoxelToWorldMatrix(), dtype=np.float64)
    spacing = np.diag(vtw)[:3]               # (sx, sy, sz)
    origin = vtw[:3, 3]                      # (px, py, pz) mm in patient
    X, Y, Z = arr.shape[1:]

    # --- Conversion of project channels → standard DICOM --------------------------
    # Project: (3, X, Y, Z) channels [dz, dy, dx]
    # DICOM:   raw float32 layout (Z, Y, X, 3) components [dx, dy, dz]
    arr_xyz_components = arr[::-1]                            # (3, X, Y, Z), [dx,dy,dz]
    arr_dicom_layout = np.transpose(arr_xyz_components, (3, 2, 1, 0))  # (Z,Y,X,3)
    vector_grid_bytes = np.ascontiguousarray(
        arr_dicom_layout, dtype="<f4"
    ).tobytes()

    # --- Source metadata -----------------------------------------------------
    fixed_dcm = _read_first_dicom(fixed_dicom_dir) if fixed_dicom_dir else None
    moving_dcm = _read_first_dicom(moving_dicom_dir) if moving_dicom_dir else None

    patient_id = getattr(fixed_dcm, "PatientID", None) or "UNKNOWN"
    patient_name = getattr(fixed_dcm, "PatientName", None) or "UNKNOWN"
    study_uid = getattr(fixed_dcm, "StudyInstanceUID", None) or generate_uid()
    fixed_for_uid = getattr(fixed_dcm, "FrameOfReferenceUID", None) or generate_uid()
    moving_for_uid = getattr(moving_dcm, "FrameOfReferenceUID", None) or fixed_for_uid

    # --- File meta -------------------------------------------------------------
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = DSR_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid(prefix="1.2.826.0.1.3680043.8.498.")
    file_meta.ImplementationVersionName = "DIRUNCv2"  # SH max 16 chars

    # --- Top level dataset -----------------------------------------------------
    ds = Dataset()
    ds.file_meta = file_meta

    now = datetime.datetime.now()
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.SOPClassUID = DSR_SOP_CLASS_UID
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "REG"
    ds.Manufacturer = "DIR_Uncertainty_v2"
    ds.SeriesDescription = "Deformable Spatial Registration"
    if method_metadata:
        method_str = ", ".join(f"{k}={v}" for k, v in method_metadata.items())
        # ContentDescription is LO (Long String, max 64 chars). We use SeriesDescription
        # only as a hint, and ContentLabel/ContentCreatorName for the method.
        ds.SeriesDescription = f"DSR ({method_metadata.get('method', 'unknown')})"
        ds.ContentDescription = method_str[:64]

    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = getattr(fixed_dcm, "PatientBirthDate", "") if fixed_dcm else ""
    ds.PatientSex = getattr(fixed_dcm, "PatientSex", "") if fixed_dcm else ""

    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""

    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.ContentLabel = "DSR_DVF"
    ds.ContentCreatorName = "DIR_Uncertainty_v2"

    # --- Reference: two items in the DeformableRegistrationSequence (identity + deformable) ---
    # The project reader looks for registrations with CodeValue != IDENTITY_FRAME_TYPECODE
    # (see utils/dicom_vf.py:82). We include both for compliance.
    ds.DeformableRegistrationSequence = Sequence([
        _build_identity_registration_item(fixed_for_uid),
        _build_deformable_registration_item(
            source_for_uid=moving_for_uid,
            origin=origin,
            spacing=spacing,
            grid_dims=(X, Y, Z),
            vector_grid_bytes=vector_grid_bytes,
        ),
    ])

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    pydicom.dcmwrite(output_path, ds, write_like_original=False)
    return output_path


def _build_identity_registration_item(for_uid: str) -> Dataset:
    """DICOM item for the fixed FoR (identity transform)."""
    item = Dataset()
    item.SourceFrameOfReferenceUID = for_uid
    code = Dataset()
    code.CodeValue = REG_TYPE_CODE_IDENTITY
    code.CodingSchemeDesignator = "DCM"
    code.CodeMeaning = "Image Content-based Alignment"
    item.RegistrationTypeCodeSequence = Sequence([code])
    matrix_seq_item = Dataset()
    matrix_seq_item.FrameOfReferenceTransformationMatrixType = "RIGID"
    matrix_seq_item.FrameOfReferenceTransformationMatrix = _identity_matrix_str_4x4()
    pre_def = Dataset()
    pre_def.MatrixSequence = Sequence([matrix_seq_item])
    item.MatrixRegistrationSequence = Sequence([pre_def])
    return item


def _build_deformable_registration_item(
    source_for_uid: str,
    origin: np.ndarray,
    spacing: np.ndarray,
    grid_dims: Tuple[int, int, int],
    vector_grid_bytes: bytes,
) -> Dataset:
    """DICOM item for the actual deformable registration."""
    item = Dataset()
    item.SourceFrameOfReferenceUID = source_for_uid

    code = Dataset()
    code.CodeValue = REG_TYPE_CODE_DEFORMABLE
    code.CodingSchemeDesignator = "DCM"
    code.CodeMeaning = "Deformable Image Registration"
    item.RegistrationTypeCodeSequence = Sequence([code])

    # PreDeformationMatrixRegistrationSequence: identity (no rigid prefix in Phase A).
    pre_item = Dataset()
    pre_item.FrameOfReferenceTransformationMatrixType = "RIGID"
    pre_item.FrameOfReferenceTransformationMatrix = _identity_matrix_str_4x4()
    item.PreDeformationMatrixRegistrationSequence = Sequence([pre_item])

    # PostDeformationMatrixRegistrationSequence: identity.
    post_item = Dataset()
    post_item.FrameOfReferenceTransformationMatrixType = "RIGID"
    post_item.FrameOfReferenceTransformationMatrix = _identity_matrix_str_4x4()
    item.PostDeformationMatrixRegistrationSequence = Sequence([post_item])

    # DeformableRegistrationGridSequence: the actual displacement field.
    grid = Dataset()
    grid.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2])]
    # Canonical axial-supine LPS orientation; the project reader does not
    # read ImageOrientationPatient but we set it for DICOM compliance.
    grid.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    grid.GridDimensions = [int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])]
    grid.GridResolution = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
    # VectorGridData: float32 little-endian, length X*Y*Z*3*4 bytes.
    # VR = "OF" (Other Float) for tag (0064,0009).
    grid.VectorGridData = vector_grid_bytes
    item.DeformableRegistrationGridSequence = Sequence([grid])

    return item


def read_dvf_dicom(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit wrapper over utils.dicom_vf.readDicomVectorField.
    Returns (vector_array (3,X,Y,Z) channels [dz,dy,dx], voxelToWorldMatrix).
    """
    from utils.dicom_vf import readDicomVectorField
    return readDicomVectorField(path)
