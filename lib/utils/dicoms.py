import pydicom
import os
import numpy as np
import copy
import warnings
from skimage.draw import polygon2mask



ROI_COUNTOUR_SEQUENCE_INFO_TAG = "30060080"
ROI_CONTOUR_SEQUENCE_TAG = "30060040"
ROI_OBSERVATION_LABEL_TAG = "30060085"
ROI_NAME_TAG = "30060026"


def getPixelArray(dcm):
    array = np.transpose(dcm.pixel_array, axes=[2, 1, 0]).astype(float)
    if hasattr(dcm, "RescaleSlope"):
        array *= float(dcm.RescaleSlope)
    if hasattr(dcm, "RescaleIntercept"):
        array = array + float(dcm.RescaleIntercept)
    if hasattr(dcm, "DoseGridScaling"):
        array *= float(dcm.DoseGridScaling)
    return array


def getVoxelToWorldMatrix(dcm):
    voxelSpacing = getVoxelSpacing(dcm)
    imagePosition = np.array(dcm.ImagePositionPatient).astype(float)
    
    # RTDOSE Z-origin adjustment
    if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2': # RT Dose Storage
        if hasattr(dcm, "GridFrameOffsetVector") and len(dcm.GridFrameOffsetVector) > 0:
            offsets = np.array(dcm.GridFrameOffsetVector).astype(float)
            # IPP is the position of the first frame.
            # We want THE MINIMUM Z position for the Scan object logic
            z_positions = imagePosition[2] + offsets
            imagePosition[2] = np.min(z_positions)

    imageOrientation = getImageOrientationPatient(dcm)
    assert np.all(imageOrientation == np.identity(3))
    orientationMatrix = imageOrientation * voxelSpacing
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, :3] = orientationMatrix
    voxelToWorldMatrix[:3, 3] = imagePosition
    return voxelToWorldMatrix


def slicedToSingleDicom(slicePaths):
    pixelArrayList = []
    sliceCoordinates = []
    
    print(f"DEBUG: Scanning {len(slicePaths)} files...")
    
    # First, filter files that are actually DICOM image slices
    valid_slices = []
    for path in slicePaths:
        if os.path.isdir(path):
            continue
        try:
            dcm = pydicom.dcmread(path)
            # We want files with Image characteristics
            modality = getattr(dcm, "Modality", "UNKNOWN")
            series_uid = getattr(dcm, "SeriesInstanceUID", None)
            
            # Skip obvious RT objects that are not images
            if modality in ["RTDOSE", "RTSTRUCT", "RTPLAN", "RTTREATMENTRECORD"]:
                continue
                
            if series_uid and (hasattr(dcm, "pixel_array") or hasattr(dcm, "PixelData")):
                valid_slices.append({"path": path, "uid": series_uid, "modality": modality})
        except Exception as e:
            # print(f"DEBUG: Error reading {path}: {e}")
            continue
            
    if not valid_slices:
        print("DEBUG: No valid image slices found in initial scan.")
        raise ValueError("No valid DICOM image slices found in the provided paths.")
        
    # Group by series and pick the one with most slices
    series_counts = {}
    for s in valid_slices:
        series_counts[s["uid"]] = series_counts.get(s["uid"], 0) + 1
    
    best_series_uid = max(series_counts, key=series_counts.get)
    print(f"DEBUG: Selected Series {best_series_uid} with {series_counts[best_series_uid]} slices.")
    
    target_slice_paths = [s["path"] for s in valid_slices if s["uid"] == best_series_uid]
    
    # Use the first slice of the chosen series as base
    baseDicom = pydicom.dcmread(target_slice_paths[0])
    
    for path in target_slice_paths:
        dcmSlice = pydicom.dcmread(path)
        if isValidSlice(dcmSlice, baseDicom):
            try:
                pixelArrayList.append(
                    convertStoredValuesToArray(dcmSlice.pixel_array, dcmSlice)
                )
                # Some DICOMs use ImagePositionPatient[2] if SliceLocation is missing
                loc = getattr(dcmSlice, "SliceLocation", None)
                if loc is None and hasattr(dcmSlice, "ImagePositionPatient"):
                    loc = dcmSlice.ImagePositionPatient[2]
                
                if loc is not None:
                    sliceCoordinates.append(loc)
                else:
                    print(f"DEBUG: Warning: Slice at {path} has no location info. Skipping.")
            except Exception as e:
                print(f"DEBUG: Error processing slice {path}: {e}")
        else:
            warnings.warn("Skipping inconsistent slice at {}".format(path))

    if not sliceCoordinates:
        raise ValueError("No slices with valid coordinate information found for the selected series.")

    # Sort and check consistency
    sorted_pairs = sorted(zip(sliceCoordinates, pixelArrayList), key=lambda x: x[0])
    sliceCoordinates = [x[0] for x in sorted_pairs]
    pixelArrayList = [x[1] for x in sorted_pairs]
    
    pixelArray = np.stack(pixelArrayList, axis=0)
    
    # Calculate real slice spacing from locs
    if len(sliceCoordinates) > 1:
        real_spacing = np.mean(np.diff(sliceCoordinates))
        baseDicom.SliceThickness = abs(real_spacing)
        print(f"DEBUG: Detected real slice spacing: {baseDicom.SliceThickness:.4f} mm (from locs)")

    newPixelArrayPosition = getPixelArrayPosition(baseDicom, sliceCoordinates)
    print(f"DEBUG: Successfully loaded volume with shape {pixelArray.shape}")
    return createDicom(pixelArray, baseDicom, newPixelArrayPosition)


def isValidSlice(dcmSlice, baseDcm):
    # TODO: add PID check, datetime check
    if not hasSameSliceThickness(dcmSlice, baseDcm):
        return False
    if not hasSamePixelSpacing(dcmSlice, baseDcm):
        return False
    if not hasSameSliceOrientation(dcmSlice, baseDcm):
        return False
    return True


def hasSameSliceThickness(dcmSlice, baseDcm):
    return dcmSlice.SliceThickness == baseDcm.SliceThickness


def hasSamePixelSpacing(dcmSlice, baseDcm):
    return dcmSlice.PixelSpacing == baseDcm.PixelSpacing


def getVoxelSpacing(dcm):
    try:
        pixel_spacing = [float(s) for s in dcm.PixelSpacing] if hasattr(dcm, 'PixelSpacing') and dcm.PixelSpacing is not None else [1.0, 1.0]
        
        # RTDOSE specific Z-spacing
        slice_thickness = getattr(dcm, 'SliceThickness', 1.0)
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2': # RT Dose Storage
            if hasattr(dcm, "GridFrameOffsetVector") and len(dcm.GridFrameOffsetVector) > 1:
                # Calculate spacing from offset vector
                offsets = np.array(dcm.GridFrameOffsetVector).astype(float)
                slice_thickness = abs(offsets[1] - offsets[0])
        
        # Handle cases where attributes exist but are None
        if pixel_spacing[0] is None or pixel_spacing[1] is None:
            pixel_spacing = [1.0, 1.0]
        if slice_thickness is None or slice_thickness == 0:
            slice_thickness = 1.0
            
        # [FIX] Swap X/Y spacing. PixelSpacing is [Rows, Cols] -> Scan expects [X(Cols), Y(Rows), Z(Slices)]
        voxelSpacing = [pixel_spacing[1], pixel_spacing[0], slice_thickness]
        return np.array([float(spacing) for spacing in voxelSpacing])
    except Exception as e:
        print(f"DEBUG: Error extracting voxel spacing: {e}. Defaulting to [1.0, 1.0, 1.0]")
        return np.array([1.0, 1.0, 1.0])


def getImageOrientationPatient(dcm):
    # TODO: when going to non identity orientations, we have to check whether the vectors should be stacked horizontally or vertically
    orientation = dcm.ImageOrientationPatient
    xVec = np.array(orientation[:3])
    yVec = np.array(orientation[3:])
    zVec = np.cross(xVec, yVec)
    return np.stack([xVec, yVec, zVec])


def hasSameSliceOrientation(dcmSlice, baseDcm):
    return dcmSlice.ImageOrientationPatient == baseDcm.ImageOrientationPatient


def isValidSliceList(sortedSliceList, sliceThickness):
    sortedSliceList = np.array(sortedSliceList)
    diff = sortedSliceList[1:] - sortedSliceList[:-1]
    return np.allclose(diff, sliceThickness)


def getPixelArrayPosition(baseDicom, sliceCoordinates):
    pixelArrayPosition = baseDicom.ImagePositionPatient
    pixelArrayPosition[2] = min(sliceCoordinates)
    return pixelArrayPosition


def createDicom(newPixelArray, baseDicom, newPixelArrayPosition):
    baseDicom = copy.deepcopy(baseDicom)
    if hasattr(baseDicom, "InstanceNumber"):
        del baseDicom.InstanceNumber
    warnings.warn(
        "Dicoms are stored in zyx coordinates, please make sure the provided pixel array is as such"
    )
    pixelDataBytes = convertArrayToStoredValues(newPixelArray, baseDicom)
    baseDicom.NumberOfFrames, baseDicom.Rows, baseDicom.Columns = newPixelArray.shape
    baseDicom.PixelData = pixelDataBytes
    baseDicom.ImagePositionPatient = newPixelArrayPosition
    return baseDicom


def convertArrayToStoredValues(array, baseDicom):
    array = (array - baseDicom.RescaleIntercept) / (baseDicom.RescaleSlope)
    return convertToTypeWithWarning(array, baseDicom.pixel_array.dtype).tobytes()


def convertStoredValuesToArray(array, baseDicom):
    return array * (baseDicom.RescaleSlope) + baseDicom.RescaleIntercept


def convertToTypeWithWarning(array, targetType):
    convertedArray = convertToType(array, targetType)
    if not np.all(convertedArray == array):
        warnings.warn(
            "Conversion of array with type {} to {} changed some of its values".format(
                array.dtype, targetType
            )
        )
    return convertedArray


def convertToType(array, targetType):
    typeInfo = np.iinfo(targetType)
    array = np.clip(array, typeInfo.min, typeInfo.max)
    if isInteger(typeInfo):
        array = np.round(array)
    return array.astype(targetType)


def isInteger(typeInfo):
    return typeInfo.kind in "iub"

def convertContourSetToArray(dcm, voxelToWorldMatrix, shape):
    arrays = []
    for contour in getContourSequence(dcm):
        arrays.append(convertContourToArray(contour, voxelToWorldMatrix, shape))
    return np.stack(arrays, 0)


def convertContourSetToPointDict(dcm, voxelToWorldMatrix, shape):
    arrays = []
    for contour in getContourSequence(dcm):
        arrays.append(convertContourToPointDict(contour, voxelToWorldMatrix, shape))
    return arrays


def convertContourToArray(contour, voxelToWorldMatrix, shape):
    array = np.zeros(shape, dtype=np.int16)
    for contourSlice in contour.ContourSequence:
        points = np.array(contourSlice.ContourData).reshape(-1, 3)
        points = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
        newPoints = np.transpose(
            np.matmul(np.linalg.inv(voxelToWorldMatrix), np.transpose(points))
        )[:, :3]
        slice = round(newPoints[0, 2])
        if abs(slice - newPoints[0, 2]) > 10 ** -5:
            warnings.warn(
                f"Found a slice position at {newPoints[0,2]}, but set to {slice}"
            )

        array[:, :, slice] += polygon2mask(shape[:2], newPoints[:, :2])
    array = np.mod(array, 2)
    return array.astype(bool)


def convertContourToPointDict(contour, voxelToWorldMatrix, shape):
    pointDict = {}
    for contourSlice in contour.ContourSequence:
        points = np.array(contourSlice.ContourData).reshape(-1, 3)
        points = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
        newPoints = np.transpose(
            np.matmul(np.linalg.inv(voxelToWorldMatrix), np.transpose(points))
        )[:, :3]
        # [FIX] Use round() instead of int() to avoid half-slice indexing errors
        slice = int(round(newPoints[0, 2]))
        if abs(slice - newPoints[0, 2]) > 10 ** -5:
            warnings.warn(
                f"Found a slice position at {newPoints[0,2]}, but set to {slice}"
            )
        if slice in pointDict:
            pointDict[slice].append(newPoints[:, :2])
        else:
            pointDict[slice] = [newPoints[:, :2]]
    return pointDict


def getContourSetNames(dcm):
    contourDataSequence = getContourData(dcm)
    contourNames = []
    for (_, contourInfo, structureInfo) in contourDataSequence:
        if ROI_NAME_TAG in structureInfo:
            contourNames.append(structureInfo.ROIName)
        else:
            contourNames.append(contourInfo.ROIObservationLabel)
    return contourNames


def getContourSetTypes(dcm):
    contourInfoSequence = getContourInfoSequence(dcm)
    return [contourInfo.RTROIInterpretedType for contourInfo in contourInfoSequence]


def changeContourSetNames(dcm, nameConversionDict):
    contourInfoSequence = getContourInfoSequence(dcm)
    for contourInfo in contourInfoSequence:
        oldName = contourInfo.ROIObservationLabel
        if oldName in nameConversionDict:
            contourInfo.ROIObservationLabel = nameConversionDict[oldName]
    return dcm


def getContourSequence(dcm):
    return [x for (x, _, _) in getContourData(dcm)]


def getContourInfoSequence(dcm):
    return [x for (_, x, _) in getContourData(dcm)]


def getStructureInfoSequence(dcm):
    return [x for (_, _, x) in getContourData(dcm)]


def getContourData(dcm):
    return [
        (contourObject, contourInfo, structureInfo)
        for (contourObject, contourInfo, structureInfo) in zip(
            dcm.ROIContourSequence,
            dcm[ROI_COUNTOUR_SEQUENCE_INFO_TAG],
            dcm.StructureSetROISequence,
        )
        if ROI_CONTOUR_SEQUENCE_TAG in contourObject
        and (ROI_OBSERVATION_LABEL_TAG in contourInfo or ROI_NAME_TAG in structureInfo)
    ]

def export_rtdose(array_zyx, reference_dcm_path, output_path, geometry_ref_dcm=None, series_desc="", origin=None, spacing=None):
    """
    Exports a 3D volume as a single multi-frame RTDOSE file.
    array_zyx: (Z, Y, X) voxels in Gy.
    reference_dcm_path: Path to existing RTDOSE for metadata.
    geometry_ref_dcm: Path to a CT slice for updated geometry (IPP, PixelSpacing).
    origin: Explicit [X, Y, Z] physical origin for Frame 0 (overrides geometry_ref_dcm)
    spacing: Explicit [dX, dY, dZ] physical spacing (overrides geometry_ref_dcm)
    """
    import pydicom
    import datetime
    import copy

    ref = pydicom.dcmread(reference_dcm_path)
    ds = copy.deepcopy(ref)
    
    ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')
    ds.SeriesDescription = series_desc if series_desc else f"DIR - {getattr(ref, 'SeriesDescription', '')}"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    
    # Extract spacing from geometry ref if not provided
    spacing_z = 1.0
    if spacing is not None:
        ds.PixelSpacing = [spacing[0], spacing[1]]
        spacing_z = spacing[2]
    elif geometry_ref_dcm and os.path.exists(geometry_ref_dcm):
        g = pydicom.dcmread(geometry_ref_dcm, stop_before_pixels=True)
        ds.PixelSpacing = g.PixelSpacing
        # Attempt to get spacing from original DOSE
        if hasattr(ref, "GridFrameOffsetVector") and len(ref.GridFrameOffsetVector) > 1:
            spacing_z = abs(float(ref.GridFrameOffsetVector[1]) - float(ref.GridFrameOffsetVector[0]))
    
    # Extract origin if not provided
    if origin is not None:
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2])]
    elif geometry_ref_dcm and os.path.exists(geometry_ref_dcm):
        g = pydicom.dcmread(geometry_ref_dcm, stop_before_pixels=True)
        ds.ImagePositionPatient = g.ImagePositionPatient

    # RTDOSE stores pixel data as (Frames, Rows, Cols)
    max_val = array_zyx.max()
    scaling = max_val / 65535.0 if max_val > 0 else 1e-4
    ds.DoseGridScaling = float(scaling)
    
    pixel_data = (array_zyx / scaling).astype(np.uint16)
    ds.PixelData = pixel_data.tobytes()
    ds.Rows, ds.Columns = pixel_data.shape[1], pixel_data.shape[2]
    ds.NumberOfFrames = pixel_data.shape[0]
    
    ds.GridFrameOffsetVector = [float(i * spacing_z) for i in range(ds.NumberOfFrames)]
    
    # Ensure Frame of Reference matches CT if provided
    if geometry_ref_dcm:
        g = pydicom.dcmread(geometry_ref_dcm, stop_before_pixels=True)
        ds.FrameOfReferenceUID = getattr(g, "FrameOfReferenceUID", ds.FrameOfReferenceUID)

    ds.save_as(output_path)
    print(f"RTDOSE exported: {output_path} (Shape: {array_zyx.shape}, Scaling: {scaling:.6f})")

def listDirAbsolute(directory):
    """
    Returns a list of absolute paths for all files in a directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]
