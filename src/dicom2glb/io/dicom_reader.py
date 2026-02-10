"""DICOM reader: load directory, group by series, detect input type, assemble volumes."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from dicom2glb.core.types import SeriesInfo
from dicom2glb.core.volume import DicomVolume, TemporalSequence

logger = logging.getLogger(__name__)


class InputType(Enum):
    SINGLE_SLICE = "single"
    VOLUME = "volume"
    TEMPORAL = "temporal"


def load_dicom_directory(
    input_path: Path,
    series_uid: str | None = None,
) -> tuple[InputType, DicomVolume | TemporalSequence]:
    """Load DICOM data from a file or directory.

    Returns the detected input type and the corresponding data structure.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return _load_single_file(input_path)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # Scan all DICOM files
    dcm_files = _scan_dicom_files(input_path)
    if not dcm_files:
        raise ValueError(f"No valid DICOM files found in {input_path}")

    # Group by series
    series_groups = _group_by_series(dcm_files)

    # Select series
    if series_uid:
        selected_uid = _match_series_uid(series_groups, series_uid)
    else:
        selected_uid = _select_best_series(series_groups)

    datasets = series_groups[selected_uid]
    logger.info(
        f"Selected series {selected_uid} with {len(datasets)} files"
    )

    # Check for multi-frame DICOM files (e.g. ultrasound cine clips)
    multiframe = _detect_multiframe(datasets)
    if multiframe is not None:
        return multiframe

    # Detect temporal data (separate files per time point)
    temporal_groups = _detect_temporal_frames(datasets)

    if temporal_groups and len(temporal_groups) > 1:
        return _build_temporal_sequence(temporal_groups, selected_uid)
    else:
        return InputType.VOLUME, _build_volume(datasets, selected_uid)


def list_series(input_path: Path) -> list[dict]:
    """List all DICOM series found in a directory."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"Not a directory: {input_path}")

    dcm_files = _scan_dicom_files(input_path)
    series_groups = _group_by_series(dcm_files)

    result = []
    for uid, datasets in series_groups.items():
        ds = datasets[0]
        result.append(
            {
                "series_uid": uid,
                "modality": getattr(ds, "Modality", "Unknown"),
                "description": getattr(ds, "SeriesDescription", ""),
                "slice_count": len(datasets),
            }
        )
    return result


def analyze_series(input_path: Path) -> list[SeriesInfo]:
    """Analyze all DICOM series in a directory with classification and recommendations.

    Returns a sorted list of SeriesInfo, with the most clinically relevant series first.
    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"Not a directory: {input_path}")

    dcm_files = _scan_dicom_files(input_path)
    if not dcm_files:
        raise ValueError(f"No valid DICOM files found in {input_path}")

    series_groups = _group_by_series(dcm_files)

    result = []
    for uid, datasets in series_groups.items():
        info = _classify_series(uid, datasets)
        result.append(info)

    result.sort(key=_series_sort_key)
    return result


def _classify_series(uid: str, datasets: list[pydicom.Dataset]) -> SeriesInfo:
    """Classify a DICOM series by inspecting its datasets."""
    ds = datasets[0]
    modality = getattr(ds, "Modality", "Unknown")
    description = getattr(ds, "SeriesDescription", "")
    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))

    # Check for multi-frame files
    multiframe_ds = [
        d for d in datasets
        if hasattr(d, "NumberOfFrames") and int(d.NumberOfFrames) > 1
    ]

    if multiframe_ds:
        best = max(multiframe_ds, key=lambda d: int(d.NumberOfFrames))
        n_frames = int(best.NumberOfFrames)
        mf_rows = int(getattr(best, "Rows", 0))
        mf_cols = int(getattr(best, "Columns", 0))

        if _has_spatial_frame_info(best):
            data_type = "3D+T volume"
            detail = f"{n_frames} frames"
            dimensions = f"{mf_rows}x{mf_cols}"
            recommended_output = "animated 3D mesh"
        else:
            data_type = "2D cine"
            detail = f"{n_frames} frames"
            dimensions = f"{mf_rows}x{mf_cols}"
            recommended_output = "textured plane"

        method = _recommend_method(modality, data_type)
        return SeriesInfo(
            series_uid=uid,
            modality=modality,
            description=description,
            file_count=len(datasets),
            data_type=data_type,
            detail=detail,
            dimensions=dimensions,
            recommended_method=method,
            recommended_output=recommended_output,
            is_multiframe=True,
            number_of_frames=n_frames,
        )

    # Check for temporal tags across separate files
    has_temporal = any(
        hasattr(d, "TemporalPositionIdentifier") or hasattr(d, "TemporalPositionIndex")
        for d in datasets
    )
    if has_temporal:
        temporal_positions = set()
        for d in datasets:
            tp = getattr(d, "TemporalPositionIdentifier", None) or getattr(d, "TemporalPositionIndex", None)
            if tp is not None:
                temporal_positions.add(int(tp))
        if len(temporal_positions) > 1:
            data_type = "3D+T volume"
            slices_per_frame = len(datasets) // len(temporal_positions)
            detail = f"{len(temporal_positions)} frames, {slices_per_frame} slices each"
            dimensions = f"{rows}x{cols}x{slices_per_frame}"
            method = _recommend_method(modality, data_type)
            return SeriesInfo(
                series_uid=uid,
                modality=modality,
                description=description,
                file_count=len(datasets),
                data_type=data_type,
                detail=detail,
                dimensions=dimensions,
                recommended_method=method,
                recommended_output="animated 3D mesh",
            )

    # Single file, single slice
    if len(datasets) == 1:
        data_type = "still image"
        detail = "1 file"
        dimensions = f"{rows}x{cols}"
        method = _recommend_method(modality, data_type)
        return SeriesInfo(
            series_uid=uid,
            modality=modality,
            description=description,
            file_count=1,
            data_type=data_type,
            detail=detail,
            dimensions=dimensions,
            recommended_method=method,
            recommended_output="textured plane",
        )

    # Multiple files → 3D volume
    data_type = "3D volume"
    detail = f"{len(datasets)} slices"
    dimensions = f"{rows}x{cols}x{len(datasets)}"
    method = _recommend_method(modality, data_type)
    return SeriesInfo(
        series_uid=uid,
        modality=modality,
        description=description,
        file_count=len(datasets),
        data_type=data_type,
        detail=detail,
        dimensions=dimensions,
        recommended_method=method,
        recommended_output="3D mesh",
    )


def _has_spatial_frame_info(ds: pydicom.Dataset) -> bool:
    """Check if a multi-frame dataset has distinct spatial positions per frame."""
    seq = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not seq:
        return False

    z_positions = set()
    for frame_item in seq:
        plane_seq = getattr(frame_item, "PlanePositionSequence", None)
        if plane_seq:
            pos = getattr(plane_seq[0], "ImagePositionPatient", None)
            if pos:
                z_positions.add(float(pos[2]))

    return len(z_positions) > 1


def _recommend_method(modality: str, data_type: str) -> str:
    """Recommend a conversion method based on modality and data type."""
    if data_type in ("still image", "2D cine"):
        return "classical"
    if modality == "CT":
        return "marching-cubes"
    if modality == "MR":
        return "marching-cubes"
    return "classical"


def _series_sort_key(info: SeriesInfo) -> tuple:
    """Sort key for series: 3D+T > 3D volume > 2D cine > still, then by count desc."""
    type_priority = {
        "3D+T volume": 0,
        "3D volume": 1,
        "2D cine": 2,
        "still image": 3,
    }
    priority = type_priority.get(info.data_type, 4)
    count = info.number_of_frames if info.is_multiframe else info.file_count
    return (priority, -count)


def _scan_dicom_files(directory: Path) -> list[pydicom.Dataset]:
    """Scan directory recursively for valid DICOM files."""
    datasets = []
    for path in directory.rglob("*"):
        if path.is_file():
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=False)
                ds.filename = str(path)
                datasets.append(ds)
            except (InvalidDicomError, Exception):
                continue
    return datasets


def _group_by_series(
    datasets: list[pydicom.Dataset],
) -> dict[str, list[pydicom.Dataset]]:
    """Group datasets by Series Instance UID."""
    groups: dict[str, list[pydicom.Dataset]] = {}
    for ds in datasets:
        uid = getattr(ds, "SeriesInstanceUID", "unknown")
        groups.setdefault(uid, []).append(ds)
    return groups


def _match_series_uid(
    series_groups: dict[str, list[pydicom.Dataset]], partial_uid: str
) -> str:
    """Find series UID matching partial string."""
    matches = [uid for uid in series_groups if partial_uid in uid]
    if not matches:
        available = "\n  ".join(series_groups.keys())
        raise ValueError(
            f"No series matching '{partial_uid}'. Available:\n  {available}"
        )
    if len(matches) > 1:
        logger.warning(
            f"Multiple series match '{partial_uid}', using first: {matches[0]}"
        )
    return matches[0]


def _select_best_series(
    series_groups: dict[str, list[pydicom.Dataset]],
) -> str:
    """Select the series with the most slices."""
    return max(series_groups, key=lambda uid: len(series_groups[uid]))


def _detect_multiframe(
    datasets: list[pydicom.Dataset],
) -> tuple[InputType, DicomVolume | TemporalSequence] | None:
    """Detect multi-frame DICOM files (ultrasound cine clips).

    Each multi-frame file contains multiple frames in a single file.
    We pick the file with the most frames and unpack it as temporal data.
    """
    multiframe_ds = [
        ds for ds in datasets
        if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1
    ]

    if not multiframe_ds:
        return None

    # Pick the file with the most frames
    best = max(multiframe_ds, key=lambda ds: int(ds.NumberOfFrames))
    n_frames = int(best.NumberOfFrames)
    series_uid = getattr(best, "SeriesInstanceUID", "unknown")

    logger.info(
        f"Multi-frame DICOM detected: {n_frames} frames in "
        f"{getattr(best, 'filename', 'unknown')}"
    )

    raw_array = best.pixel_array
    pixel_array = raw_array.astype(np.float32)

    # Preserve original RGB data before converting to grayscale
    rgb_frames: np.ndarray | None = None
    if pixel_array.ndim == 4 and pixel_array.shape[-1] in (3, 4):
        # Store original RGB as uint8 (n_frames, rows, cols, 3)
        rgb_frames = np.clip(raw_array[..., :3], 0, 255).astype(np.uint8)
        # Convert to grayscale for voxels
        pixel_array = (
            0.299 * pixel_array[..., 0]
            + 0.587 * pixel_array[..., 1]
            + 0.114 * pixel_array[..., 2]
        )

    # pixel_array is now (n_frames, rows, cols)
    if pixel_array.ndim == 2:
        # Single frame treated as 2D — shouldn't happen here but handle it
        pixel_array = pixel_array[np.newaxis, ...]

    pixel_spacing = _get_pixel_spacing(best)
    slice_thickness = float(getattr(best, "SliceThickness", 1.0))

    # Build temporal sequence: each frame becomes a DicomVolume with 1 slice
    frames = []
    for i in range(pixel_array.shape[0]):
        frame_voxels = pixel_array[i][np.newaxis, ...]  # (1, rows, cols)
        frame_rgb = rgb_frames[i][np.newaxis, ...] if rgb_frames is not None else None
        volume = DicomVolume(
            voxels=frame_voxels,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            series_uid=series_uid,
            modality=getattr(best, "Modality", "Unknown"),
            metadata={
                "patient_name": str(getattr(best, "PatientName", "")),
                "study_description": getattr(best, "StudyDescription", ""),
            },
            rgb_data=frame_rgb,
        )
        frames.append(volume)

    # Get temporal resolution from frame time if available
    frame_time = float(getattr(best, "FrameTime", 33.3))  # ms per frame

    sequence = TemporalSequence(frames=frames, temporal_resolution=frame_time)
    return InputType.TEMPORAL, sequence


def _detect_temporal_frames(
    datasets: list[pydicom.Dataset],
) -> dict[int, list[pydicom.Dataset]] | None:
    """Detect temporal frames using Temporal Position Index or similar tags."""
    temporal_groups: dict[int, list[pydicom.Dataset]] = {}

    for ds in datasets:
        # Try standard temporal tags
        temporal_idx = None
        if hasattr(ds, "TemporalPositionIndex"):
            temporal_idx = int(ds.TemporalPositionIndex)
        elif hasattr(ds, "TemporalPositionIdentifier"):
            temporal_idx = int(ds.TemporalPositionIdentifier)

        if temporal_idx is not None:
            temporal_groups.setdefault(temporal_idx, []).append(ds)

    if len(temporal_groups) > 1:
        return temporal_groups
    return None


def _build_volume(
    datasets: list[pydicom.Dataset], series_uid: str
) -> DicomVolume:
    """Assemble a 3D volume from sorted DICOM datasets."""
    # Filter to consistent shape before stacking
    datasets = _filter_consistent_shape(datasets)

    # Sort by slice position
    datasets = _sort_by_position(datasets)

    # Extract pixel data
    slices = []
    for ds in datasets:
        pixel_array = ds.pixel_array.astype(np.float32)
        # Convert RGB/multi-channel to grayscale
        if pixel_array.ndim == 3 and pixel_array.shape[2] in (3, 4):
            # ITU-R BT.601 luma weights
            pixel_array = (
                0.299 * pixel_array[..., 0]
                + 0.587 * pixel_array[..., 1]
                + 0.114 * pixel_array[..., 2]
            )
        # Apply rescale if present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * float(slope) + float(intercept)
        slices.append(pixel_array)

    voxels = np.stack(slices, axis=0).astype(np.float32)

    # Extract metadata
    ds = datasets[0]
    pixel_spacing = _get_pixel_spacing(ds)
    slice_thickness = _get_slice_thickness(datasets)
    orientation = tuple(
        float(x)
        for x in getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    )
    position = tuple(
        float(x) for x in getattr(ds, "ImagePositionPatient", [0, 0, 0])
    )

    return DicomVolume(
        voxels=voxels,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        image_orientation=orientation,
        image_position_first=position,
        series_uid=series_uid,
        modality=getattr(ds, "Modality", "Unknown"),
        metadata={
            "patient_name": str(getattr(ds, "PatientName", "")),
            "study_description": getattr(ds, "StudyDescription", ""),
        },
    )


def _build_temporal_sequence(
    temporal_groups: dict[int, list[pydicom.Dataset]], series_uid: str
) -> tuple[InputType, TemporalSequence]:
    """Build a temporal sequence from grouped datasets."""
    frames = []
    for idx in sorted(temporal_groups.keys()):
        volume = _build_volume(temporal_groups[idx], series_uid)
        frames.append(volume)

    sequence = TemporalSequence(frames=frames)
    return InputType.TEMPORAL, sequence


def _filter_consistent_shape(
    datasets: list[pydicom.Dataset],
) -> list[pydicom.Dataset]:
    """Keep only slices matching the most common (Rows, Columns, SamplesPerPixel).

    Mixed-dimension slices (scouts, localizers, different reconstructions,
    RGB vs grayscale) are dropped with a warning.
    """
    shape_groups: dict[tuple[int, int, int], list[pydicom.Dataset]] = {}
    for ds in datasets:
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        if rows is None or cols is None:
            continue
        spp = int(getattr(ds, "SamplesPerPixel", 1))
        key = (int(rows), int(cols), spp)
        shape_groups.setdefault(key, []).append(ds)

    if not shape_groups:
        return datasets  # fallback — let np.stack raise the original error

    # Pick the shape with the most slices
    best_shape = max(shape_groups, key=lambda k: len(shape_groups[k]))
    best_group = shape_groups[best_shape]

    dropped = len(datasets) - len(best_group)
    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} slice(s) with inconsistent dimensions. "
            f"Keeping {len(best_group)} slices at "
            f"{best_shape[0]}x{best_shape[1]} (channels={best_shape[2]})."
        )

    return best_group


def _sort_by_position(datasets: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    """Sort datasets by slice position along the normal direction."""

    def sort_key(ds):
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos:
            return float(pos[2])  # Z position
        instance = getattr(ds, "InstanceNumber", 0)
        return float(instance)

    return sorted(datasets, key=sort_key)


def _get_pixel_spacing(ds: pydicom.Dataset) -> tuple[float, float]:
    """Extract pixel spacing from dataset."""
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing:
        return (float(spacing[0]), float(spacing[1]))

    # Try Imager Pixel Spacing for ultrasound
    spacing = getattr(ds, "ImagerPixelSpacing", None)
    if spacing:
        return (float(spacing[0]), float(spacing[1]))

    # Default 1mm spacing
    logger.warning("No pixel spacing found, using default 1.0mm")
    return (1.0, 1.0)


def _get_slice_thickness(datasets: list[pydicom.Dataset]) -> float:
    """Extract or calculate slice thickness."""
    ds = datasets[0]

    thickness = getattr(ds, "SliceThickness", None)
    if thickness:
        return float(thickness)

    # Calculate from slice positions
    if len(datasets) >= 2:
        pos1 = getattr(datasets[0], "ImagePositionPatient", None)
        pos2 = getattr(datasets[1], "ImagePositionPatient", None)
        if pos1 and pos2:
            return abs(float(pos2[2]) - float(pos1[2]))

    logger.warning("No slice thickness found, using default 1.0mm")
    return 1.0


def _load_single_file(
    file_path: Path,
) -> tuple[InputType, DicomVolume | TemporalSequence]:
    """Load a single DICOM file.

    If the file is multi-frame (e.g. ultrasound cine), it is unpacked
    as a temporal sequence rather than treated as a single slice.
    """
    ds = pydicom.dcmread(str(file_path))

    # Multi-frame files get routed through the multiframe handler
    if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
        return _detect_multiframe([ds])  # type: ignore[return-value]

    raw_array = ds.pixel_array
    pixel_array = raw_array.astype(np.float32)

    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    pixel_array = pixel_array * float(slope) + float(intercept)

    # Preserve original RGB data before converting to grayscale
    rgb_data: np.ndarray | None = None
    if pixel_array.ndim == 3 and pixel_array.shape[-1] in (3, 4):
        # Store original RGB as uint8 (Y, X, 3)
        rgb_data = np.clip(raw_array[..., :3], 0, 255).astype(np.uint8)
        rgb_data = rgb_data[np.newaxis, ...]  # (1, Y, X, 3)
        # Convert to grayscale for voxels
        pixel_array = (
            0.299 * pixel_array[..., 0]
            + 0.587 * pixel_array[..., 1]
            + 0.114 * pixel_array[..., 2]
        )

    # Ensure 3D (add Z dimension for single slice)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    pixel_spacing = _get_pixel_spacing(ds)

    volume = DicomVolume(
        voxels=pixel_array,
        pixel_spacing=pixel_spacing,
        slice_thickness=getattr(ds, "SliceThickness", 1.0),
        series_uid=getattr(ds, "SeriesInstanceUID", ""),
        modality=getattr(ds, "Modality", "Unknown"),
        metadata={
            "patient_name": str(getattr(ds, "PatientName", "")),
            "study_description": getattr(ds, "StudyDescription", ""),
        },
        rgb_data=rgb_data,
    )

    return InputType.SINGLE_SLICE, volume
