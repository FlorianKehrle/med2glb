"""DICOM reader: load directory, group by series, detect input type, assemble volumes."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

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

    # Detect temporal data
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
        elif hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
            # Multi-frame DICOM — handled separately by echo reader
            return None

        if temporal_idx is not None:
            temporal_groups.setdefault(temporal_idx, []).append(ds)

    if len(temporal_groups) > 1:
        return temporal_groups
    return None


def _build_volume(
    datasets: list[pydicom.Dataset], series_uid: str
) -> DicomVolume:
    """Assemble a 3D volume from sorted DICOM datasets."""
    # Sort by slice position
    datasets = _sort_by_position(datasets)

    # Extract pixel data
    slices = []
    for ds in datasets:
        pixel_array = ds.pixel_array.astype(np.float32)
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


def _load_single_file(file_path: Path) -> tuple[InputType, DicomVolume]:
    """Load a single DICOM file."""
    ds = pydicom.dcmread(str(file_path))
    pixel_array = ds.pixel_array.astype(np.float32)

    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    pixel_array = pixel_array * float(slope) + float(intercept)

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
    )

    return InputType.SINGLE_SLICE, volume
