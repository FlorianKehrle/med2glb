"""Vendor-specific 3D echo DICOM reader (Philips, GE)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom

from dicom2glb.core.volume import DicomVolume, TemporalSequence

logger = logging.getLogger(__name__)

# Known private tags for 3D echo vendors
PHILIPS_PRIVATE_CREATOR = "Philips US Imaging DD 033"
GE_PRIVATE_CREATOR = "GEMS_Ultrasound_MovieGroup_001"


def load_echo_data(input_path: Path) -> TemporalSequence | DicomVolume:
    """Load 3D echo data, handling vendor-specific formats.

    Supports:
    - Multi-frame DICOM with 3D echo data (Philips, GE)
    - Standard temporal DICOM series
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return _load_multiframe_echo(input_path)

    # Try to find multi-frame files in directory
    for path in input_path.rglob("*"):
        if path.is_file():
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                if _is_multiframe_echo(ds):
                    logger.info(f"Found multi-frame echo: {path}")
                    return _load_multiframe_echo(path)
            except Exception:
                continue

    raise ValueError(f"No 3D echo data found in {input_path}")


def detect_vendor(ds: pydicom.Dataset) -> str | None:
    """Detect echo equipment vendor from DICOM dataset."""
    manufacturer = getattr(ds, "Manufacturer", "").lower()

    if "philips" in manufacturer:
        return "Philips"
    elif "ge" in manufacturer or "general electric" in manufacturer:
        return "GE"
    elif "siemens" in manufacturer:
        return "Siemens"
    elif "canon" in manufacturer or "toshiba" in manufacturer:
        return "Canon"

    return None


def _is_multiframe_echo(ds: pydicom.Dataset) -> bool:
    """Check if a dataset is a multi-frame 3D echo."""
    n_frames = getattr(ds, "NumberOfFrames", 0)
    modality = getattr(ds, "Modality", "")
    return int(n_frames) > 1 and modality in ("US", "")


def _load_multiframe_echo(file_path: Path) -> TemporalSequence | DicomVolume:
    """Load a multi-frame 3D echo DICOM file."""
    ds = pydicom.dcmread(str(file_path))
    vendor = detect_vendor(ds)
    logger.info(f"Loading multi-frame echo from {vendor or 'unknown'} vendor")

    n_frames = int(ds.NumberOfFrames)
    pixel_data = ds.pixel_array  # [frames, rows, cols] or [frames, rows, cols, channels]

    if pixel_data.ndim == 4:
        # RGB or multi-channel — use first channel
        pixel_data = pixel_data[..., 0]

    pixel_data = pixel_data.astype(np.float32)

    # Try to detect temporal vs spatial frame organization
    frame_info = _parse_frame_sequence(ds)

    if frame_info is not None:
        return _build_sequence_from_frame_info(ds, pixel_data, frame_info, vendor)

    # Fallback: try to detect 3D+T structure from frame count and dimensions
    spatial_frames, temporal_frames = _guess_3dt_structure(n_frames, ds)

    if temporal_frames > 1 and spatial_frames > 1:
        return _build_temporal_from_flat(
            ds, pixel_data, spatial_frames, temporal_frames, vendor
        )

    # Single 3D volume (no temporal dimension)
    volume = _build_single_echo_volume(ds, pixel_data, vendor)
    return volume


def _parse_frame_sequence(ds: pydicom.Dataset) -> list[dict] | None:
    """Parse Per-Frame Functional Groups to get frame organization."""
    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if pffg is None:
        return None

    frame_info = []
    for i, frame_group in enumerate(pffg):
        info = {"index": i}

        # Temporal position
        temporal_seq = getattr(frame_group, "FrameContentSequence", None)
        if temporal_seq:
            tc = temporal_seq[0]
            info["temporal_index"] = getattr(tc, "TemporalPositionIndex", None)
            info["dimension_index"] = getattr(tc, "DimensionIndexValues", None)

        # Stack position
        plane_seq = getattr(frame_group, "PlanePositionSequence", None)
        if plane_seq:
            pp = plane_seq[0]
            pos = getattr(pp, "ImagePositionPatient", None)
            if pos:
                info["position"] = [float(p) for p in pos]

        frame_info.append(info)

    # Check if we have useful temporal info
    if any(f.get("temporal_index") is not None for f in frame_info):
        return frame_info

    return None


def _build_sequence_from_frame_info(
    ds: pydicom.Dataset,
    pixel_data: np.ndarray,
    frame_info: list[dict],
    vendor: str | None,
) -> TemporalSequence:
    """Build temporal sequence using per-frame functional groups."""
    # Group by temporal index
    temporal_groups: dict[int, list[int]] = {}
    for f in frame_info:
        t_idx = f.get("temporal_index", 0) or 0
        temporal_groups.setdefault(t_idx, []).append(f["index"])

    pixel_spacing = _get_echo_spacing(ds)

    frames = []
    for t_idx in sorted(temporal_groups.keys()):
        frame_indices = sorted(temporal_groups[t_idx])
        slices = pixel_data[frame_indices]
        volume = DicomVolume(
            voxels=slices,
            pixel_spacing=pixel_spacing,
            slice_thickness=_get_echo_slice_thickness(ds, len(frame_indices)),
            series_uid=getattr(ds, "SeriesInstanceUID", ""),
            modality="US",
            vendor=vendor,
        )
        frames.append(volume)

    temporal_res = _get_temporal_resolution(ds)
    return TemporalSequence(
        frames=frames,
        temporal_resolution=temporal_res,
        is_loop=True,
    )


def _guess_3dt_structure(
    n_frames: int, ds: pydicom.Dataset
) -> tuple[int, int]:
    """Guess the spatial x temporal frame organization."""
    # Look for NumberOfTemporalPositions
    n_temporal = getattr(ds, "NumberOfTemporalPositions", None)
    if n_temporal and int(n_temporal) > 1:
        n_spatial = n_frames // int(n_temporal)
        if n_spatial * int(n_temporal) == n_frames:
            return n_spatial, int(n_temporal)

    # Try dimension organization
    dim_org = getattr(ds, "DimensionOrganizationSequence", None)
    if dim_org:
        # Some vendors encode frame structure here
        pass

    # Common 3D echo: 20-40 temporal frames × 10-30 spatial slices
    # Try to factor n_frames
    for spatial in range(max(2, n_frames // 60), min(n_frames // 2, n_frames // 3 + 1)):
        if n_frames % spatial == 0:
            temporal = n_frames // spatial
            if 5 <= temporal <= 60:
                return spatial, temporal

    return n_frames, 1


def _build_temporal_from_flat(
    ds: pydicom.Dataset,
    pixel_data: np.ndarray,
    n_spatial: int,
    n_temporal: int,
    vendor: str | None,
) -> TemporalSequence:
    """Build temporal sequence from flat frame array."""
    pixel_spacing = _get_echo_spacing(ds)
    slice_thickness = _get_echo_slice_thickness(ds, n_spatial)

    frames = []
    for t in range(n_temporal):
        start = t * n_spatial
        end = start + n_spatial
        slices = pixel_data[start:end]

        volume = DicomVolume(
            voxels=slices,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            series_uid=getattr(ds, "SeriesInstanceUID", ""),
            modality="US",
            vendor=vendor,
        )
        frames.append(volume)

    temporal_res = _get_temporal_resolution(ds)
    return TemporalSequence(
        frames=frames, temporal_resolution=temporal_res, is_loop=True
    )


def _build_single_echo_volume(
    ds: pydicom.Dataset, pixel_data: np.ndarray, vendor: str | None
) -> DicomVolume:
    """Build a single 3D volume from multi-frame echo."""
    pixel_spacing = _get_echo_spacing(ds)

    return DicomVolume(
        voxels=pixel_data,
        pixel_spacing=pixel_spacing,
        slice_thickness=_get_echo_slice_thickness(ds, len(pixel_data)),
        series_uid=getattr(ds, "SeriesInstanceUID", ""),
        modality="US",
        vendor=vendor,
    )


def _get_echo_spacing(ds: pydicom.Dataset) -> tuple[float, float]:
    """Get pixel spacing for echo data."""
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing:
        return (float(spacing[0]), float(spacing[1]))

    # Check shared functional groups
    sfgs = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if sfgs:
        pixel_measures = getattr(sfgs[0], "PixelMeasuresSequence", None)
        if pixel_measures:
            ps = getattr(pixel_measures[0], "PixelSpacing", None)
            if ps:
                return (float(ps[0]), float(ps[1]))

    # Physical delta (some vendors)
    dx = getattr(ds, "PhysicalDeltaX", None)
    dy = getattr(ds, "PhysicalDeltaY", None)
    if dx and dy:
        return (abs(float(dy)) * 10, abs(float(dx)) * 10)  # cm to mm

    logger.warning("No pixel spacing for echo, using default 1.0mm")
    return (1.0, 1.0)


def _get_echo_slice_thickness(ds: pydicom.Dataset, n_slices: int) -> float:
    """Get or estimate slice thickness for echo data."""
    thickness = getattr(ds, "SliceThickness", None)
    if thickness:
        return float(thickness)

    sfgs = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if sfgs:
        pixel_measures = getattr(sfgs[0], "PixelMeasuresSequence", None)
        if pixel_measures:
            st = getattr(pixel_measures[0], "SliceThickness", None)
            if st:
                return float(st)
            spacing = getattr(pixel_measures[0], "SpacingBetweenSlices", None)
            if spacing:
                return float(spacing)

    # Assume isotropic for echo if pixel spacing available
    ps = _get_echo_spacing(ds)
    return ps[0]


def _get_temporal_resolution(ds: pydicom.Dataset) -> float | None:
    """Get temporal resolution in ms."""
    # Frame time
    ft = getattr(ds, "FrameTime", None)
    if ft:
        return float(ft)

    # Heart rate based estimate
    hr = getattr(ds, "HeartRate", None)
    n_temporal = getattr(ds, "NumberOfTemporalPositions", None)
    if hr and n_temporal:
        cycle_ms = 60000.0 / float(hr)
        return cycle_ms / int(n_temporal)

    return None
