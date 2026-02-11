"""Gallery loader: read all DICOM slices without shape filtering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom

from dicom2glb.core.types import GallerySlice
from dicom2glb.io.dicom_reader import scan_and_group

logger = logging.getLogger(__name__)


def load_all_slices(
    input_path: Path,
    series_uid: str | None = None,
) -> list[GallerySlice]:
    """Load every DICOM slice as a GallerySlice — no shape filtering.

    If *series_uid* is given, only that series is loaded; otherwise the
    series with the most files is selected.
    """
    groups = scan_and_group(input_path)

    if series_uid:
        matches = [uid for uid in groups if series_uid in uid]
        if not matches:
            raise ValueError(f"No series matching '{series_uid}'")
        uid = matches[0]
    else:
        uid = max(groups, key=lambda k: len(groups[k]))

    datasets = groups[uid]
    slices: list[GallerySlice] = []

    for ds in datasets:
        try:
            pixel_data = _extract_pixel_data(ds)
        except Exception:
            logger.warning(f"Skipping file with unreadable pixel data: {ds.filename}")
            continue

        rows = int(getattr(ds, "Rows", pixel_data.shape[0]))
        cols = int(getattr(ds, "Columns", pixel_data.shape[1] if pixel_data.ndim >= 2 else 0))

        # Position / orientation
        pos = getattr(ds, "ImagePositionPatient", None)
        image_position = tuple(float(x) for x in pos) if pos else None

        orient = getattr(ds, "ImageOrientationPatient", None)
        image_orientation = tuple(float(x) for x in orient) if orient else None

        # Pixel spacing
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing:
            pixel_spacing = (float(spacing[0]), float(spacing[1]))
        else:
            pixel_spacing = (1.0, 1.0)

        # Temporal index
        temporal_index = None
        if hasattr(ds, "TemporalPositionIdentifier"):
            temporal_index = int(ds.TemporalPositionIdentifier)
        elif hasattr(ds, "TemporalPositionIndex"):
            temporal_index = int(ds.TemporalPositionIndex)

        instance_number = int(getattr(ds, "InstanceNumber", 0))

        slices.append(
            GallerySlice(
                pixel_data=pixel_data,
                pixel_spacing=pixel_spacing,
                image_position=image_position,
                image_orientation=image_orientation,
                instance_number=instance_number,
                filename=str(getattr(ds, "filename", "")),
                rows=rows,
                cols=cols,
                temporal_index=temporal_index,
                series_uid=uid,
                modality=getattr(ds, "Modality", "Unknown"),
            )
        )

    slices.sort(key=lambda s: s.instance_number)
    return slices


def _extract_pixel_data(ds: pydicom.Dataset) -> np.ndarray:
    """Extract 2D pixel data from a DICOM dataset.

    Returns [Y, X] grayscale (float32) or [Y, X, 3] RGB (uint8).
    Multi-frame files use the first frame only.
    """
    raw = ds.pixel_array

    # Multi-frame: squeeze or take first frame to get a 2D/3D array
    num_frames = int(getattr(ds, "NumberOfFrames", 1))
    if num_frames > 1 and raw.ndim >= 3:
        raw = raw[0]
    # Squeeze any remaining leading singleton dimensions (e.g. (1, Y, X) → (Y, X))
    while raw.ndim > 3 or (raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[-1] not in (3, 4)):
        raw = raw[0]

    spp = int(getattr(ds, "SamplesPerPixel", 1))
    if spp == 3 or (raw.ndim == 3 and raw.shape[-1] in (3, 4)):
        # RGB — keep as uint8
        return np.clip(raw[..., :3], 0, 255).astype(np.uint8)

    # Grayscale — apply rescale and normalise to uint8-ready float
    pixel_array = raw.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixel_array = pixel_array * slope + intercept
    return pixel_array
