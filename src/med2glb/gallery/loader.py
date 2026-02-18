"""Gallery loader: read all DICOM slices without shape filtering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom

from med2glb.core.types import GallerySlice
from med2glb.io.dicom_reader import scan_and_group

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
            raw = ds.pixel_array
        except Exception:
            logger.warning(f"Skipping file with unreadable pixel data: {ds.filename}")
            continue

        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))

        # Shared metadata
        pos = getattr(ds, "ImagePositionPatient", None)
        image_position = tuple(float(x) for x in pos) if pos else None

        orient = getattr(ds, "ImageOrientationPatient", None)
        image_orientation = tuple(float(x) for x in orient) if orient else None

        spacing = getattr(ds, "PixelSpacing", None)
        pixel_spacing = (float(spacing[0]), float(spacing[1])) if spacing else (1.0, 1.0)

        instance_number = int(getattr(ds, "InstanceNumber", 0))
        modality = getattr(ds, "Modality", "Unknown")
        filename = str(getattr(ds, "filename", ""))
        spp = int(getattr(ds, "SamplesPerPixel", 1))
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        num_frames = int(getattr(ds, "NumberOfFrames", 1))

        if num_frames > 1 and raw.ndim >= 3:
            # Multi-frame DICOM: expand into one GallerySlice per frame
            for fi in range(num_frames):
                frame_data = _normalize_frame(raw[fi], spp, slope, intercept)
                if rows == 0:
                    rows = frame_data.shape[0]
                if cols == 0:
                    cols = frame_data.shape[1] if frame_data.ndim >= 2 else 0
                slices.append(
                    GallerySlice(
                        pixel_data=frame_data,
                        pixel_spacing=pixel_spacing,
                        image_position=image_position,
                        image_orientation=image_orientation,
                        instance_number=instance_number,
                        filename=filename,
                        rows=rows,
                        cols=cols,
                        temporal_index=fi + 1,
                        series_uid=uid,
                        modality=modality,
                    )
                )
        else:
            # Single-frame DICOM
            frame_data = _normalize_frame(raw, spp, slope, intercept)
            if rows == 0:
                rows = frame_data.shape[0]
            if cols == 0:
                cols = frame_data.shape[1] if frame_data.ndim >= 2 else 0

            temporal_index = None
            if hasattr(ds, "TemporalPositionIdentifier"):
                temporal_index = int(ds.TemporalPositionIdentifier)
            elif hasattr(ds, "TemporalPositionIndex"):
                temporal_index = int(ds.TemporalPositionIndex)

            slices.append(
                GallerySlice(
                    pixel_data=frame_data,
                    pixel_spacing=pixel_spacing,
                    image_position=image_position,
                    image_orientation=image_orientation,
                    instance_number=instance_number,
                    filename=filename,
                    rows=rows,
                    cols=cols,
                    temporal_index=temporal_index,
                    series_uid=uid,
                    modality=modality,
                )
            )

    slices.sort(key=lambda s: s.instance_number)
    return slices


def _normalize_frame(
    raw: np.ndarray,
    spp: int,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Normalize a single frame to [Y, X] float32 or [Y, X, 3] uint8.

    Squeezes leading singleton dimensions first.
    """
    # Squeeze leading singleton dims (e.g. (1, Y, X) → (Y, X))
    while raw.ndim > 3 or (raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[-1] not in (3, 4)):
        raw = raw[0]

    if spp == 3 or (raw.ndim == 3 and raw.shape[-1] in (3, 4)):
        return np.clip(raw[..., :3], 0, 255).astype(np.uint8)

    pixel_array = raw.astype(np.float32)
    pixel_array = pixel_array * slope + intercept
    return pixel_array
