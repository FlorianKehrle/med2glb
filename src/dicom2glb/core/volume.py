"""Volume data structures for DICOM data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DicomVolume:
    """3D array of voxel intensities assembled from DICOM files."""

    voxels: np.ndarray  # float32 [Z, Y, X]
    pixel_spacing: tuple[float, float]
    slice_thickness: float
    image_orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    image_position_first: tuple[float, float, float] = (0.0, 0.0, 0.0)
    series_uid: str = ""
    modality: str = "US"
    vendor: str | None = None
    metadata: dict = field(default_factory=dict)
    rgb_data: np.ndarray | None = None  # uint8 [Z, Y, X, 3] original RGB if available

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Return (z, y, x) spacing in mm."""
        return (self.slice_thickness, self.pixel_spacing[0], self.pixel_spacing[1])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.voxels.shape


@dataclass
class TemporalSequence:
    """Ordered collection of DicomVolumes representing time frames."""

    frames: list[DicomVolume]
    temporal_resolution: float | None = None
    is_loop: bool = True

    @property
    def frame_count(self) -> int:
        return len(self.frames)
