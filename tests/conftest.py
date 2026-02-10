"""Shared test fixtures: synthetic DICOM data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dicom2glb.core.types import MaterialConfig, MeshData
from dicom2glb.core.volume import DicomVolume, TemporalSequence


@pytest.fixture
def synthetic_volume() -> DicomVolume:
    """Create a synthetic 3D volume with a sphere."""
    shape = (30, 30, 30)
    voxels = np.zeros(shape, dtype=np.float32)

    center = np.array([15, 15, 15])
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    voxels[dist < 10] = 1000.0

    return DicomVolume(
        voxels=voxels,
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
        series_uid="1.2.3.4.5",
        modality="CT",
    )


@pytest.fixture
def synthetic_temporal_sequence() -> TemporalSequence:
    """Create a synthetic temporal sequence (3 frames with varying sphere size)."""
    frames = []
    for i in range(3):
        shape = (30, 30, 30)
        voxels = np.zeros(shape, dtype=np.float32)
        radius = 8 + i * 2
        center = np.array([15, 15, 15])
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
        voxels[dist < radius] = 1000.0

        frames.append(
            DicomVolume(
                voxels=voxels,
                pixel_spacing=(1.0, 1.0),
                slice_thickness=1.0,
                series_uid="1.2.3.4.5",
                modality="US",
            )
        )

    return TemporalSequence(frames=frames, temporal_resolution=33.3)


@pytest.fixture
def synthetic_mesh() -> MeshData:
    """Create a simple triangulated cube mesh."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)

    return MeshData(
        vertices=vertices,
        faces=faces,
        structure_name="test_cube",
        material=MaterialConfig(base_color=(0.8, 0.2, 0.2), alpha=1.0),
    )


@pytest.fixture
def dicom_directory(tmp_path) -> Path:
    """Create a temporary directory with synthetic DICOM files (3D volume)."""
    series_uid = generate_uid()

    for i in range(10):
        _write_synthetic_dicom(
            tmp_path / f"slice_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=i + 1,
            slice_location=float(i),
        )

    return tmp_path


@pytest.fixture
def dicom_temporal_directory(tmp_path) -> Path:
    """Create a temporary directory with synthetic temporal DICOM files."""
    series_uid = generate_uid()

    for frame in range(3):
        for slice_idx in range(10):
            _write_synthetic_dicom(
                tmp_path / f"frame_{frame:02d}_slice_{slice_idx:03d}.dcm",
                series_uid=series_uid,
                instance_number=frame * 10 + slice_idx + 1,
                slice_location=float(slice_idx),
                temporal_position=frame + 1,
            )

    return tmp_path


def _write_synthetic_dicom(
    path: Path,
    series_uid: str,
    instance_number: int,
    slice_location: float,
    rows: int = 32,
    cols: int = 32,
    temporal_position: int | None = None,
) -> None:
    """Write a single synthetic DICOM file."""
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, slice_location]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.SliceLocation = slice_location
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    if temporal_position is not None:
        ds.TemporalPositionIdentifier = temporal_position

    # Generate pixel data with a sphere pattern
    pixel_data = np.zeros((rows, cols), dtype=np.uint16)
    center = (rows // 2, cols // 2)
    radius = min(rows, cols) // 4
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist = (yy - center[0])**2 + (xx - center[1])**2
    pixel_data[dist < radius**2] = 500

    ds.PixelData = pixel_data.tobytes()
    ds.save_as(str(path))
