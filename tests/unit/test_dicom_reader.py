"""Unit tests for DICOM reader."""

from __future__ import annotations

import pytest

from dicom2glb.core.volume import DicomVolume, TemporalSequence
from dicom2glb.io.dicom_reader import InputType, list_series, load_dicom_directory


def test_load_volume_from_directory(dicom_directory):
    input_type, data = load_dicom_directory(dicom_directory)
    assert input_type == InputType.VOLUME
    assert isinstance(data, DicomVolume)
    assert data.voxels.ndim == 3
    assert data.voxels.shape[0] == 10  # 10 slices
    assert data.pixel_spacing == (1.0, 1.0)
    assert data.slice_thickness == 1.0


def test_load_temporal_from_directory(dicom_temporal_directory):
    input_type, data = load_dicom_directory(dicom_temporal_directory)
    assert input_type == InputType.TEMPORAL
    assert isinstance(data, TemporalSequence)
    assert data.frame_count == 3
    for frame in data.frames:
        assert isinstance(frame, DicomVolume)
        assert frame.voxels.ndim == 3


def test_list_series(dicom_directory):
    series = list_series(dicom_directory)
    assert len(series) >= 1
    s = series[0]
    assert "series_uid" in s
    assert s["modality"] == "CT"
    assert s["slice_count"] == 10


def test_load_nonexistent_raises():
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        load_dicom_directory(Path("/nonexistent/path"))


def test_load_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No valid DICOM"):
        load_dicom_directory(tmp_path)
