"""Unit tests for DICOM reader."""

from __future__ import annotations

import pytest

from med2glb.core.volume import DicomVolume, TemporalSequence
from med2glb.io.dicom_reader import (
    InputType,
    analyze_series,
    list_series,
    load_dicom_directory,
)


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


def test_load_mixed_shapes_drops_inconsistent(dicom_mixed_shapes_directory):
    """Slices with different Rows/Columns are dropped, keeping the majority."""
    input_type, data = load_dicom_directory(dicom_mixed_shapes_directory)
    assert input_type == InputType.VOLUME
    assert isinstance(data, DicomVolume)
    # Should keep the 8 slices at 32x32, drop the 2 scouts at 64x64
    assert data.voxels.shape == (8, 32, 32)


def test_load_mixed_channels_drops_rgb(dicom_mixed_channels_directory):
    """RGB slices are separated from grayscale by SamplesPerPixel grouping."""
    input_type, data = load_dicom_directory(dicom_mixed_channels_directory)
    assert input_type == InputType.VOLUME
    assert isinstance(data, DicomVolume)
    # Should keep the 6 grayscale slices, drop the 3 RGB slices
    assert data.voxels.shape == (6, 32, 32)


def test_load_multi_series_picks_largest(dicom_multi_series_directory):
    """With multiple series, the one with more slices is selected."""
    input_type, data = load_dicom_directory(dicom_multi_series_directory)
    assert input_type == InputType.VOLUME
    assert isinstance(data, DicomVolume)
    # Series A has 8 slices vs Series B with 4
    assert data.voxels.shape[0] == 8


# --- analyze_series tests ---


def test_analyze_series_single(dicom_directory):
    """Single series is classified as '3D volume'."""
    result = analyze_series(dicom_directory)
    assert len(result) == 1
    info = result[0]
    assert info.data_type == "3D volume"
    assert info.modality == "CT"
    assert info.file_count == 10
    assert "10 slices" in info.detail


def test_analyze_series_multi(dicom_multi_series_directory):
    """Multiple series are detected and classified correctly."""
    result = analyze_series(dicom_multi_series_directory)
    assert len(result) == 2
    # Both should be 3D volumes
    for info in result:
        assert info.data_type == "3D volume"
    # First should be the larger series (sorted by count desc)
    assert result[0].file_count >= result[1].file_count


def test_analyze_series_temporal(dicom_temporal_directory):
    """Temporal series is classified as '3D+T volume'."""
    result = analyze_series(dicom_temporal_directory)
    assert len(result) == 1
    info = result[0]
    assert info.data_type == "3D+T volume"
    assert info.recommended_output == "animated 3D mesh"


def test_analyze_series_multiframe(dicom_multiframe_directory):
    """Multi-frame series (no spatial info) is classified as '2D cine'."""
    result = analyze_series(dicom_multiframe_directory)
    assert len(result) == 1
    info = result[0]
    assert info.data_type == "2D cine"
    assert info.is_multiframe is True
    assert info.number_of_frames == 20
    assert info.recommended_output == "textured plane"
    assert "20 frames" in info.detail
