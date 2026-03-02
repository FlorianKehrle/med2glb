"""Unit tests for DICOM pipeline functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pygltflib
import pytest

from med2glb._pipeline_dicom import (
    _export,
    _parse_multi_threshold,
    convert_series,
    enforce_size_limit,
    make_output_path,
    parse_selection,
    run_dicom_from_config,
)
from med2glb.core.types import MaterialConfig, MeshData, SeriesInfo


# --- _parse_multi_threshold tests ---


def test_parse_multi_threshold_single_layer():
    layers = _parse_multi_threshold("100:bone:0.8")
    assert len(layers) == 1
    assert layers[0].threshold == 100.0
    assert layers[0].label == "bone"
    assert layers[0].material.alpha == pytest.approx(0.8)
    assert layers[0].material.name == "bone"


def test_parse_multi_threshold_multiple_layers():
    layers = _parse_multi_threshold("100:bone:0.8,200:tissue:0.5,50:skin:0.3")
    assert len(layers) == 3
    assert layers[0].threshold == 100.0
    assert layers[1].threshold == 200.0
    assert layers[2].threshold == 50.0
    assert layers[1].label == "tissue"
    assert layers[2].material.alpha == pytest.approx(0.3)


def test_parse_multi_threshold_invalid_format():
    with pytest.raises(ValueError, match="Invalid multi-threshold format"):
        _parse_multi_threshold("100:bone")


def test_parse_multi_threshold_missing_alpha():
    with pytest.raises(ValueError, match="Invalid multi-threshold format"):
        _parse_multi_threshold("100")


# --- _export tests ---


def test_export_glb(synthetic_mesh, tmp_path):
    from med2glb.core.types import ConversionResult

    result = ConversionResult(meshes=[synthetic_mesh], method_name="test")
    output = tmp_path / "test.glb"
    _export(result, output, "glb", animate=False)
    assert output.exists()
    assert output.stat().st_size > 0


def test_export_stl(synthetic_mesh, tmp_path):
    from med2glb.core.types import ConversionResult

    result = ConversionResult(meshes=[synthetic_mesh], method_name="test")
    output = tmp_path / "test.stl"
    _export(result, output, "stl", animate=False)
    assert output.exists()


def test_export_obj(synthetic_mesh, tmp_path):
    from med2glb.core.types import ConversionResult

    result = ConversionResult(meshes=[synthetic_mesh], method_name="test")
    output = tmp_path / "test.obj"
    _export(result, output, "obj", animate=False)
    assert output.exists()


def test_export_unsupported_format(synthetic_mesh, tmp_path):
    from med2glb.core.types import ConversionResult

    result = ConversionResult(meshes=[synthetic_mesh], method_name="test")
    output = tmp_path / "test.xyz"
    with pytest.raises(ValueError, match="Unsupported format"):
        _export(result, output, "xyz", animate=False)


# --- enforce_size_limit tests ---


def test_enforce_size_limit_under_limit(tmp_path):
    """File under the limit should not be modified."""
    path = tmp_path / "small.glb"
    path.write_bytes(b"x" * 100)
    original_size = path.stat().st_size

    from rich.progress import Progress
    with Progress() as progress:
        enforce_size_limit(path, max_size_mb=1, strategy="draco", progress=progress)

    assert path.stat().st_size == original_size


def test_enforce_size_limit_nonexistent(tmp_path):
    """Non-existent file should be a no-op."""
    path = tmp_path / "missing.glb"

    from rich.progress import Progress
    with Progress() as progress:
        enforce_size_limit(path, max_size_mb=1, strategy="draco", progress=progress)
    # Should not raise


# --- convert_series tests ---


def test_convert_series_single_file(dicom_directory, tmp_path):
    """Convert a DICOM directory to GLB."""
    output = tmp_path / "output.glb"
    convert_series(
        input_path=dicom_directory,
        output=output,
        method_name="marching-cubes",
        format="glb",
        animate=False,
        threshold=250.0,
        smoothing=3,
        target_faces=5000,
        alpha=1.0,
        multi_threshold=None,
        series_uid=None,
        max_size_mb=0,
        compress_strategy="draco",
        verbose=False,
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_convert_series_with_alpha(dicom_directory, tmp_path):
    """Convert with transparency applied."""
    output = tmp_path / "transparent.glb"
    convert_series(
        input_path=dicom_directory,
        output=output,
        method_name="marching-cubes",
        format="glb",
        animate=False,
        threshold=250.0,
        smoothing=3,
        target_faces=5000,
        alpha=0.5,
        multi_threshold=None,
        series_uid=None,
        max_size_mb=0,
        compress_strategy="draco",
        verbose=False,
    )
    assert output.exists()
    gltf = pygltflib.GLTF2.load(str(output))
    # At least one material should have BLEND alpha
    assert any(m.alphaMode == pygltflib.BLEND for m in gltf.materials)


def test_convert_series_stl_format(dicom_directory, tmp_path):
    """Convert to STL format."""
    output = tmp_path / "output.stl"
    convert_series(
        input_path=dicom_directory,
        output=output,
        method_name="marching-cubes",
        format="stl",
        animate=False,
        threshold=250.0,
        smoothing=3,
        target_faces=5000,
        alpha=1.0,
        multi_threshold=None,
        series_uid=None,
        max_size_mb=0,
        compress_strategy="draco",
        verbose=False,
    )
    assert output.exists()


# --- run_dicom_from_config tests ---


def test_run_dicom_from_config(dicom_directory, tmp_path):
    """DicomConfig-based execution should produce output."""
    from med2glb.core.types import DicomConfig

    config = DicomConfig(
        input_path=dicom_directory,
        method="marching-cubes",
        format="glb",
        animate=False,
        threshold=250.0,
        smoothing=3,
        target_faces=5000,
        alpha=1.0,
        series_uid=None,
        max_size_mb=0,
        compress_strategy="draco",
        verbose=False,
        name="test",
    )
    output = tmp_path / "config_output.glb"
    run_dicom_from_config(config, output)
    assert output.exists()
