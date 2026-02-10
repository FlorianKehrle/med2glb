"""Unit tests for method registry and conversion methods."""

from __future__ import annotations

import numpy as np
import pytest

from dicom2glb.core.types import MethodParams
from dicom2glb.core.volume import DicomVolume
from dicom2glb.methods.registry import (
    _ensure_methods_loaded,
    get_method,
    list_methods,
)


@pytest.fixture(autouse=True)
def _load_methods():
    _ensure_methods_loaded()


def test_list_methods_returns_available():
    methods = list_methods()
    names = [m["name"] for m in methods]
    assert "marching-cubes" in names
    assert "classical" in names


def test_get_method_marching_cubes():
    method = get_method("marching-cubes")
    assert method.name == "marching-cubes"
    assert method.supports_animation()


def test_get_method_classical():
    method = get_method("classical")
    assert method.name == "classical"
    assert method.supports_animation()


def test_get_method_unknown_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        get_method("nonexistent")


def test_marching_cubes_convert(synthetic_volume):
    method = get_method("marching-cubes")
    params = MethodParams(threshold=500.0)
    result = method.convert(synthetic_volume, params)

    assert len(result.meshes) >= 1
    mesh = result.meshes[0]
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert len(mesh.vertices) > 0


def test_marching_cubes_auto_threshold(synthetic_volume):
    method = get_method("marching-cubes")
    params = MethodParams()
    result = method.convert(synthetic_volume, params)
    assert len(result.meshes) >= 1
    assert any("Auto-detected" in w for w in result.warnings)


def test_classical_convert(synthetic_volume):
    method = get_method("classical")
    params = MethodParams(threshold=500.0)
    result = method.convert(synthetic_volume, params)

    assert len(result.meshes) >= 1
    mesh = result.meshes[0]
    assert mesh.vertices.shape[1] == 3
    assert len(mesh.vertices) > 0
