"""Unit tests for method registry and conversion methods."""

from __future__ import annotations

import numpy as np
import pytest

from med2glb.core.types import MethodParams
from med2glb.core.volume import DicomVolume
from med2glb.methods.registry import (
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


def test_classical_progress_callback(synthetic_volume):
    """Verify classical method calls progress callback with expected steps."""
    method = get_method("classical")
    params = MethodParams(threshold=500.0)

    calls = []
    def on_progress(desc, current=None, total=None):
        calls.append((desc, current, total))

    method.convert(synthetic_volume, params, progress=on_progress)

    assert len(calls) == 5
    descriptions = [c[0] for c in calls]
    assert "Smoothing volume..." in descriptions
    assert "Computing threshold..." in descriptions
    assert "Morphological cleanup..." in descriptions
    assert "Finding largest component..." in descriptions
    assert "Extracting surface..." in descriptions

    # All steps should have current/total = i/5
    for desc, current, total in calls:
        assert total == 5
        assert current is not None


def test_marching_cubes_progress_callback(synthetic_volume):
    """Verify marching cubes method calls progress callback."""
    method = get_method("marching-cubes")
    params = MethodParams(threshold=500.0)

    calls = []
    def on_progress(desc, current=None, total=None):
        calls.append((desc, current, total))

    method.convert(synthetic_volume, params, progress=on_progress)

    assert len(calls) == 2
    assert calls[0][0] == "Computing threshold..."
    assert calls[1][0] == "Running marching cubes..."


def test_convert_without_progress_callback(synthetic_volume):
    """Verify methods work fine with progress=None (default)."""
    method = get_method("classical")
    params = MethodParams(threshold=500.0)
    result = method.convert(synthetic_volume, params)
    assert len(result.meshes) >= 1
