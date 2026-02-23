"""Tests for cli_wizard.py — input analysis, wizard logic, configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rich.console import Console

from med2glb.cli_wizard import (
    _check_ai_available,
    _parse_mesh_selection,
    run_carto_wizard,
    run_dicom_wizard,
)
from med2glb.core.types import (
    CartoConfig,
    CartoMesh,
    CartoPoint,
    CartoStudy,
    DicomConfig,
    SeriesInfo,
)


@pytest.fixture
def carto_study():
    """Minimal CARTO study for wizard testing."""
    vertices = np.array([
        [0, 0, 0], [10, 0, 0], [5, 10, 0], [0, 10, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    normals = np.tile([0, 0, 1], (4, 1)).astype(np.float64)

    mesh = CartoMesh(
        mesh_id=1,
        vertices=vertices,
        faces=faces,
        normals=normals,
        group_ids=np.zeros(4, dtype=np.int32),
        face_group_ids=np.zeros(2, dtype=np.int32),
        mesh_color=(0, 1, 0, 1),
        color_names=["LAT"],
        structure_name="TestMap",
    )
    points = [
        CartoPoint(1, np.array([1, 1, 0.0]), np.zeros(3), 2.0, 8.0, -50.0),
        CartoPoint(2, np.array([9, 1, 0.0]), np.zeros(3), 1.0, 5.0, 100.0),
    ]
    return CartoStudy(
        meshes=[mesh],
        points={"TestMap": points},
        version="6.0",
        study_name="TestStudy",
    )


@pytest.fixture
def series_list():
    """Minimal DICOM series list for wizard testing."""
    return [
        SeriesInfo(
            series_uid="1.2.3.4",
            modality="CT",
            description="Test CT Volume",
            file_count=50,
            data_type="3D volume",
            detail="50 slices",
            dimensions="512x512x50",
            recommended_method="classical",
            recommended_output="3D mesh",
        ),
        SeriesInfo(
            series_uid="1.2.3.5",
            modality="US",
            description="Echo Cine",
            file_count=1,
            data_type="2D cine",
            detail="30 frames",
            dimensions="640x480",
            recommended_method="classical",
            recommended_output="textured plane",
            is_multiframe=True,
            number_of_frames=30,
        ),
    ]


class TestParseMeshSelection:
    def test_all(self):
        assert _parse_mesh_selection("all", 3) is None

    def test_single(self):
        assert _parse_mesh_selection("2", 3) == [1]

    def test_comma_separated(self):
        assert _parse_mesh_selection("1,3", 5) == [0, 2]

    def test_out_of_range_ignored(self):
        result = _parse_mesh_selection("1,10", 3)
        assert result == [0]

    def test_all_out_of_range(self):
        result = _parse_mesh_selection("10", 3)
        assert result is None  # falls back to None


class TestRunCartoWizard:
    def test_fully_preset(self, carto_study):
        """When all presets are given, no prompts are needed."""
        console = Console(file=None, force_terminal=False)
        config = run_carto_wizard(
            carto_study, Path("/fake/input"), console,
            preset_coloring="bipolar",
            preset_animate=True,
            preset_static=False,
            preset_vectors=True,
            preset_subdivide=1,
            preset_meshes="all",
        )
        assert isinstance(config, CartoConfig)
        assert config.coloring == "bipolar"
        assert config.animate is True
        assert config.static is False
        assert config.vectors is True
        assert config.subdivide == 1
        assert config.selected_mesh_indices is None  # "all"

    def test_non_interactive_defaults(self, carto_study):
        """In non-TTY mode, defaults are used for all prompts."""
        console = Console(file=None, force_terminal=False)
        with patch("med2glb.cli_wizard.is_interactive", return_value=False):
            config = run_carto_wizard(
                carto_study, Path("/fake/input"), console,
            )
        assert config.coloring == "lat"
        assert config.animate is True
        assert config.static is True
        assert config.vectors is False
        assert config.subdivide == 2


class TestRunDicomWizard:
    def test_fully_preset(self, series_list):
        console = Console(file=None, force_terminal=False)
        config = run_dicom_wizard(
            series_list, Path("/fake/input"), console,
            preset_method="marching-cubes",
            preset_animate=True,
            preset_series="1.2.3.4",
            preset_quality="high",
        )
        assert isinstance(config, DicomConfig)
        assert config.method == "marching-cubes"
        assert config.animate is True
        assert config.smoothing == 25
        assert config.target_faces == 150000

    def test_non_interactive_defaults(self, series_list):
        console = Console(file=None, force_terminal=False)
        with patch("med2glb.cli_wizard.is_interactive", return_value=False):
            config = run_dicom_wizard(
                series_list, Path("/fake/input"), console,
            )
        # Should auto-select first series and use its recommended method
        assert config.method == "classical"
        assert config.animate is False
        assert config.smoothing == 15
        assert config.target_faces == 80000

    def test_quality_presets(self, series_list):
        console = Console(file=None, force_terminal=False)
        for quality, expected_smooth, expected_faces in [
            ("draft", 5, 40000),
            ("standard", 15, 80000),
            ("high", 25, 150000),
        ]:
            config = run_dicom_wizard(
                series_list, Path("/fake/input"), console,
                preset_quality=quality,
                preset_method="classical",
            )
            assert config.smoothing == expected_smooth
            assert config.target_faces == expected_faces
