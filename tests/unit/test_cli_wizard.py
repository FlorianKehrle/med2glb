"""Tests for cli_wizard.py — input analysis, wizard logic, configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rich.console import Console

from med2glb.cli_wizard import (
    _assess_vector_quality,
    _check_ai_available,
    _parse_mesh_selection,
    run_batch_carto_wizard,
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
            preset_vectors="yes",
            preset_subdivide=1,
            preset_meshes="all",
        )
        assert isinstance(config, CartoConfig)
        assert config.coloring == "bipolar"
        assert config.animate is True
        assert config.static is False
        assert config.vectors == "yes"
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
        assert config.vectors == "no"
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


class TestAssessVectorQuality:
    def test_sparse_points_not_suitable(self, carto_study):
        """With only 2 points, vectors should not be suitable."""
        quality = _assess_vector_quality(carto_study, None)
        assert quality.suitable is False
        assert "sparse" in quality.reason
        assert quality.valid_points == 2

    def test_no_points_not_suitable(self):
        """Mesh with no points at all."""
        mesh = CartoMesh(
            mesh_id=1,
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            normals=np.array([[0, 0, 1]] * 3, dtype=np.float64),
            group_ids=np.zeros(3, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1),
            color_names=["LAT"],
            structure_name="Empty",
        )
        study = CartoStudy(meshes=[mesh], points={}, version="6.0", study_name="Test")
        quality = _assess_vector_quality(study, None)
        assert quality.suitable is False
        assert quality.valid_points == 0

    def test_small_lat_range_not_suitable(self):
        """Many points but LAT range too small."""
        vertices = np.random.rand(100, 3).astype(np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.tile([0, 0, 1], (100, 1)).astype(np.float64)
        mesh = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(100, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="NarrowLAT",
        )
        # 60 points but all LAT between 0 and 10 ms (range < 30)
        points = [
            CartoPoint(i, np.random.rand(3), np.zeros(3), 1.0, 5.0, float(i % 10))
            for i in range(60)
        ]
        study = CartoStudy(
            meshes=[mesh], points={"NarrowLAT": points},
            version="6.0", study_name="Test",
        )
        quality = _assess_vector_quality(study, None)
        assert quality.suitable is False
        assert "range" in quality.reason

    def test_low_iqr_not_suitable(self):
        """Wide range but clustered LAT values → low IQR → not suitable."""
        vertices = np.random.rand(100, 3).astype(np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.tile([0, 0, 1], (100, 1)).astype(np.float64)
        mesh = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(100, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="ClusteredMap",
        )
        # 80 points: most clustered around 50 ms, a few outliers for wide range
        rng = np.random.default_rng(42)
        points = [
            CartoPoint(i, rng.random(3), np.zeros(3), 1.0, 5.0,
                       50.0 + rng.normal(0, 5))  # tight cluster around 50
            for i in range(76)
        ] + [
            CartoPoint(77, rng.random(3), np.zeros(3), 1.0, 5.0, -100.0),
            CartoPoint(78, rng.random(3), np.zeros(3), 1.0, 5.0, 200.0),
            CartoPoint(79, rng.random(3), np.zeros(3), 1.0, 5.0, -90.0),
            CartoPoint(80, rng.random(3), np.zeros(3), 1.0, 5.0, 190.0),
        ]
        study = CartoStudy(
            meshes=[mesh], points={"ClusteredMap": points},
            version="6.0", study_name="Test",
        )
        quality = _assess_vector_quality(study, None)
        assert quality.suitable is False
        assert "IQR" in quality.reason
        assert quality.lat_iqr_ms < 50

    def test_low_density_not_suitable(self):
        """Good LAT stats but extremely sparse sampling on a huge mesh."""
        # Very large triangle (~500,000 mm² area) with few points → density < 0.005
        vertices = np.array([
            [0, 0, 0], [1000, 0, 0], [500, 1000, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.tile([0, 0, 1], (3, 1)).astype(np.float64)
        mesh = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(3, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="SparseMap",
        )
        # 80 points on ~500k mm² mesh → density ~0.00016
        rng = np.random.default_rng(42)
        points = [
            CartoPoint(i, rng.random(3) * 1000, np.zeros(3), 1.0, 5.0,
                       float(i * 2.0))
            for i in range(80)
        ]
        study = CartoStudy(
            meshes=[mesh], points={"SparseMap": points},
            version="6.0", study_name="Test",
        )
        quality = _assess_vector_quality(study, None)
        assert quality.suitable is False
        assert "density" in quality.reason
        assert quality.point_density < 0.001

    def test_good_data_suitable(self):
        """Enough points with sufficient LAT range."""
        vertices = np.random.rand(100, 3).astype(np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.tile([0, 0, 1], (100, 1)).astype(np.float64)
        mesh = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(100, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="GoodMap",
        )
        # 80 points with LAT range 0..120 ms
        points = [
            CartoPoint(i, np.random.rand(3), np.zeros(3), 1.0, 5.0, float(i * 1.5))
            for i in range(80)
        ]
        study = CartoStudy(
            meshes=[mesh], points={"GoodMap": points},
            version="6.0", study_name="Test",
        )
        quality = _assess_vector_quality(study, None)
        assert quality.suitable is True
        assert quality.valid_points == 80
        assert quality.lat_range_ms > 30

    def test_selected_indices_filters_meshes(self):
        """Only checks selected meshes, not all."""
        vertices = np.random.rand(10, 3).astype(np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.tile([0, 0, 1], (10, 1)).astype(np.float64)
        base = dict(faces=faces, normals=normals,
                    group_ids=np.zeros(10, dtype=np.int32),
                    face_group_ids=np.zeros(1, dtype=np.int32),
                    mesh_color=(1, 0, 0, 1), color_names=["LAT"])
        mesh_good = CartoMesh(mesh_id=1, vertices=vertices, structure_name="Good", **base)
        mesh_bad = CartoMesh(mesh_id=2, vertices=vertices, structure_name="Bad", **base)

        good_pts = [
            CartoPoint(i, np.random.rand(3), np.zeros(3), 1.0, 5.0, float(i * 2.0))
            for i in range(80)
        ]
        bad_pts = [CartoPoint(1, np.array([0, 0, 0.0]), np.zeros(3), 1.0, 5.0, 0.0)]

        study = CartoStudy(
            meshes=[mesh_good, mesh_bad],
            points={"Good": good_pts, "Bad": bad_pts},
            version="6.0", study_name="Test",
        )
        # Selecting only the bad mesh
        quality = _assess_vector_quality(study, [1])
        assert quality.suitable is False

        # Selecting all — overall suitable because Good passes, but only [0] in suitable_indices
        quality_all = _assess_vector_quality(study, None)
        assert quality_all.suitable is True
        assert quality_all.suitable_indices == [0]


class TestRunBatchCartoWizard:
    def _make_study(self, name: str, n_points: int = 100, lat_range: float = 200.0) -> CartoStudy:
        """Create a minimal CartoStudy for testing."""
        vertices = np.random.RandomState(42).randn(50, 3).astype(np.float64) * 30
        faces = np.array([[0, 1, 2]] * 40, dtype=np.int32)
        normals = np.zeros_like(vertices)
        group_ids = np.zeros(len(vertices), dtype=np.int32)
        face_group_ids = np.zeros(len(faces), dtype=np.int32)

        mesh = CartoMesh(
            mesh_id=1,
            vertices=vertices,
            faces=faces,
            normals=normals,
            group_ids=group_ids,
            face_group_ids=face_group_ids,
            mesh_color=(0.0, 1.0, 0.0, 1.0),
            color_names=["LAT"],
            structure_name=name,
        )

        rng = np.random.RandomState(42)
        points = [
            CartoPoint(
                i, rng.randn(3) * 20, rng.randn(3),
                rng.uniform(0.1, 5.0), rng.uniform(1.0, 15.0),
                rng.uniform(-lat_range / 2, lat_range / 2),
            )
            for i in range(n_points)
        ]

        return CartoStudy(
            meshes=[mesh],
            points={name: points},
            version="6.0",
            study_name=name,
        )

    def test_returns_one_config_per_study(self, tmp_path):
        """Should return one CartoConfig per input study."""
        studies = [
            (tmp_path / "a", self._make_study("mesh_a")),
            (tmp_path / "b", self._make_study("mesh_b")),
        ]
        console = Console(file=None, force_terminal=False)

        configs = run_batch_carto_wizard(
            studies, console,
            preset_coloring="lat",
            preset_animate=True,
            preset_static=True,
            preset_vectors="no",
            preset_subdivide=2,
        )

        assert len(configs) == 2
        assert configs[0].input_path == tmp_path / "a"
        assert configs[1].input_path == tmp_path / "b"

    def test_shared_settings_applied(self, tmp_path):
        """All configs should share the same coloring/subdivide/animate settings."""
        studies = [
            (tmp_path / "a", self._make_study("mesh_a")),
            (tmp_path / "b", self._make_study("mesh_b")),
        ]
        console = Console(file=None, force_terminal=False)

        configs = run_batch_carto_wizard(
            studies, console,
            preset_coloring="bipolar",
            preset_animate=False,
            preset_static=True,
            preset_vectors="no",
            preset_subdivide=1,
        )

        for cfg in configs:
            assert cfg.coloring == "bipolar"
            assert cfg.subdivide == 1
            assert cfg.animate is False
            assert cfg.static is True

    def test_all_meshes_selected(self, tmp_path):
        """Batch mode should select all meshes (None = all)."""
        studies = [
            (tmp_path / "a", self._make_study("mesh_a")),
        ]
        console = Console(file=None, force_terminal=False)

        configs = run_batch_carto_wizard(
            studies, console,
            preset_coloring="lat",
            preset_animate=True,
            preset_static=True,
            preset_vectors="no",
            preset_subdivide=2,
        )

        assert configs[0].selected_mesh_indices is None

    def test_vectors_skipped_for_sparse_study(self, tmp_path):
        """Study with very few points should have vectors set to 'no'."""
        sparse_study = self._make_study("sparse", n_points=5, lat_range=10.0)
        studies = [
            (tmp_path / "sparse", sparse_study),
        ]
        console = Console(file=None, force_terminal=False)

        configs = run_batch_carto_wizard(
            studies, console,
            preset_coloring="lat",
            preset_animate=True,
            preset_static=True,
            preset_vectors="yes",
            preset_subdivide=2,
        )

        assert configs[0].vectors == "no"
