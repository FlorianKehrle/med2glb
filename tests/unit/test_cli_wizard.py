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
    estimate_time,
    _mesh_bbox_mm,
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


class TestMeshBboxMm:
    def test_basic_bbox(self):
        mesh = CartoMesh(
            mesh_id=1,
            vertices=np.array([
                [0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30],
            ], dtype=np.float64),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            normals=np.zeros((4, 3), dtype=np.float64),
            group_ids=np.zeros(4, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1),
            color_names=["LAT"],
        )
        result = _mesh_bbox_mm(mesh)
        assert result == "10×20×30 mm"

    def test_excludes_inactive_vertices(self):
        mesh = CartoMesh(
            mesh_id=1,
            vertices=np.array([
                [0, 0, 0], [10, 0, 0], [0, 10, 0], [100, 100, 100],
            ], dtype=np.float64),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            normals=np.zeros((4, 3), dtype=np.float64),
            group_ids=np.array([0, 0, 0, -1000000], dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1),
            color_names=["LAT"],
        )
        result = _mesh_bbox_mm(mesh)
        assert result == "10×10×0 mm"


class TestEstimateTime:
    def test_small_mesh_seconds(self):
        result = estimate_time(1000, 10, has_lat=False)
        assert "s" in result

    def test_large_mesh_minutes(self):
        result = estimate_time(200_000, 2000, has_lat=True)
        assert "min" in result

    def test_no_points_no_vectors(self):
        # With 0 points and no LAT, should still return a time
        result = estimate_time(500, 0, has_lat=False)
        assert "s" in result or "min" in result


class TestRunCartoWizard:
    def test_fully_preset(self, carto_study):
        """When all presets are given, no prompts are needed."""
        console = Console(file=None, force_terminal=False)
        config = run_carto_wizard(
            carto_study, Path("/fake/input"), console,
            preset_colorings=["bipolar"],
            preset_animate=True,
            preset_static=False,
            preset_vectors="yes",
            preset_subdivide=1,
            preset_meshes="all",
        )
        assert isinstance(config, CartoConfig)
        assert config.colorings == ["bipolar"]
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
        assert config.colorings == ["lat", "bipolar", "unipolar"]
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

    def test_low_gradient_coverage_not_suitable(self):
        """Points with identical LAT values → zero gradients → not suitable."""
        # 4x4 grid mesh (25 verts, 32 faces)
        verts = []
        for j in range(5):
            for i in range(5):
                verts.append([float(i) * 10, float(j) * 10, 0.0])
        vertices = np.array(verts, dtype=np.float64)
        faces_list = []
        for j in range(4):
            for i in range(4):
                v0 = j * 5 + i
                faces_list.append([v0, v0 + 1, v0 + 6])
                faces_list.append([v0, v0 + 6, v0 + 5])
        faces = np.array(faces_list, dtype=np.int32)
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0
        mesh = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(len(vertices), dtype=np.int32),
            face_group_ids=np.zeros(len(faces), dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="FlatLAT",
        )
        # 40 points all with the same integer LAT (quantized) — IDW
        # will produce uniform values → zero gradient on all faces
        points = [
            CartoPoint(i, vertices[i % len(vertices)], np.zeros(3),
                       1.0, 5.0, 50.0)  # all same LAT
            for i in range(40)
        ]
        # Add a few outliers so range passes the cheap check
        points.append(CartoPoint(90, np.array([0.0, 0.0, 0.0]),
                                 np.zeros(3), 1.0, 5.0, -10.0))
        points.append(CartoPoint(91, np.array([40.0, 40.0, 0.0]),
                                 np.zeros(3), 1.0, 5.0, 100.0))
        study = CartoStudy(
            meshes=[mesh], points={"FlatLAT": points},
            version="6.0", study_name="Test",
        )
        quality = _assess_vector_quality(study, None)
        assert quality.suitable is False
        assert "gradient" in quality.reason

    def test_good_data_suitable(self):
        """Enough points with sufficient LAT range and gradient coverage."""
        # 4x4 grid mesh (25 verts, 32 faces) — proper geometry
        verts = []
        for j in range(5):
            for i in range(5):
                verts.append([float(i) * 10, float(j) * 10, 0.0])
        vertices = np.array(verts, dtype=np.float64)
        faces_list = []
        for j in range(4):
            for i in range(4):
                v0 = j * 5 + i
                faces_list.append([v0, v0 + 1, v0 + 6])
                faces_list.append([v0, v0 + 6, v0 + 5])
        faces = np.array(faces_list, dtype=np.int32)
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0
        mesh = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(len(vertices), dtype=np.int32),
            face_group_ids=np.zeros(len(faces), dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="GoodMap",
        )
        # 80 points with smooth LAT gradient in X direction, positioned on the mesh
        rng = np.random.default_rng(42)
        points = [
            CartoPoint(i, np.array([rng.uniform(0, 40), rng.uniform(0, 40), 0.0]),
                       np.zeros(3), 1.0, 5.0,
                       float(rng.uniform(0, 40) * 3.0))  # LAT 0..120 ms
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
        # Build a proper grid mesh for the "good" map
        verts = []
        for j in range(5):
            for i in range(5):
                verts.append([float(i) * 10, float(j) * 10, 0.0])
        vertices = np.array(verts, dtype=np.float64)
        faces_list = []
        for j in range(4):
            for i in range(4):
                v0 = j * 5 + i
                faces_list.append([v0, v0 + 1, v0 + 6])
                faces_list.append([v0, v0 + 6, v0 + 5])
        faces = np.array(faces_list, dtype=np.int32)
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0

        mesh_good = CartoMesh(
            mesh_id=1, vertices=vertices, faces=faces, normals=normals,
            group_ids=np.zeros(len(vertices), dtype=np.int32),
            face_group_ids=np.zeros(len(faces), dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="Good",
        )
        # Tiny single-face mesh for the "bad" map
        bad_verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        bad_faces = np.array([[0, 1, 2]], dtype=np.int32)
        bad_normals = np.tile([0, 0, 1], (3, 1)).astype(np.float64)
        mesh_bad = CartoMesh(
            mesh_id=2, vertices=bad_verts, faces=bad_faces, normals=bad_normals,
            group_ids=np.zeros(3, dtype=np.int32),
            face_group_ids=np.zeros(1, dtype=np.int32),
            mesh_color=(1, 0, 0, 1), color_names=["LAT"],
            structure_name="Bad",
        )

        rng = np.random.default_rng(42)
        good_pts = [
            CartoPoint(i, np.array([rng.uniform(0, 40), rng.uniform(0, 40), 0.0]),
                       np.zeros(3), 1.0, 5.0, float(rng.uniform(0, 120)))
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
            preset_colorings=["lat"],
            preset_animate=True,
            preset_static=True,
            preset_vectors="no",
            preset_subdivide=2,
        )

        assert len(configs) == 2
        assert configs[0].input_path == tmp_path / "a"
        assert configs[1].input_path == tmp_path / "b"

    def test_shared_settings_applied(self, tmp_path):
        """All configs should share the same colorings/subdivide/animate settings."""
        studies = [
            (tmp_path / "a", self._make_study("mesh_a")),
            (tmp_path / "b", self._make_study("mesh_b")),
        ]
        console = Console(file=None, force_terminal=False)

        configs = run_batch_carto_wizard(
            studies, console,
            preset_colorings=["bipolar"],
            preset_animate=False,
            preset_static=True,
            preset_vectors="no",
            preset_subdivide=1,
        )

        for cfg in configs:
            assert cfg.colorings == ["bipolar"]
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
            preset_colorings=["lat"],
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
            preset_colorings=["lat"],
            preset_animate=True,
            preset_static=True,
            preset_vectors="yes",
            preset_subdivide=2,
        )

        assert configs[0].vectors == "no"


# ---------------------------------------------------------------------------
# Tests for _utils.py fmt_duration
# ---------------------------------------------------------------------------


class TestFmtDuration:
    def test_seconds_only(self):
        from med2glb._utils import fmt_duration
        assert fmt_duration(5.3) == "5.3s"
        assert fmt_duration(0.0) == "0.0s"
        assert fmt_duration(59.9) == "59.9s"

    def test_minutes_and_seconds(self):
        from med2glb._utils import fmt_duration
        assert fmt_duration(65) == "1m 5s"
        assert fmt_duration(3599) == "59m 59s"

    def test_hours(self):
        from med2glb._utils import fmt_duration
        assert fmt_duration(3661) == "1h 1m 1s"
        assert fmt_duration(7200) == "2h 0m 0s"


# ---------------------------------------------------------------------------
# Tests for equivalent command builders
# ---------------------------------------------------------------------------


class TestBuildCartoEquivCommand:
    def test_basic_carto_command(self):
        from med2glb.cli_wizard import build_carto_equiv_command

        config = CartoConfig(
            input_path=Path("/data/carto"),
            colorings=["lat", "bipolar"],
            subdivide=2,
            animate=True,
            static=True,
            vectors="no",
        )
        cmd = build_carto_equiv_command(config)
        assert "med2glb" in cmd
        assert "--coloring lat" in cmd
        assert "--coloring bipolar" in cmd
        assert "--subdivide 2" in cmd
        # Both animate and static → default behavior, neither flag needed
        assert "--animate" not in cmd
        assert "--no-animate" not in cmd

    def test_animate_only(self):
        from med2glb.cli_wizard import build_carto_equiv_command

        config = CartoConfig(
            input_path=Path("/data/carto"),
            colorings=["lat"],
            subdivide=1,
            animate=True,
            static=False,
        )
        cmd = build_carto_equiv_command(config)
        assert "--animate" in cmd

    def test_static_only(self):
        from med2glb.cli_wizard import build_carto_equiv_command

        config = CartoConfig(
            input_path=Path("/data/carto"),
            colorings=["lat"],
            subdivide=1,
            animate=False,
            static=True,
        )
        cmd = build_carto_equiv_command(config)
        assert "--no-animate" in cmd

    def test_vectors(self):
        from med2glb.cli_wizard import build_carto_equiv_command

        config = CartoConfig(
            input_path=Path("/data/carto"),
            colorings=["lat"],
            subdivide=2,
            vectors="yes",
        )
        cmd = build_carto_equiv_command(config)
        assert "--vectors" in cmd

    def test_batch_mode(self):
        from med2glb.cli_wizard import build_carto_equiv_command

        config = CartoConfig(
            input_path=Path("/data/carto"),
            colorings=["lat"],
            subdivide=2,
        )
        cmd = build_carto_equiv_command(config, batch=True)
        assert "--batch" in cmd

    def test_output_path(self):
        from med2glb.cli_wizard import build_carto_equiv_command

        config = CartoConfig(
            input_path=Path("/data/carto"),
            colorings=["lat"],
            subdivide=2,
        )
        cmd = build_carto_equiv_command(config, output_path=Path("/out/result.glb"))
        assert "-o" in cmd


class TestBuildDicomEquivCommand:
    def test_basic_dicom_command(self):
        from med2glb.cli_wizard import build_dicom_equiv_command

        config = DicomConfig(
            input_path=Path("/data/dicom"),
            method="classical",
        )
        cmd = build_dicom_equiv_command(config)
        assert "med2glb" in cmd
        assert "--method classical" in cmd
        # Defaults not included
        assert "--smoothing" not in cmd
        assert "--faces" not in cmd

    def test_non_default_params(self):
        from med2glb.cli_wizard import build_dicom_equiv_command

        config = DicomConfig(
            input_path=Path("/data/dicom"),
            method="marching-cubes",
            smoothing=5,
            target_faces=50000,
            threshold=200.0,
            alpha=0.8,
            animate=True,
        )
        cmd = build_dicom_equiv_command(config)
        assert "--method marching-cubes" in cmd
        assert "--smoothing 5" in cmd
        assert "--faces 50000" in cmd
        assert "--threshold 200.0" in cmd
        assert "--alpha 0.8" in cmd
        assert "--animate" in cmd

    def test_series_uid(self):
        from med2glb.cli_wizard import build_dicom_equiv_command

        config = DicomConfig(
            input_path=Path("/data/dicom"),
            method="classical",
            series_uid="1.2.3.4",
        )
        cmd = build_dicom_equiv_command(config)
        assert "--series 1.2.3.4" in cmd

    def test_output_path(self):
        from med2glb.cli_wizard import build_dicom_equiv_command

        config = DicomConfig(
            input_path=Path("/data/dicom"),
            method="classical",
        )
        cmd = build_dicom_equiv_command(config, output_path=Path("/out/result.glb"))
        assert "-o" in cmd


# ---------------------------------------------------------------------------
# Tests for conversion log with equivalent command
# ---------------------------------------------------------------------------


class TestConversionLogEquivCommand:
    def test_carto_log_includes_equiv_command(self, tmp_path):
        from med2glb.io.conversion_log import append_carto_entry

        append_carto_entry(
            tmp_path,
            structure_name="LA",
            carto_version="7.2",
            study_name="Test",
            coloring="lat",
            color_range="lat",
            subdivide=2,
            active_vertices=1000,
            total_vertices=1200,
            face_count=2000,
            mapping_points=500,
            variant_outputs=[],
            elapsed_seconds=10.5,
            source_path="/data/carto",
            equivalent_command='med2glb "/data/carto" --coloring lat --subdivide 2',
        )

        log = (tmp_path / "med2glb_log.txt").read_text()
        assert "Equivalent command:" in log
        assert "--coloring lat" in log

    def test_carto_log_no_equiv_command(self, tmp_path):
        from med2glb.io.conversion_log import append_carto_entry

        append_carto_entry(
            tmp_path,
            structure_name="LA",
            carto_version="7.2",
            study_name="Test",
            coloring="lat",
            color_range="lat",
            subdivide=2,
            active_vertices=1000,
            total_vertices=1200,
            face_count=2000,
            mapping_points=500,
            variant_outputs=[],
            elapsed_seconds=10.5,
            source_path="/data/carto",
        )

        log = (tmp_path / "med2glb_log.txt").read_text()
        assert "Equivalent command:" not in log

    def test_dicom_log_includes_equiv_command(self, tmp_path):
        from med2glb.io.conversion_log import append_dicom_entry

        out_file = tmp_path / "test.glb"
        out_file.write_bytes(b"\x00" * 100)

        append_dicom_entry(
            tmp_path,
            output_path=out_file,
            method_name="classical",
            input_path="/data/dicom",
            input_type="3D volume",
            series_uid=None,
            mesh_count=1,
            vertex_count=500,
            face_count=1000,
            file_size_kb=0.1,
            animated=False,
            elapsed_seconds=5.2,
            equivalent_command='med2glb "/data/dicom" --method classical',
        )

        log = (tmp_path / "med2glb_log.txt").read_text()
        assert "Equivalent command:" in log
        assert "--method classical" in log


# ---------------------------------------------------------------------------
# Tests for SeriesInfo with new fields
# ---------------------------------------------------------------------------


class TestSeriesInfoNewFields:
    def test_default_none(self):
        info = SeriesInfo(
            series_uid="1.2.3",
            modality="CT",
            description="Test",
            file_count=10,
            data_type="3D volume",
            detail="10 slices",
            dimensions="512x512x10",
            recommended_method="classical",
            recommended_output="3D mesh",
        )
        assert info.spacing is None
        assert info.est_time is None

    def test_with_spacing_and_time(self):
        info = SeriesInfo(
            series_uid="1.2.3",
            modality="CT",
            description="Test",
            file_count=10,
            data_type="3D volume",
            detail="10 slices",
            dimensions="512x512x10",
            recommended_method="classical",
            recommended_output="3D mesh",
            spacing="0.5 × 0.5 × 1.0 mm",
            est_time="~3.3s",
        )
        assert info.spacing == "0.5 × 0.5 × 1.0 mm"
        assert info.est_time == "~3.3s"
