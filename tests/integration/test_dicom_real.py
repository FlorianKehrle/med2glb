"""Integration tests for real DICOM data.

Uses actual DICOM files from test_data/DICOM/ when available.
Tests skip automatically if the referenced data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MethodParams
from med2glb.io.dicom_reader import InputType
from med2glb.methods.registry import _ensure_methods_loaded, get_method

_REPO = Path(__file__).parent.parent.parent
DICOM_DATA = _REPO / "test_data" / "DICOM"

# Persistent output directory for inspecting GLBs after test runs
_GLB_OUTPUT = _REPO / "test_output" / "dicom"

# Transplant patient — 97-slice CT volume (512x512, 1.27mm spacing, 3mm thickness)
TRANSPLANT_CT = DICOM_DATA / "Transplant" / "BPL-CT"

# Sehnenfadenabriss — multi-frame ultrasound (GE PACS format)
SEHNE_DIR = DICOM_DATA / "Sehnenfadenabriss" / "GEMS_IMG"


@pytest.fixture(autouse=True)
def _load_methods():
    _ensure_methods_loaded()


@pytest.fixture
def transplant_output() -> Path:
    """Persistent output for Transplant CT GLBs."""
    d = _GLB_OUTPUT / "transplant-ct"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sehne_output() -> Path:
    """Persistent output for Sehnenfadenabriss US GLBs."""
    d = _GLB_OUTPUT / "sehnenfadenabriss-us"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Transplant CT (3D volume)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not TRANSPLANT_CT.exists() or not list(TRANSPLANT_CT.glob("*.dcm")),
    reason="Transplant CT data not available",
)
class TestTransplantCT:
    def test_analyze_series(self):
        """Series analysis should detect a single 3D volume series."""
        from med2glb.io.dicom_reader import analyze_series

        series = analyze_series(TRANSPLANT_CT)
        assert len(series) >= 1

        # Should be classified as a 3D volume
        vol_series = [s for s in series if "3D" in s.data_type or "volume" in s.data_type]
        assert len(vol_series) >= 1, f"Expected 3D volume, got: {[s.data_type for s in series]}"
        assert vol_series[0].modality == "CT"

    def test_load_volume(self):
        """Loading should produce a valid 3D volume."""
        from med2glb.io.dicom_reader import InputType, load_dicom_directory

        input_type, volume = load_dicom_directory(TRANSPLANT_CT)
        assert input_type == InputType.VOLUME
        assert volume.voxels.ndim == 3
        assert volume.voxels.shape[0] >= 90  # ~97 slices
        assert volume.voxels.shape[1] == 512
        assert volume.voxels.shape[2] == 512

    def test_classical_method(self, transplant_output):
        """Classical method (Gaussian + adaptive threshold) on real CT."""
        from med2glb.io.dicom_reader import load_dicom_directory

        _, volume = load_dicom_directory(TRANSPLANT_CT)
        method = get_method("classical")
        params = MethodParams(smoothing_iterations=5, target_faces=20000)
        result = method.convert(volume, params)

        assert len(result.meshes) >= 1
        assert result.meshes[0].vertices.shape[0] > 100

        from med2glb.glb.builder import build_glb
        output = transplant_output / "transplant-ct_classical_s5_20k.glb"
        build_glb(result.meshes, output)
        assert output.exists()
        assert output.stat().st_size > 1000

    def test_marching_cubes_method(self, transplant_output):
        """Marching cubes method on real CT."""
        from med2glb.io.dicom_reader import load_dicom_directory
        from med2glb.mesh.processing import process_mesh

        _, volume = load_dicom_directory(TRANSPLANT_CT)
        method = get_method("marching-cubes")
        params = MethodParams(threshold=200.0, smoothing_iterations=5, target_faces=20000)
        result = method.convert(volume, params)

        assert len(result.meshes) >= 1

        # Process and export
        processed = [process_mesh(m, smoothing_iterations=5, target_faces=20000) for m in result.meshes]
        from med2glb.glb.builder import build_glb
        output = transplant_output / "transplant-ct_mc_t200_s5_20k.glb"
        build_glb(processed, output)
        assert output.exists()

        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.meshes) >= 1

    def test_marching_cubes_multi_threshold(self, transplant_output):
        """Multi-threshold marching cubes on real CT (bone + tissue)."""
        from med2glb.io.dicom_reader import load_dicom_directory
        from med2glb.mesh.processing import process_mesh

        _, volume = load_dicom_directory(TRANSPLANT_CT)
        method = get_method("marching-cubes")

        # Two thresholds: soft tissue and bone
        thresholds = [
            (100.0, "tissue", 0.5),
            (300.0, "bone", 1.0),
        ]
        all_meshes = []
        for threshold, label, alpha in thresholds:
            params = MethodParams(threshold=threshold, smoothing_iterations=3, target_faces=10000)
            result = method.convert(volume, params)
            for mesh in result.meshes:
                mesh.material.alpha = alpha
                all_meshes.append(mesh)

        assert len(all_meshes) >= 2

        from med2glb.glb.builder import build_glb
        output = transplant_output / "transplant-ct_mc_multi-t100-t300_s3_10k.glb"
        build_glb(all_meshes, output)
        assert output.exists()
        assert output.stat().st_size > 1000



# ---------------------------------------------------------------------------
# Sehnenfadenabriss (multi-frame ultrasound)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not SEHNE_DIR.exists(),
    reason="Sehnenfadenabriss ultrasound data not available",
)
class TestSehnenfadenabrissUS:
    def test_analyze_series(self):
        """Should detect multi-frame ultrasound cine data."""
        from med2glb.io.dicom_reader import analyze_series

        series = analyze_series(SEHNE_DIR)
        assert len(series) >= 1

        # Should be ultrasound with multiple frames
        us_series = [s for s in series if s.modality == "US"]
        assert len(us_series) >= 1, f"Expected US modality, got: {[s.modality for s in series]}"

        # At least one should be multi-frame (cine)
        cine = [s for s in us_series if s.is_multiframe and s.number_of_frames > 1]
        assert len(cine) >= 1, f"Expected multi-frame US, got frames: {[s.number_of_frames for s in us_series]}"

    def test_load_multiframe(self):
        """Loading should produce temporal or single-slice data from multi-frame US."""
        from med2glb.io.dicom_reader import load_dicom_directory

        input_type, data = load_dicom_directory(SEHNE_DIR)
        # Multi-frame US detected as temporal (3D+T) or single slice (2D cine)
        assert input_type in (InputType.TEMPORAL, InputType.SINGLE_SLICE)

    def test_textured_plane_glb(self, sehne_output):
        """Build a textured plane GLB from ultrasound data."""
        from med2glb.io.dicom_reader import load_dicom_directory

        input_type, data = load_dicom_directory(SEHNE_DIR)
        if input_type == InputType.SINGLE_SLICE:
            from med2glb.glb.texture import build_textured_plane_glb
            output = sehne_output / "sehnenfadenabriss_us_textured-plane.glb"
            build_textured_plane_glb(data, output)
            assert output.exists()
            assert output.stat().st_size > 100


# ---------------------------------------------------------------------------
# Directory scanner (mixed data detection)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not DICOM_DATA.exists(),
    reason="DICOM test data directory not available",
)
class TestDicomDirectoryScanner:
    def test_scan_finds_dicom_data(self):
        """scan_directory should find DICOM data in test_data/DICOM."""
        from med2glb.cli_wizard import scan_directory

        entries = scan_directory(DICOM_DATA)
        dicom_entries = [e for e in entries if e.kind == "dicom"]
        assert len(dicom_entries) >= 1, f"Expected DICOM entries, got: {[e.kind for e in entries]}"

    def test_analyze_transplant_input(self):
        """analyze_input should detect Transplant CT as DICOM."""
        from med2glb.cli_wizard import analyze_input

        detected = analyze_input(TRANSPLANT_CT)
        assert detected.kind == "dicom"
        assert detected.series_list is not None
        assert len(detected.series_list) >= 1
