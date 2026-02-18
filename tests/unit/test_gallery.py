"""Tests for gallery mode: loader, individual, lightbox, and spatial builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import GallerySlice
from med2glb.gallery.individual import build_individual_glbs
from med2glb.gallery.lightbox import build_lightbox_glb
from med2glb.gallery.loader import load_all_slices
from med2glb.gallery.spatial import build_spatial_glb


# ─── Loader Tests ──────────────────────────────────────────────

class TestLoadAllSlices:
    def test_preserves_mixed_dimensions(self, dicom_gallery_mixed_directory):
        """All 6 files should be loaded — no shape filtering."""
        slices = load_all_slices(dicom_gallery_mixed_directory)
        assert len(slices) == 6
        dims = {(s.rows, s.cols) for s in slices}
        assert (32, 32) in dims
        assert (64, 64) in dims

    def test_sorted_by_instance(self, dicom_directory):
        """Slices should be sorted by instance_number."""
        slices = load_all_slices(dicom_directory)
        numbers = [s.instance_number for s in slices]
        assert numbers == sorted(numbers)

    def test_extracts_spatial_metadata(self, dicom_directory):
        """Each slice should have position and orientation metadata."""
        slices = load_all_slices(dicom_directory)
        for sl in slices:
            assert sl.image_position is not None
            assert sl.image_orientation is not None
            assert len(sl.image_orientation) == 6

    def test_extracts_temporal_index(self, dicom_temporal_gallery_directory):
        """Temporal DICOM files should have temporal_index populated."""
        slices = load_all_slices(dicom_temporal_gallery_directory)
        temporal_indices = {s.temporal_index for s in slices}
        assert temporal_indices == {1, 2}


# ─── Individual GLB Tests ──────────────────────────────────────

class TestBuildIndividualGlbs:
    def test_creates_all_files(self, dicom_directory, tmp_path):
        """One GLB file per slice."""
        slices = load_all_slices(dicom_directory)
        out_dir = tmp_path / "individual"
        paths = build_individual_glbs(slices, out_dir, animate=False)
        assert len(paths) == len(slices)
        for p in paths:
            assert p.exists()
            assert p.suffix == ".glb"

    def test_files_are_valid_glb(self, dicom_directory, tmp_path):
        """Each output should be a parseable GLB."""
        slices = load_all_slices(dicom_directory)
        out_dir = tmp_path / "individual"
        paths = build_individual_glbs(slices, out_dir, animate=False)
        gltf = pygltflib.GLTF2.load(str(paths[0]))
        assert len(gltf.meshes) == 1
        assert len(gltf.nodes) == 1

    def test_animated_creates_files(self, dicom_temporal_gallery_directory, tmp_path):
        """Animated individual GLBs should be created for temporal data."""
        slices = load_all_slices(dicom_temporal_gallery_directory)
        out_dir = tmp_path / "individual_anim"
        paths = build_individual_glbs(slices, out_dir, animate=True)
        # 3 spatial positions → 3 GLB files
        assert len(paths) == 3
        gltf = pygltflib.GLTF2.load(str(paths[0]))
        # Should have animation
        assert len(gltf.animations) == 1


# ─── Lightbox Tests ────────────────────────────────────────────

class TestBuildLightboxGlb:
    def test_creates_valid_glb(self, dicom_directory, tmp_path):
        """Lightbox output should be a valid GLB."""
        slices = load_all_slices(dicom_directory)
        out = tmp_path / "lightbox.glb"
        build_lightbox_glb(slices, out, columns=6, animate=False)
        assert out.exists()
        gltf = pygltflib.GLTF2.load(str(out))
        assert len(gltf.nodes) == len(slices)
        assert len(gltf.textures) == len(slices)

    def test_grid_positions(self, dicom_directory, tmp_path):
        """Nodes should be placed in a 6-column grid layout."""
        slices = load_all_slices(dicom_directory)
        out = tmp_path / "lightbox_grid.glb"
        build_lightbox_glb(slices, out, columns=3, animate=False)
        gltf = pygltflib.GLTF2.load(str(out))
        # First node at (0, 0, 0), second at (step_x, 0, 0), etc.
        translations = [n.translation for n in gltf.nodes]
        # All should have translation set
        for t in translations:
            assert t is not None
        # First row: cols 0, 1, 2 should have same Y
        assert translations[0][1] == translations[1][1] == translations[2][1]
        # Second row should have different Y
        if len(translations) > 3:
            assert translations[3][1] < translations[0][1]

    def test_animated(self, dicom_temporal_gallery_directory, tmp_path):
        """Animated lightbox should have animation channels."""
        slices = load_all_slices(dicom_temporal_gallery_directory)
        out = tmp_path / "lightbox_anim.glb"
        build_lightbox_glb(slices, out, columns=3, animate=True)
        gltf = pygltflib.GLTF2.load(str(out))
        assert len(gltf.animations) == 1
        # Should have channels for all temporal frame nodes
        assert len(gltf.animations[0].channels) > 0


# ─── Spatial Tests ─────────────────────────────────────────────

class TestBuildSpatialGlb:
    def test_with_positions(self, dicom_directory, tmp_path):
        """Spatial GLB should use matrix transforms from DICOM positions."""
        slices = load_all_slices(dicom_directory)
        out = tmp_path / "spatial.glb"
        build_spatial_glb(slices, out, animate=False)
        assert out.exists()
        gltf = pygltflib.GLTF2.load(str(out))
        # Nodes should have matrix transforms
        for node in gltf.nodes:
            assert node.matrix is not None

    def test_skips_without_positions(self, tmp_path):
        """Without position metadata, should skip and return False."""
        slices = [
            GallerySlice(
                pixel_data=np.zeros((32, 32), dtype=np.float32),
                pixel_spacing=(1.0, 1.0),
                image_position=None,
                image_orientation=None,
                instance_number=i,
                filename=f"test_{i}.dcm",
                rows=32,
                cols=32,
                temporal_index=None,
                series_uid="1.2.3",
                modality="CT",
            )
            for i in range(4)
        ]
        out = tmp_path / "spatial_skip.glb"
        result = build_spatial_glb(slices, out, animate=False)
        assert result is False
        assert not out.exists()

    def test_animated(self, dicom_temporal_gallery_directory, tmp_path):
        """Spatial animated GLB should have animation channels."""
        slices = load_all_slices(dicom_temporal_gallery_directory)
        out = tmp_path / "spatial_anim.glb"
        build_spatial_glb(slices, out, animate=True)
        gltf = pygltflib.GLTF2.load(str(out))
        assert len(gltf.animations) == 1
        assert len(gltf.animations[0].channels) > 0
