"""Unit tests for compute_wavefront_colors() — T063."""

from __future__ import annotations

import numpy as np
import pytest

from med2glb.glb.carto_builder import compute_wavefront_colors


class TestWavefrontBasic:
    """Basic correctness checks."""

    def test_output_shape(self):
        n_verts, n_frames = 100, 30
        lat = np.linspace(0, 1, n_verts)
        base = np.tile([0.8, 0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)
        assert result.shape == (n_frames, n_verts, 4)
        assert result.dtype == np.float32

    def test_nan_vertices_unchanged(self):
        """Vertices with NaN LAT should stay at base color across all frames."""
        n_verts, n_frames = 50, 10
        lat = np.linspace(0, 1, n_verts)
        lat[0] = np.nan
        lat[25] = np.nan
        lat[49] = np.nan
        base = np.random.RandomState(42).rand(n_verts, 4).astype(np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)

        for f in range(n_frames):
            np.testing.assert_array_equal(result[f, 0, :], base[0, :])
            np.testing.assert_array_equal(result[f, 25, :], base[25, :])
            np.testing.assert_array_equal(result[f, 49, :], base[49, :])

    def test_alpha_preserved(self):
        """Alpha channel should always match the base color alpha."""
        n_verts, n_frames = 80, 15
        lat = np.linspace(0, 1, n_verts)
        base = np.random.RandomState(7).rand(n_verts, 4).astype(np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)
        for f in range(n_frames):
            np.testing.assert_array_almost_equal(result[f, :, 3], base[:, 3])

    def test_output_in_01_range(self):
        """All output values must be in [0, 1]."""
        n_verts, n_frames = 200, 30
        lat = np.linspace(0, 1, n_verts)
        base = np.random.RandomState(99).rand(n_verts, 4).astype(np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


class TestWavefrontCoverage:
    """Verify that the wavefront covers ~25-30% of the mesh surface."""

    def test_coverage_per_frame(self):
        """At each frame, a meaningful fraction of vertices should be lit."""
        n_verts = 1000
        n_frames = 30
        lat = np.linspace(0, 1, n_verts)
        base = np.full((n_verts, 4), [0.5, 0.0, 0.0, 1.0], dtype=np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)

        for f in range(n_frames):
            # "Lit" = color differs meaningfully from base (L2 norm > 0.05)
            diff = np.linalg.norm(result[f, :, :3] - base[:, :3], axis=1)
            lit_fraction = np.mean(diff > 0.05)
            # Real CARTO3 shows ~25-30% coverage; allow 15-45% for tolerance
            assert 0.15 < lit_fraction < 0.50, (
                f"Frame {f}: lit fraction {lit_fraction:.2%} outside expected range"
            )


class TestWavefrontSeamless:
    """Verify seamless looping at frame boundaries."""

    def test_first_last_frame_continuity(self):
        """Frame 0 and frame N-1 should be similar (seamless loop)."""
        n_verts = 500
        n_frames = 30
        lat = np.linspace(0, 1, n_verts)
        base = np.full((n_verts, 4), [0.3, 0.3, 0.3, 1.0], dtype=np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)

        # Consecutive frame deltas (including wrap-around)
        deltas = []
        for f in range(n_frames):
            f_next = (f + 1) % n_frames
            d = np.mean(np.abs(result[f_next] - result[f]))
            deltas.append(d)

        # The wrap-around delta (last→first) should be comparable to other deltas
        wrap_delta = deltas[-1]
        other_deltas = deltas[:-1]
        mean_delta = np.mean(other_deltas)
        # Wrap delta should be within 3x of the average inter-frame delta
        assert wrap_delta < mean_delta * 3.0, (
            f"Wrap-around delta {wrap_delta:.4f} much larger than mean "
            f"inter-frame delta {mean_delta:.4f}"
        )

    def test_all_vertices_activated(self):
        """Over one full cycle, every mapped vertex should be significantly
        brightened at some point."""
        n_verts = 200
        n_frames = 30
        lat = np.linspace(0, 1, n_verts)
        base = np.full((n_verts, 4), [0.2, 0.0, 0.0, 1.0], dtype=np.float32)
        result = compute_wavefront_colors(lat, base, n_frames)

        max_brightness = np.max(
            np.linalg.norm(result[:, :, :3] - base[:, :3], axis=2), axis=0
        )
        assert np.all(max_brightness > 0.1), "Some vertices never activated"
