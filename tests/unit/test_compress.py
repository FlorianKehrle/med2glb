"""Tests for GLB file size constraint and compression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from med2glb.glb.compress import (
    constrain_glb_size,
    optimize_textures_ktx2,
    _has_toktx,
    _image_to_ktx2,
    _reencode_image,
)


def _build_textured_glb(path: Path, num_textures: int = 1, size: int = 128) -> None:
    """Build a simple textured GLB for testing compression."""
    from med2glb.glb.builder import _pad_to_4
    from med2glb.glb.texture import pixel_data_to_png

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=list(range(num_textures)))],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
        )],
    )
    binary_data = bytearray()

    # Shared quad geometry
    vertices = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)
    texcoords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1]] * 4, dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    # Write textures
    for i in range(num_textures):
        # Create a non-trivial PNG (random noise to prevent easy compression)
        rng = np.random.RandomState(42 + i)
        pixel_data = rng.randint(0, 255, (size, size), dtype=np.uint8)
        png_bytes = pixel_data_to_png(pixel_data.astype(np.float32))

        img_offset = len(binary_data)
        binary_data.extend(png_bytes)
        _pad_to_4(binary_data)

        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=img_offset, byteLength=len(png_bytes),
        ))
        gltf.images.append(pygltflib.Image(bufferView=bv_idx, mimeType="image/png"))
        gltf.textures.append(pygltflib.Texture(sampler=0, source=i))
        gltf.materials.append(pygltflib.Material(
            name=f"mat_{i}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=i),
                metallicFactor=0.0, roughnessFactor=1.0,
            ),
            doubleSided=True,
        ))

    # Write shared geometry
    for arr, target in [
        (vertices, pygltflib.ARRAY_BUFFER),
        (normals, pygltflib.ARRAY_BUFFER),
        (texcoords, pygltflib.ARRAY_BUFFER),
        (indices, pygltflib.ELEMENT_ARRAY_BUFFER),
    ]:
        off = len(binary_data)
        raw = arr.tobytes()
        binary_data.extend(raw)
        _pad_to_4(binary_data)
        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=off, byteLength=len(raw), target=target,
        ))

    pos_bv = num_textures
    norm_bv = num_textures + 1
    tc_bv = num_textures + 2
    idx_bv = num_textures + 3

    gltf.accessors.extend([
        pygltflib.Accessor(bufferView=pos_bv, componentType=pygltflib.FLOAT, count=4, type=pygltflib.VEC3,
                           max=vertices.max(axis=0).tolist(), min=vertices.min(axis=0).tolist()),
        pygltflib.Accessor(bufferView=norm_bv, componentType=pygltflib.FLOAT, count=4, type=pygltflib.VEC3),
        pygltflib.Accessor(bufferView=tc_bv, componentType=pygltflib.FLOAT, count=4, type=pygltflib.VEC2),
        pygltflib.Accessor(bufferView=idx_bv, componentType=pygltflib.UNSIGNED_SHORT, count=6,
                           type=pygltflib.SCALAR, max=[3], min=[0]),
    ])

    for i in range(num_textures):
        gltf.meshes.append(pygltflib.Mesh(
            name=f"quad_{i}",
            primitives=[pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
                indices=3, material=i,
            )],
        ))
        gltf.nodes.append(pygltflib.Node(name=f"node_{i}", mesh=i))

    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(path))


def _build_animated_textured_glb(path: Path, num_frames: int = 3, size: int = 128) -> None:
    """Build a textured GLB with scale-toggle animation (like gallery lightbox)."""
    from med2glb.glb.builder import _pad_to_4, write_accessor
    from med2glb.glb.texture import pixel_data_to_png

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
        )],
        animations=[],
    )
    binary_data = bytearray()

    # Shared quad geometry
    vertices = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)
    texcoords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1]] * 4, dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    for arr, target in [
        (vertices, pygltflib.ARRAY_BUFFER),
        (normals, pygltflib.ARRAY_BUFFER),
        (texcoords, pygltflib.ARRAY_BUFFER),
        (indices, pygltflib.ELEMENT_ARRAY_BUFFER),
    ]:
        off = len(binary_data)
        raw = arr.tobytes()
        binary_data.extend(raw)
        _pad_to_4(binary_data)
        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=off, byteLength=len(raw), target=target,
        ))

    pos_bv, norm_bv, tc_bv, idx_bv = 0, 1, 2, 3
    gltf.accessors.extend([
        pygltflib.Accessor(bufferView=pos_bv, componentType=pygltflib.FLOAT, count=4, type=pygltflib.VEC3,
                           max=vertices.max(axis=0).tolist(), min=vertices.min(axis=0).tolist()),
        pygltflib.Accessor(bufferView=norm_bv, componentType=pygltflib.FLOAT, count=4, type=pygltflib.VEC3),
        pygltflib.Accessor(bufferView=tc_bv, componentType=pygltflib.FLOAT, count=4, type=pygltflib.VEC2),
        pygltflib.Accessor(bufferView=idx_bv, componentType=pygltflib.UNSIGNED_SHORT, count=6,
                           type=pygltflib.SCALAR, max=[3], min=[0]),
    ])

    # Create one textured node per frame
    node_indices = []
    for i in range(num_frames):
        rng = np.random.RandomState(42 + i)
        pixel_data = rng.randint(0, 255, (size, size), dtype=np.uint8)
        png_bytes = pixel_data_to_png(pixel_data.astype(np.float32))

        img_offset = len(binary_data)
        binary_data.extend(png_bytes)
        _pad_to_4(binary_data)

        img_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=img_offset, byteLength=len(png_bytes),
        ))
        gltf.images.append(pygltflib.Image(bufferView=img_bv_idx, mimeType="image/png"))
        gltf.textures.append(pygltflib.Texture(sampler=0, source=i))
        gltf.materials.append(pygltflib.Material(
            name=f"frame_{i}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=i),
                metallicFactor=0.0, roughnessFactor=1.0,
            ),
            doubleSided=True,
        ))

        mesh_idx = len(gltf.meshes)
        gltf.meshes.append(pygltflib.Mesh(
            name=f"frame_{i}",
            primitives=[pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
                indices=3, material=i,
            )],
        ))

        scale = [1.0, 1.0, 1.0] if i == 0 else [0.0, 0.0, 0.0]
        node_idx = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(name=f"frame_{i}", mesh=mesh_idx, scale=scale))
        gltf.scenes[0].nodes.append(node_idx)
        node_indices.append(node_idx)

    # Add scale-toggle animation
    dt = 0.0333
    keyframe_times = np.array([i * dt for i in range(num_frames)], dtype=np.float32)
    time_acc = write_accessor(gltf, binary_data, keyframe_times, None, pygltflib.FLOAT, pygltflib.SCALAR, True)

    channels = []
    samplers = []
    for i, node_idx in enumerate(node_indices):
        scales = np.zeros((num_frames, 3), dtype=np.float32)
        scales[i] = [1.0, 1.0, 1.0]
        s_acc = write_accessor(gltf, binary_data, scales, None, pygltflib.FLOAT, pygltflib.VEC3)
        sampler_idx = len(samplers)
        samplers.append(pygltflib.AnimationSampler(
            input=time_acc, output=s_acc, interpolation=pygltflib.ANIM_STEP,
        ))
        channels.append(pygltflib.AnimationChannel(
            sampler=sampler_idx,
            target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale"),
        ))

    gltf.animations.append(pygltflib.Animation(name="cycle", channels=channels, samplers=samplers))

    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))
    path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(path))


class TestConstrainGlbSize:
    """Tests for constrain_glb_size."""

    def test_no_compression_when_under_limit(self, tmp_path):
        """File under limit should not be modified."""
        path = tmp_path / "small.glb"
        _build_textured_glb(path, num_textures=1, size=32)
        original_size = path.stat().st_size

        result = constrain_glb_size(path, max_bytes=10 * 1024 * 1024, strategy="jpeg")
        assert result is False
        assert path.stat().st_size == original_size

    def test_jpeg_compression_reduces_size(self, tmp_path):
        """JPEG strategy should reduce file size."""
        path = tmp_path / "large.glb"
        _build_textured_glb(path, num_textures=5, size=256)
        original_size = path.stat().st_size

        # Set a limit just under the original size to force compression
        result = constrain_glb_size(path, max_bytes=original_size - 1, strategy="jpeg")
        assert result is True
        assert path.stat().st_size < original_size

    def test_downscale_compression_reduces_size(self, tmp_path):
        """Downscale strategy should reduce file size."""
        path = tmp_path / "large.glb"
        _build_textured_glb(path, num_textures=5, size=256)
        original_size = path.stat().st_size

        result = constrain_glb_size(path, max_bytes=original_size - 1, strategy="downscale")
        assert result is True
        assert path.stat().st_size < original_size

    def test_draco_falls_back_to_downscale(self, tmp_path):
        """Draco strategy should fall back to downscale for textured GLBs."""
        path = tmp_path / "large.glb"
        _build_textured_glb(path, num_textures=5, size=256)
        original_size = path.stat().st_size

        result = constrain_glb_size(path, max_bytes=original_size - 1, strategy="draco")
        assert result is True
        assert path.stat().st_size < original_size

    def test_compressed_glb_is_valid(self, tmp_path):
        """Compressed GLB should still be loadable."""
        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=3, size=128)
        original_size = path.stat().st_size

        constrain_glb_size(path, max_bytes=original_size - 1, strategy="jpeg")

        # Load and verify structure
        gltf = pygltflib.GLTF2.load(str(path))
        assert len(gltf.meshes) == 3
        assert len(gltf.images) == 3
        assert len(gltf.nodes) == 3
        assert gltf.binary_blob() is not None

    def test_nonexistent_file_returns_false(self, tmp_path):
        """Non-existent file should return False."""
        result = constrain_glb_size(tmp_path / "missing.glb", max_bytes=1024)
        assert result is False

    def test_invalid_strategy_raises(self, tmp_path):
        """Invalid strategy should raise ValueError."""
        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=1, size=32)
        with pytest.raises(ValueError, match="Unknown compression strategy"):
            constrain_glb_size(path, max_bytes=1, strategy="invalid")

    def test_draco_preserves_animations(self, tmp_path):
        """Draco strategy should not strip animations from animated GLBs."""
        path = tmp_path / "animated.glb"
        _build_animated_textured_glb(path, num_frames=3, size=256)
        original_size = path.stat().st_size

        gltf_before = pygltflib.GLTF2.load(str(path))
        assert len(gltf_before.animations) == 1
        channels_before = len(gltf_before.animations[0].channels)

        constrain_glb_size(path, max_bytes=original_size - 1, strategy="draco")

        gltf_after = pygltflib.GLTF2.load(str(path))
        assert len(gltf_after.animations) == 1
        assert len(gltf_after.animations[0].channels) == channels_before

    def test_jpeg_preserves_animations(self, tmp_path):
        """JPEG strategy should preserve animations."""
        path = tmp_path / "animated.glb"
        _build_animated_textured_glb(path, num_frames=3, size=256)
        original_size = path.stat().st_size

        constrain_glb_size(path, max_bytes=original_size - 1, strategy="jpeg")

        gltf = pygltflib.GLTF2.load(str(path))
        assert len(gltf.animations) == 1
        assert len(gltf.animations[0].channels) > 0

    def test_downscale_preserves_animations(self, tmp_path):
        """Downscale strategy should preserve animations."""
        path = tmp_path / "animated.glb"
        _build_animated_textured_glb(path, num_frames=3, size=256)
        original_size = path.stat().st_size

        constrain_glb_size(path, max_bytes=original_size - 1, strategy="downscale")

        gltf = pygltflib.GLTF2.load(str(path))
        assert len(gltf.animations) == 1
        assert len(gltf.animations[0].channels) > 0


class TestReencodeImage:
    """Tests for image re-encoding."""

    def test_jpeg_reencode(self):
        """PNG data should be re-encoded as JPEG."""
        from med2glb.glb.texture import pixel_data_to_png

        pixel_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        png_bytes = pixel_data_to_png(pixel_data.astype(np.float32))

        jpeg_bytes = _reencode_image(png_bytes, "JPEG", 80, 1.0)
        assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic number
        assert len(jpeg_bytes) < len(png_bytes)

    def test_downscale_reduces_resolution(self):
        """Downscaling should produce smaller PNG."""
        from med2glb.glb.texture import pixel_data_to_png

        pixel_data = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        png_bytes = pixel_data_to_png(pixel_data.astype(np.float32))

        scaled_bytes = _reencode_image(png_bytes, "PNG", 0, 0.5)
        # Verify it's valid PNG (starts with PNG magic)
        assert scaled_bytes[:4] == b"\x89PNG"
        assert len(scaled_bytes) < len(png_bytes)

    def test_invalid_data_returned_as_is(self):
        """Non-image data should be returned unchanged."""
        bad_data = b"not an image"
        result = _reencode_image(bad_data, "JPEG", 80, 1.0)
        assert result == bad_data


class TestKtx2Compression:
    """Tests for KTX2 / Basis Universal texture compression."""

    def test_has_toktx_available(self, monkeypatch):
        """_has_toktx returns True when toktx is on PATH."""
        import shutil
        import med2glb.glb.compress as compress_mod

        # Clear the lru_cache before patching
        compress_mod._has_toktx.cache_clear()
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/toktx" if name == "toktx" else None)
        assert _has_toktx() is True
        compress_mod._has_toktx.cache_clear()

    def test_has_toktx_unavailable(self, monkeypatch):
        """_has_toktx returns False when toktx is not installed."""
        import shutil
        import med2glb.glb.compress as compress_mod

        compress_mod._has_toktx.cache_clear()
        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert _has_toktx() is False
        compress_mod._has_toktx.cache_clear()

    def test_image_to_ktx2_fallback(self, monkeypatch):
        """When toktx is unavailable, _image_to_ktx2 returns None gracefully."""
        import shutil
        import med2glb.glb.compress as compress_mod

        compress_mod._has_toktx.cache_clear()
        monkeypatch.setattr(shutil, "which", lambda name: None)
        result = _image_to_ktx2(b"\x89PNG fake data")
        assert result is None
        compress_mod._has_toktx.cache_clear()

    def test_strategy_ktx2_dispatch(self, tmp_path):
        """constrain_glb_size recognizes the 'ktx2' strategy."""
        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=1, size=32)
        # Should not raise ValueError for the ktx2 strategy
        # (may return False if toktx is not installed, which is fine)
        result = constrain_glb_size(path, max_bytes=1, strategy="ktx2")
        assert isinstance(result, bool)

    def test_optimize_textures_ktx2_no_toktx(self, tmp_path, monkeypatch):
        """optimize_textures_ktx2 returns False when toktx is not installed."""
        import shutil
        import med2glb.glb.compress as compress_mod

        compress_mod._has_toktx.cache_clear()
        monkeypatch.setattr(shutil, "which", lambda name: None)

        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=1, size=64)
        original_size = path.stat().st_size

        result = optimize_textures_ktx2(path)
        assert result is False
        assert path.stat().st_size == original_size
        compress_mod._has_toktx.cache_clear()

    def test_optimize_textures_ktx2_nonexistent(self):
        """optimize_textures_ktx2 returns False for non-existent file."""
        result = optimize_textures_ktx2(Path("/nonexistent/file.glb"))
        assert result is False

    @pytest.mark.skipif(
        not __import__("shutil").which("toktx"),
        reason="toktx not installed",
    )
    def test_image_to_ktx2_real(self):
        """Integration: real PNG→KTX2 conversion with toktx."""
        from med2glb.glb.texture import pixel_data_to_png

        pixel_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        png_bytes = pixel_data_to_png(pixel_data.astype(np.float32))

        ktx2_bytes = _image_to_ktx2(png_bytes)
        assert ktx2_bytes is not None
        # KTX2 files start with the KTX2 identifier
        assert ktx2_bytes[:7] == b"\xabKTX 20"

    @pytest.mark.skipif(
        not __import__("shutil").which("toktx"),
        reason="toktx not installed",
    )
    def test_optimize_textures_ktx2_real(self, tmp_path):
        """Integration: optimize_textures_ktx2 converts PNG textures to KTX2."""
        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=2, size=64)
        original_size = path.stat().st_size

        result = optimize_textures_ktx2(path)
        assert result is True

        # Verify GLB is still valid and has KTX2 textures
        gltf = pygltflib.GLTF2.load(str(path))
        assert "KHR_texture_basisu" in gltf.extensionsUsed
        for img in gltf.images:
            assert img.mimeType == "image/ktx2"


# ---------------------------------------------------------------------------
# gltfpack + auto strategy tests
# ---------------------------------------------------------------------------

class TestGltfpackDetection:
    def test_has_gltfpack_returns_bool(self):
        from med2glb.glb.compress import _has_gltfpack
        assert isinstance(_has_gltfpack(), bool)

    def test_glb_has_animations_static(self, tmp_path):
        """Static GLB should not be detected as animated."""
        from med2glb.glb.compress import _glb_has_animations
        path = tmp_path / "static.glb"
        _build_textured_glb(path, num_textures=1, size=32)
        assert _glb_has_animations(path) is False


class TestAutoStrategy:
    def test_auto_strategy_picks_downscale_no_tools(self, tmp_path, monkeypatch):
        """Without gltfpack or toktx, auto should fall back to downscale."""
        from med2glb.glb import compress
        monkeypatch.setattr(compress, "_has_gltfpack", lambda: False)
        monkeypatch.setattr(compress, "_has_toktx", lambda: False)
        monkeypatch.setattr(compress, "_has_draco", lambda: False)

        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=5, size=256)
        original_size = path.stat().st_size

        result = constrain_glb_size(path, original_size // 2, strategy="auto")
        assert result is True
        assert path.stat().st_size < original_size

    def test_auto_strategy_valid(self, tmp_path):
        """Auto strategy should not raise for any GLB."""
        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=3, size=128)
        original_size = path.stat().st_size

        # Should not raise
        constrain_glb_size(path, original_size // 2, strategy="auto")

    def test_gltfpack_strategy_fallback(self, tmp_path, monkeypatch):
        """Without gltfpack, strategy should fall back to downscale."""
        from med2glb.glb import compress
        monkeypatch.setattr(compress, "_has_gltfpack", lambda: False)

        path = tmp_path / "test.glb"
        _build_textured_glb(path, num_textures=5, size=256)
        original_size = path.stat().st_size

        result = constrain_glb_size(path, original_size // 2, strategy="gltfpack")
        assert result is True
        assert path.stat().st_size < original_size


class TestSmallImageExclusion:
    def test_skip_small_identifies_large_images_only(self, tmp_path):
        """With skip_small=True, small images (<50KB) should be excluded."""
        from med2glb.glb.compress import _load_and_identify_images, _SMALL_IMAGE_THRESHOLD

        # Build GLB with one large (256x256) and one tiny (8x8) texture
        path = tmp_path / "mixed.glb"
        _build_mixed_size_glb(path)

        # With skip_small=True (default)
        gltf, blob, image_bv_set = _load_and_identify_images(path)
        # Only the large image should be included
        large_count = 0
        for bv_idx in image_bv_set:
            bv = gltf.bufferViews[bv_idx]
            assert bv.byteLength >= _SMALL_IMAGE_THRESHOLD
            large_count += 1
        assert large_count >= 1

    def test_no_skip_includes_all_images(self, tmp_path):
        """With skip_small=False, all images should be included."""
        from med2glb.glb.compress import _load_and_identify_images

        path = tmp_path / "mixed.glb"
        _build_mixed_size_glb(path)

        _, _, all_set = _load_and_identify_images(path, skip_small=False)
        _, _, skip_set = _load_and_identify_images(path, skip_small=True)
        assert len(all_set) >= len(skip_set)


def _build_mixed_size_glb(path: Path) -> None:
    """Build a GLB with one large texture (256x256) and one tiny (4x4)."""
    from med2glb.glb.builder import _pad_to_4
    from PIL import Image as PILImage
    import io as _io

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[pygltflib.Sampler()],
    )
    binary_data = bytearray()

    for size in [256, 4]:
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr)
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        offset = len(binary_data)
        binary_data.extend(png_bytes)
        _pad_to_4(binary_data)

        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=offset, byteLength=len(png_bytes),
        ))
        idx = len(gltf.images)
        gltf.images.append(pygltflib.Image(bufferView=bv_idx, mimeType="image/png"))
        gltf.textures.append(pygltflib.Texture(sampler=0, source=idx))

    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))
    gltf.save(str(path))
