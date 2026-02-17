"""Tests for GLB file size constraint and compression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from dicom2glb.glb.compress import constrain_glb_size, _reencode_image


def _build_textured_glb(path: Path, num_textures: int = 1, size: int = 128) -> None:
    """Build a simple textured GLB for testing compression."""
    from dicom2glb.glb.builder import _pad_to_4
    from dicom2glb.glb.texture import pixel_data_to_png

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


class TestReencodeImage:
    """Tests for image re-encoding."""

    def test_jpeg_reencode(self):
        """PNG data should be re-encoded as JPEG."""
        from dicom2glb.glb.texture import pixel_data_to_png

        pixel_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        png_bytes = pixel_data_to_png(pixel_data.astype(np.float32))

        jpeg_bytes = _reencode_image(png_bytes, "JPEG", 80, 1.0)
        assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic number
        assert len(jpeg_bytes) < len(png_bytes)

    def test_downscale_reduces_resolution(self):
        """Downscaling should produce smaller PNG."""
        from dicom2glb.glb.texture import pixel_data_to_png

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
