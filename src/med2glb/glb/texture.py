"""Textured plane GLB: embed a DICOM image as a textured quad via pygltflib."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.volume import DicomVolume, TemporalSequence
from med2glb.glb.builder import _pad_to_4


def build_textured_plane_glb(volume: DicomVolume, output_path: Path) -> None:
    """Build a GLB containing a textured quad from a single DICOM slice."""
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(name="echo_frame", mesh=0)],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[],
    )

    binary_data = bytearray()

    # Get 2D pixel data (first slice of the volume)
    # Prefer original RGB data for texture if available
    if volume.rgb_data is not None:
        pixel_data = volume.rgb_data[0]  # [Y, X, 3]
        rows, cols = pixel_data.shape[:2]
    else:
        pixel_data = volume.voxels[0]  # [Y, X]
        rows, cols = pixel_data.shape

    # Compute physical dimensions from pixel spacing
    row_spacing, col_spacing = volume.pixel_spacing
    width = cols * col_spacing / 1000.0  # Convert mm to meters for glTF
    height = rows * row_spacing / 1000.0

    # Create quad vertices (two triangles forming a rectangle)
    half_w = width / 2
    half_h = height / 2
    vertices = np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float32,
    )

    # UV coordinates
    texcoords = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Two triangles
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    # Normals (all pointing +Z)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # --- Encode image as PNG ---
    png_bytes = _pixel_data_to_png(pixel_data)

    # Write image data first
    img_offset = len(binary_data)
    binary_data.extend(png_bytes)
    _pad_to_4(binary_data)

    img_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=img_offset,
            byteLength=len(png_bytes),
        )
    )

    gltf.images.append(
        pygltflib.Image(bufferView=img_bv_idx, mimeType="image/png")
    )

    gltf.samplers.append(
        pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        )
    )

    gltf.textures.append(pygltflib.Texture(sampler=0, source=0))

    # Material with base color texture
    gltf.materials.append(
        pygltflib.Material(
            name="dicom_image",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        )
    )

    # --- Write vertex data ---
    # Positions
    pos_data = vertices.tobytes()
    pos_offset = len(binary_data)
    binary_data.extend(pos_data)
    _pad_to_4(binary_data)

    pos_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=pos_offset,
            byteLength=len(pos_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=pos_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        )
    )

    # Normals
    norm_data = normals.tobytes()
    norm_offset = len(binary_data)
    binary_data.extend(norm_data)
    _pad_to_4(binary_data)

    norm_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=norm_offset,
            byteLength=len(norm_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    norm_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=norm_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC3,
        )
    )

    # Texcoords
    tc_data = texcoords.tobytes()
    tc_offset = len(binary_data)
    binary_data.extend(tc_data)
    _pad_to_4(binary_data)

    tc_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=tc_offset,
            byteLength=len(tc_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    tc_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=tc_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC2,
        )
    )

    # Indices
    idx_data = indices.tobytes()
    idx_offset = len(binary_data)
    binary_data.extend(idx_data)
    _pad_to_4(binary_data)

    idx_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=idx_offset,
            byteLength=len(idx_data),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    idx_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=idx_bv_idx,
            componentType=pygltflib.UNSIGNED_SHORT,
            count=6,
            type=pygltflib.SCALAR,
            max=[3],
            min=[0],
        )
    )

    # Mesh primitive
    gltf.meshes.append(
        pygltflib.Mesh(
            name="echo_frame",
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=pos_acc_idx,
                        NORMAL=norm_acc_idx,
                        TEXCOORD_0=tc_acc_idx,
                    ),
                    indices=idx_acc_idx,
                    material=0,
                )
            ],
        )
    )

    # Buffer
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def build_animated_textured_plane_glb(
    sequence: TemporalSequence,
    output_path: Path,
) -> None:
    """Build an animated GLB from 2D cine using per-frame texture planes.

    Each frame is a separate textured quad with its own full-resolution RGB
    texture.  Animation switches visible frame via scale: the active frame
    has scale [1,1,1], all others [0,0,0].  This produces output that looks
    identical to the original DICOM viewer.
    """
    n_frames = sequence.frame_count

    # Get dimensions from first frame
    vol = sequence.frames[0]
    if vol.rgb_data is not None:
        rows, cols = vol.rgb_data.shape[1], vol.rgb_data.shape[2]
    else:
        rows, cols = vol.voxels.shape[1], vol.voxels.shape[2]

    row_spacing, col_spacing = vol.pixel_spacing
    width = cols * col_spacing / 1000.0   # mm -> metres
    height = rows * row_spacing / 1000.0

    # Shared quad geometry
    half_w, half_h = width / 2, height / 2
    vertices = np.array([
        [-half_w, -half_h, 0.0],
        [ half_w, -half_h, 0.0],
        [ half_w,  half_h, 0.0],
        [-half_w,  half_h, 0.0],
    ], dtype=np.float32)
    texcoords = np.array([
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
    ], dtype=np.float32)
    normals = np.array([
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    # --- Build glTF ---
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=list(range(n_frames)))],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[],
        animations=[],
    )
    binary_data = bytearray()

    # One shared sampler
    gltf.samplers.append(pygltflib.Sampler(
        magFilter=pygltflib.LINEAR,
        minFilter=pygltflib.LINEAR,
        wrapS=pygltflib.CLAMP_TO_EDGE,
        wrapT=pygltflib.CLAMP_TO_EDGE,
    ))

    # --- Encode each frame as a PNG image -> texture -> material ---
    for i in range(n_frames):
        fv = sequence.frames[i]
        if fv.rgb_data is not None:
            png_bytes = _pixel_data_to_png(fv.rgb_data[0])
        else:
            png_bytes = _pixel_data_to_png(fv.voxels[0])

        img_offset = len(binary_data)
        binary_data.extend(png_bytes)
        _pad_to_4(binary_data)

        img_bv = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=img_offset, byteLength=len(png_bytes),
        ))
        gltf.images.append(pygltflib.Image(bufferView=img_bv, mimeType="image/png"))
        gltf.textures.append(pygltflib.Texture(sampler=0, source=i))
        gltf.materials.append(pygltflib.Material(
            name=f"frame_{i}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=i),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        ))

    # --- Shared geometry (written once, referenced by all meshes) ---
    def _write_accessor(data_array, target, comp_type, acc_type, with_minmax=False):
        raw = data_array.tobytes()
        off = len(binary_data)
        binary_data.extend(raw)
        _pad_to_4(binary_data)
        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=off, byteLength=len(raw), target=target,
        ))
        acc_idx = len(gltf.accessors)
        kwargs = dict(bufferView=bv_idx, componentType=comp_type,
                      count=len(data_array), type=acc_type)
        if with_minmax:
            mx = data_array.max(axis=0).tolist()
            mn = data_array.min(axis=0).tolist()
            # glTF requires min/max to be arrays, not scalars
            if not isinstance(mx, list):
                mx = [mx]
                mn = [mn]
            kwargs["max"] = mx
            kwargs["min"] = mn
        gltf.accessors.append(pygltflib.Accessor(**kwargs))
        return acc_idx

    pos_acc = _write_accessor(vertices, pygltflib.ARRAY_BUFFER,
                              pygltflib.FLOAT, pygltflib.VEC3, True)
    norm_acc = _write_accessor(normals, pygltflib.ARRAY_BUFFER,
                               pygltflib.FLOAT, pygltflib.VEC3)
    tc_acc = _write_accessor(texcoords, pygltflib.ARRAY_BUFFER,
                             pygltflib.FLOAT, pygltflib.VEC2)
    idx_acc = _write_accessor(indices, pygltflib.ELEMENT_ARRAY_BUFFER,
                              pygltflib.UNSIGNED_SHORT, pygltflib.SCALAR, True)

    # --- One mesh + node per frame ---
    for i in range(n_frames):
        gltf.meshes.append(pygltflib.Mesh(
            name=f"echo_frame_{i}",
            primitives=[pygltflib.Primitive(
                attributes=pygltflib.Attributes(
                    POSITION=pos_acc, NORMAL=norm_acc, TEXCOORD_0=tc_acc,
                ),
                indices=idx_acc,
                material=i,
            )],
        ))
        # First frame visible (scale 1), rest hidden (scale 0)
        s = [1.0, 1.0, 1.0] if i == 0 else [0.0, 0.0, 0.0]
        gltf.nodes.append(pygltflib.Node(
            name=f"echo_frame_{i}", mesh=i, scale=s,
        ))

    # --- Animation: switch visible frame via scale ---
    frame_time_ms = sequence.temporal_resolution or 33.3
    dt = frame_time_ms / 1000.0

    # Shared keyframe times (one per frame)
    keyframe_times = np.array([i * dt for i in range(n_frames)], dtype=np.float32)
    time_raw = keyframe_times.tobytes()
    time_off = len(binary_data)
    binary_data.extend(time_raw)
    _pad_to_4(binary_data)

    time_bv = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=time_off, byteLength=len(time_raw),
    ))
    time_acc = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=time_bv, componentType=pygltflib.FLOAT,
        count=n_frames, type=pygltflib.SCALAR,
        max=[float(keyframe_times[-1])],
        min=[float(keyframe_times[0])],
    ))

    # Per-node scale output: visible=[1,1,1] at its own keyframe, [0,0,0] otherwise
    channels = []
    samplers = []
    for i in range(n_frames):
        scales = np.zeros((n_frames, 3), dtype=np.float32)
        scales[i] = [1.0, 1.0, 1.0]

        s_raw = scales.tobytes()
        s_off = len(binary_data)
        binary_data.extend(s_raw)
        _pad_to_4(binary_data)

        s_bv = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=s_off, byteLength=len(s_raw),
        ))
        s_acc = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=s_bv, componentType=pygltflib.FLOAT,
            count=n_frames, type=pygltflib.VEC3,
        ))

        sampler_idx = len(samplers)
        samplers.append(pygltflib.AnimationSampler(
            input=time_acc,
            output=s_acc,
            interpolation=pygltflib.ANIM_STEP,
        ))
        channels.append(pygltflib.AnimationChannel(
            sampler=sampler_idx,
            target=pygltflib.AnimationChannelTarget(node=i, path="scale"),
        ))

    gltf.animations.append(pygltflib.Animation(
        name="cardiac_cycle",
        channels=channels,
        samplers=samplers,
    ))

    # --- Finalize ---
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def _pixel_data_to_png(pixel_data: np.ndarray) -> bytes:
    """Convert pixel array to PNG bytes.

    Accepts:
      - (Y, X) grayscale: normalizes to 0-255 and encodes as L
      - (Y, X, 3) RGB uint8: encodes directly as RGB
    """
    from PIL import Image

    if pixel_data.ndim == 3 and pixel_data.shape[-1] == 3:
        # RGB data — use as-is
        data = np.clip(pixel_data, 0, 255).astype(np.uint8)
        img = Image.fromarray(data, mode="RGB")
    else:
        # Grayscale — normalize to 0-255
        data = pixel_data.astype(np.float64)
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
        img = Image.fromarray(data, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Public alias for use by gallery modules
pixel_data_to_png = _pixel_data_to_png
