"""GLB size constraint: compress textures and meshes to fit a maximum file size."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pygltflib

logger = logging.getLogger(__name__)


def constrain_glb_size(
    path: Path,
    max_bytes: int,
    strategy: str = "draco",
) -> bool:
    """Compress a GLB file to fit under max_bytes.

    Strategies (tried in order within each):
      - draco: Draco mesh compression, then texture downscaling if needed.
      - downscale: Progressively reduce texture resolution (lossless PNG).
      - jpeg: Re-encode textures as JPEG with decreasing quality.

    Returns True if any compression was applied.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size <= max_bytes:
        return False

    if strategy == "draco":
        return _strategy_draco(path, max_bytes)
    elif strategy == "downscale":
        return _strategy_downscale(path, max_bytes)
    elif strategy == "jpeg":
        return _strategy_jpeg(path, max_bytes)
    else:
        raise ValueError(f"Unknown compression strategy: {strategy}")


# ---------------------------------------------------------------------------
# Strategy: Draco (mesh compression + downscale fallback)
# ---------------------------------------------------------------------------

def _strategy_draco(path: Path, max_bytes: int) -> bool:
    """Draco mesh compression, falling back to texture downscale."""
    applied = False

    # Step 1: Try Draco mesh compression
    if _has_draco():
        if _apply_draco(path):
            applied = True
            if path.stat().st_size <= max_bytes:
                return True

    # Step 2: Fall back to texture downscale for remaining savings
    if _strategy_downscale(path, max_bytes):
        applied = True

    return applied


def _has_draco() -> bool:
    """Check if Draco encoding is available via trimesh."""
    try:
        from trimesh.exchange import gltf as _  # noqa: F401
        import trimesh
        # trimesh bundles draco via google-draco or draco packages
        resolver = trimesh.resolvers.FilePathResolver("")
        return True
    except Exception:
        return False


def _apply_draco(path: Path) -> bool:
    """Compress mesh data in a GLB using trimesh's Draco support."""
    try:
        # Trimesh does not preserve glTF animations — skip if present
        gltf_check = pygltflib.GLTF2.load(str(path))
        if gltf_check.animations:
            logger.debug("Skipping Draco compression — GLB contains animations")
            return False

        import trimesh

        scene = trimesh.load(str(path), force="scene")

        # Check if the scene actually contains meshes (not just textures)
        has_meshes = False
        for geom in scene.geometry.values():
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                has_meshes = True
                break

        if not has_meshes:
            logger.debug("No mesh geometry found, skipping Draco compression")
            return False

        original_size = path.stat().st_size
        data = scene.export(file_type="glb")
        new_size = len(data)

        if new_size < original_size:
            with open(str(path), "wb") as f:
                f.write(data)
            logger.info(
                f"Draco compression: {_fmt(original_size)} → {_fmt(new_size)} "
                f"({100 - new_size * 100 // original_size}% reduction)"
            )
            return True

    except Exception as e:
        logger.warning(f"Draco compression failed: {e}. Falling back to downscale.")

    return False


# ---------------------------------------------------------------------------
# Strategy: Texture downscale (lossless PNG, reduced resolution)
# ---------------------------------------------------------------------------

# (scale_factor,) — tried in order until file fits
_DOWNSCALE_STEPS = [0.75, 0.5, 0.375, 0.25]


def _strategy_downscale(path: Path, max_bytes: int) -> bool:
    """Progressively downscale textures while keeping PNG format."""
    gltf, blob, image_bv_set = _load_and_identify_images(path)
    if not image_bv_set:
        return False

    original_size = path.stat().st_size

    for scale in _DOWNSCALE_STEPS:
        new_blob = _rebuild_blob(gltf, blob, image_bv_set, "PNG", 0, scale)
        if _estimate_file_size(gltf, new_blob) <= max_bytes:
            _save_rebuilt(gltf, new_blob, image_bv_set, "image/png", path)
            logger.info(
                f"Texture downscale ({scale:.0%}): "
                f"{_fmt(original_size)} → {_fmt(path.stat().st_size)}"
            )
            return True

    # Apply maximum downscale even if still over limit
    new_blob = _rebuild_blob(gltf, blob, image_bv_set, "PNG", 0, _DOWNSCALE_STEPS[-1])
    _save_rebuilt(gltf, new_blob, image_bv_set, "image/png", path)
    logger.warning(
        f"Texture downscale applied ({_DOWNSCALE_STEPS[-1]:.0%}) but still "
        f"exceeds {_fmt(max_bytes)} ({_fmt(path.stat().st_size)})"
    )
    return True


# ---------------------------------------------------------------------------
# Strategy: JPEG re-encoding
# ---------------------------------------------------------------------------

# (quality, scale) — tried in order
_JPEG_STEPS = [
    (90, 1.0),
    (80, 1.0),
    (70, 1.0),
    (60, 0.75),
    (50, 0.5),
    (30, 0.25),
]


def _strategy_jpeg(path: Path, max_bytes: int) -> bool:
    """Re-encode textures as JPEG with progressive quality reduction."""
    gltf, blob, image_bv_set = _load_and_identify_images(path)
    if not image_bv_set:
        return False

    original_size = path.stat().st_size

    for quality, scale in _JPEG_STEPS:
        new_blob = _rebuild_blob(gltf, blob, image_bv_set, "JPEG", quality, scale)
        if _estimate_file_size(gltf, new_blob) <= max_bytes:
            _save_rebuilt(gltf, new_blob, image_bv_set, "image/jpeg", path)
            logger.info(
                f"JPEG compression (q={quality}, scale={scale}): "
                f"{_fmt(original_size)} → {_fmt(path.stat().st_size)}"
            )
            return True

    # Apply maximum compression even if still over limit
    q, s = _JPEG_STEPS[-1]
    new_blob = _rebuild_blob(gltf, blob, image_bv_set, "JPEG", q, s)
    _save_rebuilt(gltf, new_blob, image_bv_set, "image/jpeg", path)
    logger.warning(
        f"JPEG compression applied (q={q}, scale={s}) but still "
        f"exceeds {_fmt(max_bytes)} ({_fmt(path.stat().st_size)})"
    )
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_and_identify_images(
    path: Path,
) -> tuple[pygltflib.GLTF2, bytes, set[int]]:
    """Load a GLB and identify which bufferViews hold image data."""
    gltf = pygltflib.GLTF2.load(str(path))
    blob = gltf.binary_blob() or b""
    image_bv_set = {img.bufferView for img in gltf.images if img.bufferView is not None}
    return gltf, blob, image_bv_set


def _rebuild_blob(
    gltf: pygltflib.GLTF2,
    old_blob: bytes,
    image_bv_set: set[int],
    fmt: str,
    quality: int,
    scale: float,
) -> bytearray:
    """Rebuild the binary blob, re-encoding image bufferViews.

    Non-image bufferViews are copied as-is. Image bufferViews are
    re-encoded with the specified format, quality, and scale.
    """
    new_blob = bytearray()
    new_offsets: dict[int, tuple[int, int]] = {}  # bv_idx → (offset, length)

    # Process in offset order so padding stays consistent
    bv_order = sorted(
        range(len(gltf.bufferViews)),
        key=lambda i: gltf.bufferViews[i].byteOffset,
    )

    for bv_idx in bv_order:
        bv = gltf.bufferViews[bv_idx]
        old_data = old_blob[bv.byteOffset : bv.byteOffset + bv.byteLength]

        new_offset = len(new_blob)

        if bv_idx in image_bv_set:
            new_data = _reencode_image(old_data, fmt, quality, scale)
        else:
            new_data = bytes(old_data)

        new_offsets[bv_idx] = (new_offset, len(new_data))
        new_blob.extend(new_data)

        # 4-byte alignment
        remainder = len(new_blob) % 4
        if remainder:
            new_blob.extend(b"\x00" * (4 - remainder))

    # Store offset map for _save_rebuilt
    gltf._new_offsets = new_offsets  # type: ignore[attr-defined]
    return new_blob


def _reencode_image(data: bytes, fmt: str, quality: int, scale: float) -> bytes:
    """Re-encode image bytes with the given format, quality, and scale."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        return data  # Not decodable — keep as-is

    if scale < 1.0:
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _estimate_file_size(gltf: pygltflib.GLTF2, blob: bytearray) -> int:
    """Estimate total GLB file size (header + JSON chunk + binary chunk)."""
    # GLB: 12-byte header + 8-byte JSON chunk header + JSON + 8-byte bin header + bin
    # JSON size varies but is typically small; estimate 4KB overhead
    return len(blob) + 4096


def _save_rebuilt(
    gltf: pygltflib.GLTF2,
    new_blob: bytearray,
    image_bv_set: set[int],
    mime_type: str,
    path: Path,
) -> None:
    """Apply new offsets/lengths and save the GLB."""
    offsets = getattr(gltf, "_new_offsets", {})
    for bv_idx, (offset, length) in offsets.items():
        gltf.bufferViews[bv_idx].byteOffset = offset
        gltf.bufferViews[bv_idx].byteLength = length

    for img in gltf.images:
        if img.bufferView in image_bv_set:
            img.mimeType = mime_type

    gltf.buffers[0].byteLength = len(new_blob)
    gltf.set_binary_blob(bytes(new_blob))

    if hasattr(gltf, "_new_offsets"):
        del gltf._new_offsets

    gltf.save(str(path))


def _fmt(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"
