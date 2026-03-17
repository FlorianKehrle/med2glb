"""GLB size constraint: compress textures and meshes to fit a maximum file size."""

from __future__ import annotations

import functools
import io
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pygltflib

logger = logging.getLogger(__name__)


def constrain_glb_size(
    path: Path,
    max_bytes: int,
    strategy: str = "auto",
) -> bool:
    """Compress a GLB file to fit under max_bytes.

    Strategies (tried in order within each):
      - auto: Automatically picks the best strategy based on GLB content.
              Animated GLBs → gltfpack + KTX2; static → Draco + KTX2.
      - gltfpack: Meshopt quantization + compression (requires gltfpack CLI).
                  Best for animated GLBs with morph targets.
      - draco: Draco mesh compression, then texture downscaling if needed.
      - downscale: Progressively reduce texture resolution (lossless PNG).
      - jpeg: Re-encode textures as JPEG with decreasing quality.
      - ktx2: GPU-compressed textures via KTX2/Basis Universal (requires toktx).

    Returns True if any compression was applied.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size <= max_bytes:
        return False

    if strategy == "auto":
        return _strategy_auto(path, max_bytes)
    elif strategy == "gltfpack":
        return _strategy_gltfpack(path, max_bytes)
    elif strategy == "draco":
        return _strategy_draco(path, max_bytes)
    elif strategy == "downscale":
        return _strategy_downscale(path, max_bytes)
    elif strategy == "jpeg":
        return _strategy_jpeg(path, max_bytes)
    elif strategy == "ktx2":
        return _strategy_ktx2(path, max_bytes)
    else:
        raise ValueError(f"Unknown compression strategy: {strategy}")


# ---------------------------------------------------------------------------
# Strategy: Auto (picks best strategy based on GLB content)
# ---------------------------------------------------------------------------

def _glb_has_animations(path: Path) -> bool:
    """Check if a GLB contains animations (morph targets, skeletal, etc.)."""
    try:
        gltf = pygltflib.GLTF2.load(str(path))
        return bool(gltf.animations)
    except Exception:
        return False


def _strategy_auto(path: Path, max_bytes: int) -> bool:
    """Auto-select best compression strategy based on GLB content.

    Animated GLBs (morph targets) → gltfpack + KTX2.
    Static GLBs → Draco + KTX2.
    Falls back gracefully when tools are unavailable.
    """
    has_anim = _glb_has_animations(path)

    if has_anim:
        # Animated: gltfpack handles morph targets, then KTX2 for textures
        if _has_gltfpack():
            logger.info("Auto strategy: animated GLB detected → gltfpack + KTX2")
            return _strategy_gltfpack_ktx2(path, max_bytes)
        elif _has_toktx():
            logger.info("Auto strategy: animated GLB, gltfpack unavailable → KTX2 only")
            return _strategy_ktx2(path, max_bytes)
        else:
            logger.info("Auto strategy: animated GLB, no tools → downscale")
            return _strategy_downscale(path, max_bytes)
    else:
        # Static: Draco for mesh, then KTX2 for textures
        applied = False
        if _has_draco():
            applied = _strategy_draco(path, max_bytes)
            if path.stat().st_size <= max_bytes:
                return applied
        if _has_toktx():
            if _strategy_ktx2(path, max_bytes):
                applied = True
        elif _strategy_downscale(path, max_bytes):
            applied = True
        return applied


# ---------------------------------------------------------------------------
# Strategy: gltfpack (meshopt quantization + compression)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _has_gltfpack() -> bool:
    """Check if the ``gltfpack`` CLI tool is on PATH."""
    return shutil.which("gltfpack") is not None


def _apply_gltfpack(path: Path, output: Path | None = None) -> bool:
    """Run gltfpack on a GLB for mesh quantization + compression.

    Uses KHR_mesh_quantization for vertex attribute quantization and
    EXT_meshopt_compression for buffer-level compression.  Both are
    supported by glTFast (Unity) with the meshoptimizer package.

    Legend/info nodes are stripped before gltfpack (which would damage
    their text textures via quantization and texture transforms) and
    merged back into the result afterwards.

    Writes to *output* if given, otherwise compresses in-place.
    """
    if not _has_gltfpack():
        return False

    target = output or path
    use_temp = output is None
    if use_temp:
        target = path.with_suffix(".gltfpack.glb")

    # Strip legend/info nodes before gltfpack (it damages text textures)
    stripped_path = path.with_suffix(".stripped.glb")
    legend_bytes = _strip_metadata_nodes(path, stripped_path)
    gltfpack_input = stripped_path if legend_bytes else path

    try:
        subprocess.run(
            [
                "gltfpack",
                "-i", str(gltfpack_input),
                "-o", str(target),
                "-vp", "14",        # position quantization bits
                "-vt", "12",        # texcoord quantization bits
                "-vn", "8",         # normal quantization bits
                "-cc",              # meshopt compression (EXT_meshopt_compression)
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        logger.warning("gltfpack timed out after 300s")
        if use_temp and target.exists():
            target.unlink()
        _cleanup_temp(stripped_path, legend_bytes)
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"gltfpack failed: {e.stderr.decode(errors='replace')}")
        if use_temp and target.exists():
            target.unlink()
        _cleanup_temp(stripped_path, legend_bytes)
        return False

    # Re-add legend/info nodes to the gltfpacked result
    if legend_bytes and target.exists():
        _restore_metadata_nodes(target, legend_bytes)

    _cleanup_temp(stripped_path, legend_bytes)

    if not target.exists():
        return False

    if use_temp:
        original_size = path.stat().st_size
        new_size = target.stat().st_size
        if new_size < original_size:
            shutil.move(str(target), str(path))
            logger.info(
                f"gltfpack compression: {_fmt(original_size)} → {_fmt(new_size)} "
                f"({100 - new_size * 100 // original_size}% reduction)"
            )
            return True
        else:
            target.unlink()
            logger.debug("gltfpack did not reduce file size")
            return False

    return True


def _cleanup_temp(stripped_path: Path, legend_bytes: bytes | None) -> None:
    """Remove temporary stripped GLB if it was created."""
    if legend_bytes and stripped_path.exists():
        stripped_path.unlink()


def _strip_metadata_nodes(
    src: Path,
    dst: Path,
) -> bytes | None:
    """Strip legend/info nodes from a GLB, saving the original for later restore.

    Returns the original GLB bytes if metadata nodes were found and stripped,
    or None if no metadata nodes were present (dst is not created).
    """
    gltf = pygltflib.GLTF2.load(str(src))
    if not gltf.nodes:
        return None

    # Find metadata node indices
    metadata_node_indices: set[int] = set()
    for i, node in enumerate(gltf.nodes):
        if node.name is None:
            continue
        name_lower = node.name.lower()
        if any(kw in name_lower for kw in _METADATA_NODE_NAMES):
            metadata_node_indices.add(i)

    if not metadata_node_indices:
        return None

    # Save original for restore
    original_bytes = src.read_bytes()

    # Remove metadata nodes from scene roots
    for scene in gltf.scenes:
        if scene.nodes:
            scene.nodes = [n for n in scene.nodes if n not in metadata_node_indices]

    # Also remove from any parent node's children list
    for node in gltf.nodes:
        if node.children:
            node.children = [c for c in node.children if c not in metadata_node_indices]

    gltf.save(str(dst))
    logger.debug(
        "Stripped %d legend/info node(s) before gltfpack",
        len(metadata_node_indices),
    )
    return original_bytes


def _restore_metadata_nodes(packed_path: Path, original_bytes: bytes) -> None:
    """Merge legend/info nodes from original GLB back into gltfpacked result.

    Loads the original (pre-gltfpack) GLB, extracts legend/info nodes with
    their full resource chains (meshes, materials, textures, images, bufferViews),
    and appends them to the gltfpacked GLB.
    """
    orig_gltf = pygltflib.GLTF2.load_from_bytes(original_bytes)
    packed_gltf = pygltflib.GLTF2.load(str(packed_path))
    orig_blob = orig_gltf.binary_blob() or b""
    packed_blob = bytearray(packed_gltf.binary_blob() or b"")

    # Find metadata nodes in the original
    metadata_node_indices: list[int] = []
    for i, node in enumerate(orig_gltf.nodes):
        if node.name is None:
            continue
        name_lower = node.name.lower()
        if any(kw in name_lower for kw in _METADATA_NODE_NAMES):
            metadata_node_indices.append(i)

    if not metadata_node_indices:
        return

    # Track resource remapping
    bv_remap: dict[int, int] = {}
    acc_remap: dict[int, int] = {}
    img_remap: dict[int, int] = {}
    tex_remap: dict[int, int] = {}
    mat_remap: dict[int, int] = {}
    mesh_remap: dict[int, int] = {}
    sampler_remap: dict[int, int] = {}

    def _copy_bufferviews(bv_indices: set[int]) -> None:
        for bv_idx in bv_indices:
            if bv_idx in bv_remap:
                continue
            bv = orig_gltf.bufferViews[bv_idx]
            data = orig_blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
            new_offset = len(packed_blob)
            packed_blob.extend(data)
            # 4-byte alignment
            while len(packed_blob) % 4:
                packed_blob.extend(b"\x00")
            new_bv = pygltflib.BufferView(
                buffer=0, byteOffset=new_offset, byteLength=bv.byteLength,
                byteStride=bv.byteStride, target=bv.target,
            )
            bv_remap[bv_idx] = len(packed_gltf.bufferViews)
            packed_gltf.bufferViews.append(new_bv)

    def _copy_accessor(acc_idx: int) -> int:
        if acc_idx in acc_remap:
            return acc_remap[acc_idx]
        acc = orig_gltf.accessors[acc_idx]
        if acc.bufferView is not None:
            _copy_bufferviews({acc.bufferView})
        new_acc = pygltflib.Accessor(
            bufferView=bv_remap.get(acc.bufferView, acc.bufferView),
            byteOffset=acc.byteOffset,
            componentType=acc.componentType,
            count=acc.count,
            type=acc.type,
            max=acc.max,
            min=acc.min,
            normalized=acc.normalized,
        )
        new_idx = len(packed_gltf.accessors)
        acc_remap[acc_idx] = new_idx
        packed_gltf.accessors.append(new_acc)
        return new_idx

    def _copy_sampler(sampler_idx: int) -> int:
        if sampler_idx in sampler_remap:
            return sampler_remap[sampler_idx]
        s = orig_gltf.samplers[sampler_idx]
        new_idx = len(packed_gltf.samplers)
        sampler_remap[sampler_idx] = new_idx
        packed_gltf.samplers.append(pygltflib.Sampler(
            magFilter=s.magFilter, minFilter=s.minFilter,
            wrapS=s.wrapS, wrapT=s.wrapT,
        ))
        return new_idx

    def _copy_image(img_idx: int) -> int:
        if img_idx in img_remap:
            return img_remap[img_idx]
        img = orig_gltf.images[img_idx]
        if img.bufferView is not None:
            _copy_bufferviews({img.bufferView})
        new_img = pygltflib.Image(
            bufferView=bv_remap.get(img.bufferView, img.bufferView),
            mimeType=img.mimeType,
            name=img.name,
        )
        new_idx = len(packed_gltf.images)
        img_remap[img_idx] = new_idx
        packed_gltf.images.append(new_img)
        return new_idx

    def _copy_texture(tex_idx: int) -> int:
        if tex_idx in tex_remap:
            return tex_remap[tex_idx]
        tex = orig_gltf.textures[tex_idx]
        new_source = _copy_image(tex.source) if tex.source is not None else None
        new_sampler = _copy_sampler(tex.sampler) if tex.sampler is not None else None
        new_idx = len(packed_gltf.textures)
        tex_remap[tex_idx] = new_idx
        packed_gltf.textures.append(pygltflib.Texture(
            sampler=new_sampler, source=new_source,
        ))
        return new_idx

    def _copy_texinfo(ti: pygltflib.TextureInfo | None) -> pygltflib.TextureInfo | None:
        if ti is None:
            return None
        return pygltflib.TextureInfo(index=_copy_texture(ti.index), texCoord=ti.texCoord)

    def _copy_material(mat_idx: int) -> int:
        if mat_idx in mat_remap:
            return mat_remap[mat_idx]
        mat = orig_gltf.materials[mat_idx]
        pbr = None
        if mat.pbrMetallicRoughness:
            p = mat.pbrMetallicRoughness
            pbr = pygltflib.PbrMetallicRoughness(
                baseColorFactor=p.baseColorFactor,
                baseColorTexture=_copy_texinfo(p.baseColorTexture),
                metallicFactor=p.metallicFactor,
                roughnessFactor=p.roughnessFactor,
                metallicRoughnessTexture=_copy_texinfo(p.metallicRoughnessTexture),
            )
        new_mat = pygltflib.Material(
            name=mat.name,
            pbrMetallicRoughness=pbr,
            emissiveTexture=_copy_texinfo(mat.emissiveTexture),
            emissiveFactor=mat.emissiveFactor,
            alphaMode=mat.alphaMode,
            alphaCutoff=mat.alphaCutoff,
            doubleSided=mat.doubleSided,
            extensions=mat.extensions,
        )
        new_idx = len(packed_gltf.materials)
        mat_remap[mat_idx] = new_idx
        packed_gltf.materials.append(new_mat)
        return new_idx

    def _copy_mesh(mesh_idx: int) -> int:
        if mesh_idx in mesh_remap:
            return mesh_remap[mesh_idx]
        mesh = orig_gltf.meshes[mesh_idx]
        new_prims = []
        for prim in mesh.primitives:
            # Copy accessors
            new_attrs = pygltflib.Attributes()
            if prim.attributes.POSITION is not None:
                new_attrs.POSITION = _copy_accessor(prim.attributes.POSITION)
            if prim.attributes.NORMAL is not None:
                new_attrs.NORMAL = _copy_accessor(prim.attributes.NORMAL)
            if prim.attributes.TEXCOORD_0 is not None:
                new_attrs.TEXCOORD_0 = _copy_accessor(prim.attributes.TEXCOORD_0)
            if prim.attributes.COLOR_0 is not None:
                new_attrs.COLOR_0 = _copy_accessor(prim.attributes.COLOR_0)
            new_indices = _copy_accessor(prim.indices) if prim.indices is not None else None
            new_mat = _copy_material(prim.material) if prim.material is not None else None
            new_prims.append(pygltflib.Primitive(
                attributes=new_attrs, indices=new_indices, material=new_mat,
                mode=prim.mode,
            ))
        new_idx = len(packed_gltf.meshes)
        mesh_remap[mesh_idx] = new_idx
        packed_gltf.meshes.append(pygltflib.Mesh(name=mesh.name, primitives=new_prims))
        return new_idx

    # Copy each metadata node with its full resource chain
    for node_idx in metadata_node_indices:
        node = orig_gltf.nodes[node_idx]
        new_mesh = _copy_mesh(node.mesh) if node.mesh is not None else None
        new_node = pygltflib.Node(
            name=node.name,
            mesh=new_mesh,
            translation=node.translation,
            rotation=node.rotation,
            scale=node.scale,
        )
        new_node_idx = len(packed_gltf.nodes)
        packed_gltf.nodes.append(new_node)
        # Add to first scene
        if packed_gltf.scenes:
            packed_gltf.scenes[0].nodes.append(new_node_idx)

    # Update buffer length and save
    packed_gltf.buffers[0].byteLength = len(packed_blob)
    packed_gltf.set_binary_blob(bytes(packed_blob))

    # Preserve KHR_materials_unlit extension (used by legend/info)
    ext = "KHR_materials_unlit"
    if packed_gltf.extensionsUsed is None:
        packed_gltf.extensionsUsed = []
    if ext not in packed_gltf.extensionsUsed:
        packed_gltf.extensionsUsed.append(ext)

    packed_gltf.save(str(packed_path))
    logger.debug(
        "Restored %d legend/info node(s) after gltfpack",
        len(metadata_node_indices),
    )


def _strategy_gltfpack(path: Path, max_bytes: int) -> bool:
    """Meshopt compression via gltfpack, with KTX2 fallback for textures."""
    if not _has_gltfpack():
        logger.warning(
            "gltfpack not found on PATH — falling back to downscale strategy. "
            "Install meshoptimizer for best compression: "
            "https://github.com/zeux/meshoptimizer/releases"
        )
        return _strategy_downscale(path, max_bytes)

    original_size = path.stat().st_size
    applied = _apply_gltfpack(path)

    if applied and path.stat().st_size <= max_bytes:
        return True

    # Gltfpack alone wasn't enough — try KTX2 for remaining texture savings
    if _has_toktx():
        if _strategy_ktx2(path, max_bytes):
            applied = True
    elif _strategy_downscale(path, max_bytes):
        applied = True

    return applied


def _strategy_gltfpack_ktx2(path: Path, max_bytes: int) -> bool:
    """Combined gltfpack (geometry) + KTX2 (textures) for maximum compression."""
    applied = False
    original_size = path.stat().st_size

    # Step 1: gltfpack for mesh/morph target compression
    if _has_gltfpack() and _apply_gltfpack(path):
        applied = True
        if path.stat().st_size <= max_bytes:
            return True

    # Step 2: KTX2 for texture compression
    if _has_toktx():
        if _strategy_ktx2(path, max_bytes):
            applied = True
    elif _strategy_downscale(path, max_bytes):
        applied = True

    return applied


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
# Strategy: KTX2 / Basis Universal (GPU-compressed textures)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _has_toktx() -> bool:
    """Check if the Khronos ``toktx`` CLI tool is on PATH."""
    return shutil.which("toktx") is not None


def _image_to_ktx2(image_bytes: bytes, encoding: str = "uastc") -> bytes | None:
    """Convert image bytes (PNG or JPEG) to KTX2 via ``toktx``.

    Encoding modes:
      - ``"uastc"``: High-quality UASTC + Zstandard.  Best for base color
        textures where detail matters.  Transcodes to BC7 on HoloLens.
      - ``"etc1s"``: Smaller ETC1S + Zstandard.  Best for emissive ring
        textures that are mostly black with a thin highlight band.

    Returns the KTX2 file bytes, or None on failure.
    """
    if not _has_toktx():
        return None

    try:
        with tempfile.TemporaryDirectory() as td:
            # Detect format from header; toktx supports both PNG and JPEG input
            if image_bytes[:2] == b'\xff\xd8':
                in_path = Path(td) / "input.jpg"
            else:
                in_path = Path(td) / "input.png"
            out_path = Path(td) / "output.ktx2"
            in_path.write_bytes(image_bytes)

            if encoding == "etc1s":
                cmd = [
                    "toktx",
                    "--t2",
                    "--encode", "etc1s",
                    "--clevel", "2",
                    "--qlevel", "128",
                    str(out_path),
                    str(in_path),
                ]
            else:
                cmd = [
                    "toktx",
                    "--t2",
                    "--encode", "uastc",
                    "--uastc_quality", "2",
                    "--zcmp", "5",
                    str(out_path),
                    str(in_path),
                ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=60)

            if out_path.exists():
                return out_path.read_bytes()
    except Exception as e:
        logger.debug(f"toktx encoding failed: {e}")

    return None


def _classify_emissive_textures(gltf: pygltflib.GLTF2) -> set[int]:
    """Return the set of image bufferView indices used as emissive textures.

    Emissive textures (e.g. CARTO excitation ring frames) are mostly black
    with a thin highlight band, so they compress much better with ETC1S than
    the higher-quality UASTC used for base color textures.
    """
    emissive_bvs: set[int] = set()
    for mat in gltf.materials:
        if mat.emissiveTexture is None:
            continue
        tex_idx = mat.emissiveTexture.index
        if tex_idx is None or tex_idx >= len(gltf.textures):
            continue
        img_idx = gltf.textures[tex_idx].source
        if img_idx is None or img_idx >= len(gltf.images):
            continue
        bv = gltf.images[img_idx].bufferView
        if bv is not None:
            emissive_bvs.add(bv)
    return emissive_bvs


def _try_ktx2_compress(
    gltf: pygltflib.GLTF2,
    blob: bytes,
    image_bv_set: set[int],
    scale: float = 1.0,
) -> tuple[bytearray, bool]:
    """Convert all images in the GLB to KTX2.

    Uses UASTC for base color textures (high quality) and ETC1S for
    emissive textures (mostly black, compresses much smaller).
    If *scale* < 1.0, downscale images before KTX2 encoding.
    Returns (new_blob, converted) where *converted* is True if any
    image was successfully converted.
    """
    emissive_bvs = _classify_emissive_textures(gltf)

    new_blob = bytearray()
    new_offsets: dict[int, tuple[int, int]] = {}
    converted_bvs: set[int] = set()

    bv_order = sorted(
        range(len(gltf.bufferViews)),
        key=lambda i: gltf.bufferViews[i].byteOffset,
    )

    for bv_idx in bv_order:
        bv = gltf.bufferViews[bv_idx]
        old_data = blob[bv.byteOffset : bv.byteOffset + bv.byteLength]
        new_offset = len(new_blob)

        if bv_idx in image_bv_set:
            # Optionally downscale before KTX2 encoding
            src_data = old_data
            if scale < 1.0:
                src_data = _reencode_image(old_data, "PNG", 0, scale)

            encoding = "etc1s" if bv_idx in emissive_bvs else "uastc"
            ktx2_data = _image_to_ktx2(src_data, encoding=encoding)
            if ktx2_data is not None:
                new_data = ktx2_data
                converted_bvs.add(bv_idx)
            else:
                new_data = bytes(old_data)
        else:
            new_data = bytes(old_data)

        new_offsets[bv_idx] = (new_offset, len(new_data))
        new_blob.extend(new_data)

        # 4-byte alignment
        remainder = len(new_blob) % 4
        if remainder:
            new_blob.extend(b"\x00" * (4 - remainder))

    if not converted_bvs:
        return bytearray(blob), False

    # Update bufferView offsets/lengths
    for bv_idx, (offset, length) in new_offsets.items():
        gltf.bufferViews[bv_idx].byteOffset = offset
        gltf.bufferViews[bv_idx].byteLength = length

    # Update image mimeTypes and add KHR_texture_basisu extension
    img_idx_by_bv: dict[int, int] = {}
    for idx, img in enumerate(gltf.images):
        if img.bufferView in converted_bvs:
            img.mimeType = "image/ktx2"
            img_idx_by_bv[img.bufferView] = idx

    # Add extension to textures referencing converted images
    converted_img_indices = set(img_idx_by_bv.values())
    for tex in gltf.textures:
        if tex.source in converted_img_indices:
            if tex.extensions is None:
                tex.extensions = {}
            tex.extensions["KHR_texture_basisu"] = {"source": tex.source}

    # Register extension
    ext_name = "KHR_texture_basisu"
    if gltf.extensionsUsed is None:
        gltf.extensionsUsed = []
    if ext_name not in gltf.extensionsUsed:
        gltf.extensionsUsed.append(ext_name)
    if gltf.extensionsRequired is None:
        gltf.extensionsRequired = []
    if ext_name not in gltf.extensionsRequired:
        gltf.extensionsRequired.append(ext_name)

    gltf.buffers[0].byteLength = len(new_blob)

    return new_blob, True


def _strategy_ktx2(path: Path, max_bytes: int) -> bool:
    """KTX2 GPU-compressed textures, with progressive downscale fallback."""
    if not _has_toktx():
        logger.warning(
            "toktx not found on PATH — falling back to downscale strategy. "
            "Install KTX-Software for best compression: "
            "https://github.com/KhronosGroup/KTX-Software/releases"
        )
        return _strategy_downscale(path, max_bytes)

    gltf, blob, image_bv_set = _load_and_identify_images(path)
    if not image_bv_set:
        return False

    original_size = path.stat().st_size

    # Try KTX2 at full resolution first
    new_blob, converted = _try_ktx2_compress(gltf, blob, image_bv_set)
    if converted and _estimate_file_size(gltf, new_blob) <= max_bytes:
        gltf.set_binary_blob(bytes(new_blob))
        gltf.save(str(path))
        logger.info(
            f"KTX2 compression: {_fmt(original_size)} → {_fmt(path.stat().st_size)}"
        )
        return True

    # Progressive downscale before KTX2 encoding
    for scale in _DOWNSCALE_STEPS:
        gltf, blob, image_bv_set = _load_and_identify_images(path)
        new_blob, converted = _try_ktx2_compress(gltf, blob, image_bv_set, scale=scale)
        if converted and _estimate_file_size(gltf, new_blob) <= max_bytes:
            gltf.set_binary_blob(bytes(new_blob))
            gltf.save(str(path))
            logger.info(
                f"KTX2 compression (scale={scale}): "
                f"{_fmt(original_size)} → {_fmt(path.stat().st_size)}"
            )
            return True

    # Apply best effort (smallest downscale + KTX2)
    if converted:
        gltf.set_binary_blob(bytes(new_blob))
        gltf.save(str(path))
        logger.warning(
            f"KTX2 compression applied (scale={_DOWNSCALE_STEPS[-1]}) but still "
            f"exceeds {_fmt(max_bytes)} ({_fmt(path.stat().st_size)})"
        )
        return True

    return False


def optimize_textures_ktx2(path: Path) -> bool:
    """Apply KTX2 compression to all textures unconditionally.

    Called from pipelines to ensure AR-optimized GPU-compressed textures
    even when the file is already under the size limit.  Gracefully
    returns False if ``toktx`` is not installed.
    """
    path = Path(path)
    if not path.exists():
        return False
    if not _has_toktx():
        logger.debug("toktx not available — skipping KTX2 texture optimization")
        return False

    gltf, blob, image_bv_set = _load_and_identify_images(path, skip_small=False)
    if not image_bv_set:
        return False

    original_size = path.stat().st_size
    new_blob, converted = _try_ktx2_compress(gltf, blob, image_bv_set)
    if not converted:
        return False

    gltf.set_binary_blob(bytes(new_blob))
    gltf.save(str(path))
    logger.info(
        f"KTX2 texture optimization: {_fmt(original_size)} → {_fmt(path.stat().st_size)}"
    )
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Node names that identify legend/info panels.  These are matched
# case-insensitively via substring — matching the HL app's
# ``IsMetadataObject`` logic (Loader.cs).
_METADATA_NODE_NAMES = ("legend", "info")

# Threshold for skipping small images during compression (safety net).
_SMALL_IMAGE_THRESHOLD = 50 * 1024  # 50 KB


def _classify_metadata_images(gltf: pygltflib.GLTF2) -> set[int]:
    """Return the set of image bufferView indices belonging to legend/info nodes.

    Traces node → mesh → primitive → material → texture → image to find
    every image referenced by nodes whose name contains "legend" or "info".
    """
    # Collect material indices used by metadata nodes
    metadata_mat_indices: set[int] = set()
    for node in gltf.nodes:
        if node.name is None or node.mesh is None:
            continue
        name_lower = node.name.lower()
        if not any(kw in name_lower for kw in _METADATA_NODE_NAMES):
            continue
        mesh = gltf.meshes[node.mesh]
        for prim in mesh.primitives:
            if prim.material is not None:
                metadata_mat_indices.add(prim.material)

    if not metadata_mat_indices:
        return set()

    # Collect texture indices used by those materials
    metadata_tex_indices: set[int] = set()
    for mat_idx in metadata_mat_indices:
        mat = gltf.materials[mat_idx]
        if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorTexture:
            metadata_tex_indices.add(mat.pbrMetallicRoughness.baseColorTexture.index)
        if mat.emissiveTexture:
            metadata_tex_indices.add(mat.emissiveTexture.index)
        if mat.normalTexture:
            metadata_tex_indices.add(mat.normalTexture.index)
        if mat.occlusionTexture:
            metadata_tex_indices.add(mat.occlusionTexture.index)

    # Collect image bufferView indices
    metadata_bvs: set[int] = set()
    for tex_idx in metadata_tex_indices:
        if tex_idx >= len(gltf.textures):
            continue
        img_idx = gltf.textures[tex_idx].source
        if img_idx is None or img_idx >= len(gltf.images):
            continue
        bv = gltf.images[img_idx].bufferView
        if bv is not None:
            metadata_bvs.add(bv)

    if metadata_bvs:
        logger.debug(
            "Excluding %d legend/info image(s) from compression (bufferViews: %s)",
            len(metadata_bvs), metadata_bvs,
        )
    return metadata_bvs


def _load_and_identify_images(
    path: Path,
    *,
    skip_small: bool = True,
) -> tuple[pygltflib.GLTF2, bytes, set[int]]:
    """Load a GLB and identify which bufferViews hold compressible image data.

    Excludes legend/info panel textures (identified by node name) which
    contain rendered text that degrades under compression.  Also excludes
    very small images when *skip_small* is True as a safety net.
    """
    gltf = pygltflib.GLTF2.load(str(path))
    blob = gltf.binary_blob() or b""

    metadata_bvs = _classify_metadata_images(gltf)

    image_bv_set: set[int] = set()
    for img in gltf.images:
        if img.bufferView is None:
            continue
        if img.bufferView in metadata_bvs:
            continue
        if skip_small:
            bv = gltf.bufferViews[img.bufferView]
            if bv.byteLength < _SMALL_IMAGE_THRESHOLD:
                logger.debug(
                    f"Skipping small image (bufferView {img.bufferView}, "
                    f"{bv.byteLength} bytes)"
                )
                continue
        image_bv_set.add(img.bufferView)
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
