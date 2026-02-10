"""Marching cubes method: basic isosurface extraction."""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from skimage import measure

from dicom2glb.core.types import ConversionResult, MaterialConfig, MeshData, MethodParams
from dicom2glb.core.volume import DicomVolume
from dicom2glb.glb.materials import default_material
from dicom2glb.methods.base import ConversionMethod
from dicom2glb.methods.registry import register_method


@register_method("marching-cubes")
class MarchingCubesMethod(ConversionMethod):
    """Basic isosurface extraction at configurable threshold."""

    description = "Basic isosurface extraction at configurable threshold."
    recommended_for = "Quick preview of any modality."

    def convert(self, volume: DicomVolume, params: MethodParams) -> ConversionResult:
        start = time.time()
        warnings = []

        threshold = params.threshold
        if threshold is None:
            threshold = _auto_threshold(volume.voxels)
            warnings.append(f"Auto-detected threshold: {threshold:.1f}")

        # Handle multi-threshold
        if params.multi_threshold:
            return _multi_threshold_extract(volume, params, start)

        # Run marching cubes
        voxels = volume.voxels
        spacing = volume.spacing

        try:
            verts, faces, normals, _ = measure.marching_cubes(
                voxels, level=threshold, spacing=spacing
            )
        except ValueError:
            raise ValueError(
                f"Marching cubes failed. The threshold {threshold:.1f} may be "
                "outside the data range "
                f"[{voxels.min():.1f}, {voxels.max():.1f}]."
            )

        mesh = MeshData(
            vertices=np.array(verts, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32),
            normals=np.array(normals, dtype=np.float32),
            structure_name="isosurface",
            material=default_material(),
        )

        return ConversionResult(
            meshes=[mesh],
            method_name="marching-cubes",
            processing_time=time.time() - start,
            warnings=warnings,
        )

    def supports_animation(self) -> bool:
        return True


def _auto_threshold(voxels: np.ndarray) -> float:
    """Estimate a good threshold using Otsu's method on non-zero voxels."""
    from skimage.filters import threshold_otsu

    vmin, vmax = float(voxels.min()), float(voxels.max())
    nonzero = voxels[voxels > 0]
    if len(nonzero) == 0:
        return vmax / 2

    try:
        threshold = float(threshold_otsu(nonzero))
    except ValueError:
        threshold = vmax / 2

    # Clamp to strictly within data range so marching cubes can find a surface
    if threshold >= vmax:
        threshold = (vmin + vmax) / 2
    elif threshold <= vmin:
        threshold = (vmin + vmax) / 2

    return threshold


def _multi_threshold_extract(
    volume: DicomVolume, params: MethodParams, start: float
) -> ConversionResult:
    """Extract multiple isosurfaces at different thresholds."""
    meshes = []
    warnings = []
    spacing = volume.spacing

    for layer in params.multi_threshold:
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                volume.voxels, level=layer.threshold, spacing=spacing
            )
            mesh = MeshData(
                vertices=np.array(verts, dtype=np.float32),
                faces=np.array(faces, dtype=np.int32),
                normals=np.array(normals, dtype=np.float32),
                structure_name=layer.label,
                material=layer.material,
            )
            meshes.append(mesh)
        except ValueError:
            warnings.append(
                f"Threshold {layer.threshold} for '{layer.label}' produced no surface"
            )

    if not meshes:
        raise ValueError("No valid surfaces extracted from multi-threshold config")

    return ConversionResult(
        meshes=meshes,
        method_name="marching-cubes",
        processing_time=time.time() - start,
        warnings=warnings,
    )
