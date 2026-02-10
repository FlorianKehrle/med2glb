"""Classical method: full pipeline with volume smoothing, adaptive threshold, morphological ops."""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from dicom2glb.core.types import ConversionResult, MeshData, MethodParams
from dicom2glb.core.volume import DicomVolume
from dicom2glb.glb.materials import default_material
from dicom2glb.methods.base import ConversionMethod
from dicom2glb.methods.registry import register_method


@register_method("classical")
class ClassicalMethod(ConversionMethod):
    """Full pipeline: volume smoothing, adaptive threshold, morphological ops, marching cubes."""

    description = (
        "Full pipeline: volume smoothing, adaptive threshold, "
        "morphological ops, marching cubes."
    )
    recommended_for = "3D echo data, noisy datasets."

    def convert(self, volume: DicomVolume, params: MethodParams) -> ConversionResult:
        start = time.time()
        warnings = []

        voxels = volume.voxels.copy()
        spacing = volume.spacing

        # Step 1: Gaussian volume smoothing to reduce noise
        sigma = _compute_sigma(spacing)
        voxels = ndimage.gaussian_filter(voxels, sigma=sigma)

        # Step 2: Threshold
        threshold = params.threshold
        if threshold is None:
            threshold = _adaptive_threshold(voxels)
            warnings.append(f"Auto-detected threshold: {threshold:.1f}")

        binary = voxels >= threshold

        # Step 3: Morphological operations to clean up
        binary = _morphological_cleanup(binary)

        # Step 4: Keep only largest connected component
        binary = _keep_largest_component(binary)

        if not binary.any():
            raise ValueError(
                "No valid structure found after morphological processing. "
                "Try adjusting --threshold or using --method marching-cubes."
            )

        # Step 5: Marching cubes on cleaned binary volume
        # Use smoothed voxels for better surface quality
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                voxels, level=threshold, spacing=spacing
            )
        except ValueError:
            # Fallback: marching cubes on binary mask
            verts, faces, normals, _ = measure.marching_cubes(
                binary.astype(np.float32), level=0.5, spacing=spacing
            )
            warnings.append("Used binary fallback for surface extraction")

        mesh = MeshData(
            vertices=np.array(verts, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32),
            normals=np.array(normals, dtype=np.float32),
            structure_name="cardiac_structure",
            material=default_material(),
        )

        return ConversionResult(
            meshes=[mesh],
            method_name="classical",
            processing_time=time.time() - start,
            warnings=warnings,
        )

    def supports_animation(self) -> bool:
        return True


def _compute_sigma(spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compute Gaussian sigma based on voxel spacing for ~1mm physical smoothing."""
    return tuple(1.0 / s for s in spacing)


def _adaptive_threshold(voxels: np.ndarray) -> float:
    """Compute adaptive threshold using Otsu's method on nonzero voxels."""
    nonzero = voxels[voxels > 0]
    if len(nonzero) == 0:
        return float(voxels.max()) / 2

    try:
        return float(threshold_otsu(nonzero))
    except ValueError:
        return float(voxels.max()) / 2


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Apply morphological closing then opening to clean noise."""
    # Closing: fill small holes
    struct = morphology.ball(2)
    binary = ndimage.binary_closing(binary, structure=struct, iterations=1)

    # Opening: remove small noise
    struct = morphology.ball(1)
    binary = ndimage.binary_opening(binary, structure=struct, iterations=1)

    return binary


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    labeled, num_features = ndimage.label(binary)
    if num_features <= 1:
        return binary

    # Find the largest component
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest = np.argmax(sizes) + 1

    return labeled == largest
