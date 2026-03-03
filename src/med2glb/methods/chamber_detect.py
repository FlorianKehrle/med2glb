"""Chamber detect method: multi-structure cardiac chamber detection via intensity heuristics."""

from __future__ import annotations

import logging
import time

import numpy as np

from med2glb.core.types import ConversionResult, MeshData, MethodParams
from med2glb.core.volume import DicomVolume
from med2glb.glb.materials import get_cardiac_material
from med2glb.methods.base import ConversionMethod, ProgressCallback
from med2glb.methods.registry import register_method

logger = logging.getLogger(__name__)


@register_method("chamber-detect")
class ChamberDetectMethod(ConversionMethod):
    """Multi-structure cardiac chamber detection using intensity heuristics."""

    description = "Multi-structure cardiac chamber detection (intensity heuristic)."
    recommended_for = "3D echo, cardiac volumes with contrast."

    def convert(
        self,
        volume: DicomVolume,
        params: MethodParams,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        start = time.time()
        warnings = []

        def _report(desc: str, current: int | None = None, total: int | None = None):
            if progress is not None:
                progress(desc, current, total)

        _report("Running chamber detection...")
        try:
            masks = _detect_chambers(volume)
        except Exception as e:
            logger.warning(f"Chamber detection failed: {e}")
            raise ValueError(
                f"Chamber detection failed: {e}. "
                "Try --method classical as fallback."
            )

        from skimage import measure

        meshes = []
        spacing = volume.spacing
        total_structures = len(masks)

        for i, (structure_name, mask) in enumerate(masks.items(), 1):
            _report(f"Extracting {structure_name}...", i, total_structures)
            if not mask.any():
                continue

            try:
                verts, faces, normals, _ = measure.marching_cubes(
                    mask.astype(np.float32), level=0.5, spacing=spacing
                )
                material = get_cardiac_material(structure_name)
                mesh = MeshData(
                    vertices=np.array(verts, dtype=np.float32),
                    faces=np.array(faces, dtype=np.int32),
                    normals=np.array(normals, dtype=np.float32),
                    structure_name=structure_name,
                    material=material,
                )
                meshes.append(mesh)
            except ValueError:
                warnings.append(f"No surface extracted for {structure_name}")

        if not meshes:
            raise ValueError(
                "Chamber detection found no segmentable structures. "
                "Try --method classical as fallback."
            )

        return ConversionResult(
            meshes=meshes,
            method_name="chamber-detect",
            processing_time=time.time() - start,
            warnings=warnings,
        )

    def supports_animation(self) -> bool:
        return True


def _detect_chambers(volume: DicomVolume) -> dict[str, np.ndarray]:
    """Detect cardiac chambers using intensity-based heuristics.

    Returns dict mapping structure_name -> binary mask [Z, Y, X].

    Approach:
    1. Normalize volume to [0, 255]
    2. Otsu threshold to separate blood pool from tissue
    3. Morphological cleanup and connected component labeling
    4. Assign largest components to heart chambers by size
    """
    # Normalize volume
    voxels = volume.voxels.copy()
    vmin, vmax = voxels.min(), voxels.max()
    if vmax > vmin:
        voxels = (voxels - vmin) / (vmax - vmin) * 255.0
    else:
        raise ValueError("Volume has uniform intensity — cannot segment")

    logger.info("Running chamber detection...")

    masks = _segment_echo_heuristic(voxels, volume)

    return masks


def _segment_echo_heuristic(
    voxels: np.ndarray, volume: DicomVolume
) -> dict[str, np.ndarray]:
    """Segment echo data using intensity-based heuristics.

    Produces reasonable results for echo data where:
    - Blood pool appears as high-intensity regions
    - Myocardium appears as medium-intensity tissue
    - Background is low intensity
    """
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import ball

    # Compute thresholds
    nonzero = voxels[voxels > 10]
    if len(nonzero) == 0:
        raise ValueError("No significant signal in volume")

    try:
        t1 = threshold_otsu(nonzero)
    except ValueError:
        t1 = nonzero.mean()

    # Blood pool: high intensity
    blood = voxels > t1 * 1.2
    blood = ndimage.binary_closing(blood, structure=ball(2))
    blood = ndimage.binary_opening(blood, structure=ball(1))

    # Myocardium: medium intensity around blood pool
    tissue = (voxels > t1 * 0.5) & (voxels <= t1 * 1.2)
    tissue = ndimage.binary_closing(tissue, structure=ball(2))

    # Dilate blood pool and intersect with tissue for myocardium approximation
    blood_dilated = ndimage.binary_dilation(blood, structure=ball(3))
    myocardium = tissue & blood_dilated & (~blood)

    # Label connected components of blood pool as chambers
    labeled_blood, n_components = ndimage.label(blood)

    masks = {}
    if myocardium.any():
        masks["myocardium"] = myocardium

    # Assign largest blood components to heart chambers
    if n_components >= 1:
        sizes = ndimage.sum(blood, labeled_blood, range(1, n_components + 1))
        sorted_labels = np.argsort(sizes)[::-1] + 1

        chamber_names = ["left_ventricle", "left_atrium", "right_ventricle", "right_atrium"]
        for i, label_idx in enumerate(sorted_labels[:4]):
            if i < len(chamber_names):
                masks[chamber_names[i]] = labeled_blood == label_idx

    return masks
