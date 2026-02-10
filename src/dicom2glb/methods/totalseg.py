"""TotalSegmentator method: AI cardiac segmentation for CT."""

from __future__ import annotations

import time

import numpy as np

from dicom2glb.core.types import ConversionResult, MeshData, MethodParams
from dicom2glb.core.volume import DicomVolume
from dicom2glb.glb.materials import get_cardiac_material
from dicom2glb.methods.base import ConversionMethod
from dicom2glb.methods.registry import register_method

# Mapping from TotalSegmentator labels to our cardiac structure names
TOTALSEG_CARDIAC_LABELS = {
    "heart_myocardium": "myocardium",
    "heart_atrium_left": "left_atrium",
    "heart_atrium_right": "right_atrium",
    "heart_ventricle_left": "left_ventricle",
    "heart_ventricle_right": "right_ventricle",
    "aorta": "aorta",
    "pulmonary_artery": "pulmonary_artery",
}


@register_method("totalseg")
class TotalSegmentatorMethod(ConversionMethod):
    """AI cardiac segmentation via TotalSegmentator."""

    description = "AI cardiac segmentation via TotalSegmentator."
    recommended_for = "Cardiac CT with contrast."
    requires_ai = True

    @classmethod
    def check_dependencies(cls) -> tuple[bool, str]:
        try:
            import totalsegmentator  # noqa: F401
            return True, "TotalSegmentator installed."
        except ImportError:
            return False, "Install with: pip install dicom2glb[ai]"

    def convert(self, volume: DicomVolume, params: MethodParams) -> ConversionResult:
        start = time.time()

        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError:
            raise ImportError(
                "TotalSegmentator not installed. Install with: pip install dicom2glb[ai]"
            )

        import nibabel as nib
        from skimage import measure

        warnings = []

        # Convert DicomVolume to nibabel nifti for TotalSegmentator
        affine = _build_affine(volume)
        nifti_img = nib.Nifti1Image(
            volume.voxels.transpose(2, 1, 0),  # [X, Y, Z] for nifti
            affine=affine,
        )

        # Run TotalSegmentator with cardiac task
        segmentation = totalsegmentator(
            nifti_img,
            task="total",
            fast=False,
            ml=True,
        )
        seg_data = segmentation.get_fdata().transpose(2, 1, 0)  # Back to [Z, Y, X]

        # Extract meshes for each cardiac structure
        meshes = []
        spacing = volume.spacing

        for label_name, structure_name in TOTALSEG_CARDIAC_LABELS.items():
            # Find the label index — TotalSegmentator uses integer labels
            # We need to check if this structure was segmented
            mask = _get_structure_mask(seg_data, label_name)
            if mask is None or not mask.any():
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
                warnings.append(f"No surface for {structure_name}")

        if not meshes:
            raise ValueError(
                "TotalSegmentator found no cardiac structures. "
                "Ensure input is a contrast-enhanced cardiac CT."
            )

        return ConversionResult(
            meshes=meshes,
            method_name="totalseg",
            processing_time=time.time() - start,
            warnings=warnings,
        )


def _build_affine(volume: DicomVolume) -> np.ndarray:
    """Build a nifti-style affine matrix from DICOM metadata."""
    affine = np.eye(4)
    affine[0, 0] = volume.pixel_spacing[1]  # X spacing
    affine[1, 1] = volume.pixel_spacing[0]  # Y spacing
    affine[2, 2] = volume.slice_thickness  # Z spacing

    pos = volume.image_position_first
    affine[0, 3] = pos[0]
    affine[1, 3] = pos[1]
    affine[2, 3] = pos[2]

    return affine


def _get_structure_mask(
    seg_data: np.ndarray, label_name: str
) -> np.ndarray | None:
    """Get binary mask for a specific structure from segmentation output.

    TotalSegmentator may output multi-label or multi-file segmentation.
    This handles the common multi-label integer array case.
    """
    # TotalSegmentator label mapping (subset for cardiac)
    # These label indices may vary by version
    label_map = {
        "heart_myocardium": 522,
        "heart_atrium_left": 525,
        "heart_atrium_right": 526,
        "heart_ventricle_left": 523,
        "heart_ventricle_right": 524,
        "aorta": 52,
        "pulmonary_artery": 57,
    }

    label_idx = label_map.get(label_name)
    if label_idx is None:
        return None

    mask = seg_data == label_idx
    if not mask.any():
        # Try adjacent labels in case of version differences
        return None

    return mask
