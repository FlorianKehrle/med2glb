"""TotalSegmentator method: AI cardiac segmentation for CT."""

from __future__ import annotations

import time

import numpy as np

from dicom2glb.core.types import ConversionResult, MeshData, MethodParams
from dicom2glb.core.volume import DicomVolume
from dicom2glb.glb.materials import get_cardiac_material
from dicom2glb.methods.base import ConversionMethod, ProgressCallback
from dicom2glb.methods.registry import register_method

# heartchambers_highres label indices (1-based) mapped to display names.
# These are the integer labels in the multilabel nifti output.
HEARTCHAMBERS_LABELS = {
    1: "myocardium",
    2: "left_atrium",
    3: "left_ventricle",
    4: "right_atrium",
    5: "right_ventricle",
    6: "aorta",
    7: "pulmonary_artery",
}


@register_method("totalseg")
class TotalSegmentatorMethod(ConversionMethod):
    """AI cardiac segmentation via TotalSegmentator."""

    description = "AI cardiac chamber segmentation via TotalSegmentator (heartchambers_highres)."
    recommended_for = "Cardiac CT with contrast."
    requires_ai = True

    @classmethod
    def check_dependencies(cls) -> tuple[bool, str]:
        try:
            import totalsegmentator  # noqa: F401
            return True, "TotalSegmentator installed."
        except ImportError:
            return False, "Install with: pip install dicom2glb[ai]"

    def convert(
        self,
        volume: DicomVolume,
        params: MethodParams,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        start = time.time()

        def _report(desc: str, current: int | None = None, total: int | None = None):
            if progress is not None:
                progress(desc, current, total)

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
        _report("Preparing volume for TotalSegmentator...")
        affine = _build_affine(volume)
        nifti_img = nib.Nifti1Image(
            volume.voxels.transpose(2, 1, 0),  # [X, Y, Z] for nifti
            affine=affine,
        )

        # Run TotalSegmentator with heartchambers_highres task for
        # individual cardiac structures (myocardium, chambers, great vessels).
        _report("Running TotalSegmentator heartchambers_highres...")
        segmentation = totalsegmentator(
            nifti_img,
            task="heartchambers_highres",
            ml=True,
        )
        seg_data = segmentation.get_fdata().transpose(2, 1, 0)  # Back to [Z, Y, X]

        # Extract meshes for each cardiac structure
        meshes = []
        spacing = volume.spacing
        total_structures = len(HEARTCHAMBERS_LABELS)

        for i, (label_idx, structure_name) in enumerate(
            HEARTCHAMBERS_LABELS.items(), 1
        ):
            _report(f"Extracting {structure_name}...", i, total_structures)
            mask = seg_data == label_idx
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


