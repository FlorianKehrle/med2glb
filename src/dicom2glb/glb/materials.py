"""PBR material definitions with cardiac structure color map."""

from __future__ import annotations

from dicom2glb.core.types import MaterialConfig

# Default PBR materials for segmented cardiac structures
CARDIAC_MATERIALS: dict[str, MaterialConfig] = {
    "left_ventricle": MaterialConfig(
        base_color=(0.85, 0.20, 0.20), alpha=0.9, name="Left Ventricle"
    ),
    "right_ventricle": MaterialConfig(
        base_color=(0.20, 0.40, 0.85), alpha=0.9, name="Right Ventricle"
    ),
    "left_atrium": MaterialConfig(
        base_color=(0.90, 0.45, 0.45), alpha=0.7, name="Left Atrium"
    ),
    "right_atrium": MaterialConfig(
        base_color=(0.45, 0.55, 0.90), alpha=0.7, name="Right Atrium"
    ),
    "myocardium": MaterialConfig(
        base_color=(0.80, 0.60, 0.50), alpha=1.0, name="Myocardium"
    ),
    "aorta": MaterialConfig(
        base_color=(0.90, 0.30, 0.30), alpha=0.8, name="Aorta"
    ),
    "pulmonary_artery": MaterialConfig(
        base_color=(0.30, 0.30, 0.90), alpha=0.8, name="Pulmonary Artery"
    ),
    "unknown": MaterialConfig(
        base_color=(0.70, 0.70, 0.70), alpha=0.8, name="Unknown"
    ),
}


def get_cardiac_material(structure_name: str) -> MaterialConfig:
    """Get the PBR material for a cardiac structure name."""
    return CARDIAC_MATERIALS.get(structure_name, CARDIAC_MATERIALS["unknown"])


def default_material(
    alpha: float = 1.0,
    color: tuple[float, float, float] = (0.8, 0.2, 0.2),
) -> MaterialConfig:
    """Create a default material with configurable alpha and color."""
    return MaterialConfig(base_color=color, alpha=alpha, name="Default")
