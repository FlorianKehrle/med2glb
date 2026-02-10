"""Core data types for the dicom2glb pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaterialConfig:
    """PBR material properties for a cardiac structure."""

    base_color: tuple[float, float, float] = (0.8, 0.2, 0.2)
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.7
    name: str = ""


@dataclass
class MeshData:
    """Triangulated surface mesh with material properties."""

    vertices: np.ndarray  # float32 [N, 3]
    faces: np.ndarray  # int32 [M, 3]
    normals: np.ndarray | None = None  # float32 [N, 3]
    structure_name: str = "unknown"
    material: MaterialConfig = field(default_factory=MaterialConfig)


@dataclass
class ThresholdLayer:
    """Single layer in multi-threshold extraction."""

    threshold: float
    label: str
    material: MaterialConfig


@dataclass
class MethodParams:
    """Configuration parameters for a conversion method."""

    threshold: float | None = None
    smoothing_iterations: int = 15
    target_faces: int = 80000
    multi_threshold: list[ThresholdLayer] | None = None


@dataclass
class ConversionResult:
    """Output from a conversion method."""

    meshes: list[MeshData]
    method_name: str
    processing_time: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class SeriesInfo:
    """Classification metadata for a DICOM series."""

    series_uid: str
    modality: str
    description: str
    file_count: int
    data_type: str  # "2D cine", "3D volume", "3D+T volume", "still image"
    detail: str  # "132 frames", "64 slices", etc.
    dimensions: str  # "600x800", "512x512x120"
    recommended_method: str  # "classical", "marching-cubes", etc.
    recommended_output: str  # "textured plane", "3D mesh", "animated 3D mesh"
    is_multiframe: bool = False
    number_of_frames: int = 0


@dataclass
class AnimatedResult:
    """Extends ConversionResult for temporal data with morph targets."""

    base_meshes: list[MeshData]
    morph_targets: list[list[np.ndarray]]  # Per-mesh vertex displacements per frame
    frame_times: list[float]
    loop_duration: float
    method_name: str = ""
    processing_time: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def meshes(self) -> list[MeshData]:
        """Alias for base_meshes for compatibility with ConversionResult."""
        return self.base_meshes
