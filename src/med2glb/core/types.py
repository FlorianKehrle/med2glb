"""Core data types for the med2glb pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MaterialConfig:
    """PBR material properties for a cardiac structure."""

    base_color: tuple[float, float, float] = (0.8, 0.2, 0.2)
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.7
    name: str = ""
    unlit: bool = False


@dataclass
class MeshData:
    """Triangulated surface mesh with material properties."""

    vertices: np.ndarray  # float32 [N, 3]
    faces: np.ndarray  # int32 [M, 3]
    normals: np.ndarray | None = None  # float32 [N, 3]
    structure_name: str = "unknown"
    material: MaterialConfig = field(default_factory=MaterialConfig)
    vertex_colors: np.ndarray | None = None  # float32 [N, 4] RGBA


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


@dataclass
class GallerySlice:
    """Single DICOM slice with metadata for gallery mode (no shape filtering)."""

    pixel_data: np.ndarray  # [Y, X] or [Y, X, 3], uint8-ready
    pixel_spacing: tuple[float, float]
    image_position: tuple[float, float, float] | None
    image_orientation: tuple[float, ...] | None  # 6 direction cosines
    instance_number: int
    filename: str
    rows: int
    cols: int
    temporal_index: int | None
    series_uid: str
    modality: str


@dataclass
class CartoPoint:
    """Single electro-anatomical mapping point from a CARTO _car.txt file."""

    point_id: int
    position: np.ndarray  # float64 [3] — X, Y, Z
    orientation: np.ndarray  # float64 [3]
    bipolar_voltage: float  # mV
    unipolar_voltage: float  # mV
    lat: float  # ms, NaN if sentinel -10000


@dataclass
class CartoMesh:
    """Surface mesh from a CARTO .mesh file."""

    mesh_id: int
    vertices: np.ndarray  # float64 [N, 3]
    faces: np.ndarray  # int32 [M, 3]
    normals: np.ndarray  # float64 [N, 3]
    group_ids: np.ndarray  # int32 [N] — vertex group IDs
    face_group_ids: np.ndarray  # int32 [M] — face group IDs
    mesh_color: tuple[float, float, float, float]  # RGBA default color
    color_names: list[str]  # names of per-vertex color channels
    structure_name: str = ""


@dataclass
class CartoStudy:
    """A complete CARTO export directory with meshes and mapping points."""

    meshes: list[CartoMesh]
    points: dict[str, list[CartoPoint]]  # keyed by mesh/map name
    version: str  # e.g. "4.0", "5.0", "6.0"
    study_name: str = ""


@dataclass
class CartoConfig:
    """Configuration for CARTO pipeline (from wizard or CLI flags)."""

    input_path: Path
    name: str = ""  # descriptive name (e.g. "LA_lat_sub2")
    output_dir: Path | None = None
    selected_mesh_indices: list[int] | None = None  # None = all
    coloring: str = "lat"
    subdivide: int = 2
    animate: bool = True       # default: both static + animated
    static: bool = True        # default: both
    vectors: str = "no"  # "no", "yes" (both with/without), "only" (vectors only)
    vector_mesh_indices: list[int] | None = None  # meshes suitable for vectors; None = all
    target_faces: int = 80000
    max_size_mb: int = 99
    compress_strategy: str = "draco"
    verbose: bool = False


@dataclass
class DicomConfig:
    """Configuration for DICOM pipeline (from wizard or CLI flags)."""

    input_path: Path
    name: str = ""  # descriptive name (e.g. "transplant-ct_mc_t200_s5_20k")
    output: Path | None = None
    method: str = "classical"
    format: str = "glb"
    animate: bool = False
    threshold: float | None = None
    smoothing: int = 15
    target_faces: int = 80000
    alpha: float = 1.0
    series_uid: str | None = None
    max_size_mb: int = 99
    compress_strategy: str = "draco"
    gallery: bool = False
    verbose: bool = False
