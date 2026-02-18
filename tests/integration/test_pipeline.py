"""Integration test: end-to-end DICOM directory to GLB file."""

from __future__ import annotations

from pathlib import Path

import pygltflib
import pytest

from med2glb.core.types import MethodParams
from med2glb.io.dicom_reader import InputType, load_dicom_directory
from med2glb.io.exporters import export_glb, export_stl
from med2glb.mesh.processing import process_mesh
from med2glb.methods.registry import _ensure_methods_loaded, get_method


@pytest.fixture(autouse=True)
def _load_methods():
    _ensure_methods_loaded()


def test_volume_to_glb(dicom_directory, tmp_path):
    """End-to-end: DICOM directory -> marching-cubes -> GLB."""
    input_type, volume = load_dicom_directory(dicom_directory)
    assert input_type == InputType.VOLUME

    method = get_method("marching-cubes")
    params = MethodParams(threshold=250.0, smoothing_iterations=5, target_faces=5000)
    result = method.convert(volume, params)

    # Process meshes
    processed = [process_mesh(m, smoothing_iterations=5, target_faces=5000) for m in result.meshes]
    result.meshes = processed

    # Export
    output = tmp_path / "test.glb"
    export_glb(result.meshes, output)

    assert output.exists()
    gltf = pygltflib.GLTF2.load(str(output))
    assert len(gltf.meshes) >= 1


def test_volume_to_stl(dicom_directory, tmp_path):
    """End-to-end: DICOM directory -> classical -> STL."""
    input_type, volume = load_dicom_directory(dicom_directory)

    method = get_method("classical")
    params = MethodParams(threshold=250.0, smoothing_iterations=5, target_faces=5000)
    result = method.convert(volume, params)

    processed = [process_mesh(m, smoothing_iterations=5, target_faces=5000) for m in result.meshes]
    result.meshes = processed

    output = tmp_path / "test.stl"
    export_stl(result.meshes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_single_slice_to_textured_glb(tmp_path):
    """End-to-end: single DICOM file -> textured plane GLB."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian

    # Create a synthetic single DICOM file
    dcm_path = tmp_path / "single.dcm"
    ds = FileDataset(str(dcm_path), Dataset(), preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.5, 0.5]
    ds.Modality = "CT"

    import numpy as np
    pixel_data = np.random.randint(0, 1000, (64, 64), dtype=np.uint16)
    ds.PixelData = pixel_data.tobytes()
    ds.save_as(str(dcm_path))

    # Load and verify single slice detection
    input_type, volume = load_dicom_directory(dcm_path)
    assert input_type == InputType.SINGLE_SLICE

    # Build textured plane GLB
    from med2glb.glb.texture import build_textured_plane_glb
    output = tmp_path / "plane.glb"
    build_textured_plane_glb(volume, output)

    assert output.exists()
    gltf = pygltflib.GLTF2.load(str(output))
    assert len(gltf.meshes) == 1
    assert len(gltf.textures) == 1
    assert len(gltf.images) == 1


def test_temporal_to_animated_glb(dicom_temporal_directory, tmp_path):
    """End-to-end: temporal DICOM -> animated GLB with morph targets."""
    input_type, data = load_dicom_directory(dicom_temporal_directory)
    assert input_type == InputType.TEMPORAL

    method = get_method("marching-cubes")
    params = MethodParams(threshold=250.0, smoothing_iterations=3, target_faces=2000)

    # Convert each frame
    from med2glb.core.volume import TemporalSequence
    assert isinstance(data, TemporalSequence)

    frame_results = []
    for frame in data.frames:
        result = method.convert(frame, params)
        processed = [process_mesh(m, smoothing_iterations=3, target_faces=2000) for m in result.meshes]
        result.meshes = processed
        frame_results.append(result)

    # Build morph targets
    from med2glb.mesh.temporal import build_morph_targets_from_frames
    animated = build_morph_targets_from_frames(frame_results, data.temporal_resolution)

    assert len(animated.base_meshes) >= 1
    assert len(animated.morph_targets) >= 1
    assert len(animated.frame_times) == data.frame_count

    # Export animated GLB
    from med2glb.glb.animation import build_animated_glb
    output = tmp_path / "animated.glb"
    build_animated_glb(animated, output)

    assert output.exists()
    gltf = pygltflib.GLTF2.load(str(output))
    assert len(gltf.animations) >= 1
