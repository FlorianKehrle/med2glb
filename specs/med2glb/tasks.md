# Tasks: med2glb — Medical Imaging to GLB Converter

**Input**: Design documents from `/specs/med2glb/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-contract.md
**Status**: All phases implemented. This document records the delivered task history.

## Format: `[ID] [Status] [Story] Description`

- **[P]**: Could run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1-US8)

---

## Phase 1: Setup ✅

- [x] T001 Create project directory structure (src/med2glb/ with io/, methods/, mesh/, glb/, core/, gallery/)
- [x] T002 Create pyproject.toml with dependencies (pydicom, numpy, scipy, scikit-image, trimesh, pygltflib, typer, rich, xatlas), [ai] extra, [dev] extra, and `med2glb` CLI entry point
- [x] T003 [P] Create src/med2glb/__init__.py and __main__.py
- [x] T004 [P] Create __init__.py files for all subpackages

---

## Phase 2: Foundational ✅

- [x] T005 [P] Implement core types (MeshData, MaterialConfig, MethodParams, ThresholdLayer, ConversionResult, AnimatedResult) in core/types.py
- [x] T006 [P] Implement DicomVolume and TemporalSequence in core/volume.py
- [x] T007 Implement ConversionMethod ABC in methods/base.py
- [x] T008 Implement method registry with @register_method in methods/registry.py
- [x] T009 Implement DICOM reader: series analysis, volume assembly, type detection in io/dicom_reader.py
- [x] T010 [P] Implement mesh processing: Taubin smoothing, quadric decimation, normal computation in mesh/processing.py
- [x] T011 [P] Implement PBR material definitions with cardiac color map in glb/materials.py
- [x] T012 Implement GLB builder with PBR materials in glb/builder.py
- [x] T013 Implement CLI skeleton with typer in cli.py
- [x] T014 [P] Implement multi-format exporter in io/exporters.py

---

## Phase 3: DICOM Animated GLB + Methods (US2 + US7) ✅

- [x] T015 [US2] Implement vendor-specific 3D echo reader in io/echo_reader.py
- [x] T016 [P] [US7] Implement marching-cubes method with auto-threshold and multi-threshold in methods/marching_cubes.py
- [x] T017 [P] [US7] Implement classical method with Otsu + region-growing in methods/classical.py
- [x] T018 [US2] Implement temporal mesh smoothing in mesh/temporal.py
- [x] T019 [US2] Implement consistent mesh topology via cKDTree correspondence in mesh/temporal.py
- [x] T020 [US2] Implement morph target animation builder in glb/animation.py
- [x] T021 [US2] Wire CLI --animate flag and end-to-end DICOM pipeline
- [x] T022 [US7] Wire CLI --list-methods and --method switching
- [x] T023 [US7] Implement compare method (side-by-side method comparison)

---

## Phase 4: CARTO 3 EP Mapping (US1) ✅

- [x] T024 [US1] Implement CARTO .mesh parser (v4, v5, v6) in io/carto_reader.py
- [x] T025 [US1] Implement CARTO _car.txt parser with version detection in io/carto_reader.py
- [x] T026 [US1] Implement CARTO directory detection and study loading in io/carto_reader.py
- [x] T027 [P] [US1] Implement clinical colormaps (LAT, bipolar, unipolar) in io/carto_colormaps.py
- [x] T028 [P] [US1] Implement sparse-to-dense point mapping (KDTree, IDW) in io/carto_mapper.py
- [x] T029 [US1] Implement Loop subdivision with metadata propagation in io/carto_mapper.py
- [x] T030 [US1] Implement xatlas UV unwrap + barycentric rasterization in glb/vertex_color_bake.py
- [x] T031 [US1] Implement CARTO static GLB builder with baked textures in glb/carto_builder.py
- [x] T032 [US1] Implement animated excitation ring via emissive overlay in glb/carto_builder.py
- [x] T033 [US1] Implement animated bake cache (shared UV + textures across variants) in glb/carto_builder.py
- [x] T034 [US1] Implement LAT face gradient computation (Gram matrix) in mesh/lat_vectors.py
- [x] T035 [US1] Implement streamline tracing (half-edge traversal, momentum coasting) in mesh/lat_vectors.py
- [x] T036 [US1] Implement animated dash generation for streamlines in mesh/lat_vectors.py
- [x] T037 [US1] Implement arrow/dash geometry builder in glb/arrow_builder.py
- [x] T038 [US1] Implement color legend + info panel in glb/legend_builder.py
- [x] T039 [US1] Implement vector quality assessment in cli_wizard.py
- [x] T040 [US1] Wire CARTO pipeline in _pipeline_carto.py
- [x] T041 [US1] Implement batch processing for multiple CARTO exports

---

## Phase 5: Cardiac CT/MRI Segmentation (US3) ✅

- [x] T042 [US3] Implement TotalSegmentator method in methods/totalseg.py
- [x] T043 [US3] Implement chamber-detect method in methods/chamber_detect.py
- [x] T044 [US3] Wire multi-structure GLB export with per-structure materials

---

## Phase 6: Interactive Wizard (US4) ✅

- [x] T045 [US4] Implement directory scanning and data detection in cli_wizard.py
- [x] T046 [US4] Implement CARTO wizard (mesh/coloring/animation/vector prompts) in cli_wizard.py
- [x] T047 [US4] Implement DICOM wizard (series/method/quality prompts) in cli_wizard.py
- [x] T048 [US4] Implement batch CARTO wizard in cli_wizard.py
- [x] T049 [US4] Wire wizard bypass when pipeline flags present in cli.py

---

## Phase 7: Gallery Mode (US5) ✅

- [x] T050 [US5] Implement gallery slice loader in gallery/loader.py
- [x] T051 [P] [US5] Implement individual GLBs in gallery/individual.py
- [x] T052 [P] [US5] Implement lightbox grid in gallery/lightbox.py
- [x] T053 [P] [US5] Implement spatial fan in gallery/spatial.py
- [x] T054 [US5] Implement shared gallery utilities in gallery/_glb_utils.py
- [x] T055 [US5] Wire gallery pipeline in _pipeline_gallery.py

---

## Phase 8: GLB Compression (US6) ✅

- [x] T056 [US6] Implement Draco compression strategy in glb/compress.py
- [x] T057 [US6] Implement texture downscale strategy in glb/compress.py
- [x] T058 [US6] Implement JPEG re-encoding strategy in glb/compress.py
- [x] T059 [US6] Implement KTX2 compression (toktx integration) in glb/compress.py
- [x] T060 [US6] Wire --compress CLI flow in cli.py

---

## Phase 9: Documentation & Polish (US8) ✅

- [x] T061 [US8] Write comprehensive README.md with CARTO and DICOM workflows
- [x] T062 [US8] Polish CLI help output with rich formatting
- [x] T063 [US8] Validate pyproject.toml and pip install in clean venv
- [x] T064 Implement conversion statistics logging in io/conversion_log.py
- [x] T065 Implement Rich console utilities in _console.py

---

## Phase 10: Testing ✅

- [x] T066 [P] Create test fixtures (synthetic DICOM, CARTO, mesh data) in tests/conftest.py
- [x] T067 [P] Unit tests for CARTO reader in tests/unit/test_carto_reader.py
- [x] T068 [P] Unit tests for CARTO mapper in tests/unit/test_carto_mapper.py
- [x] T069 [P] Unit tests for CARTO GLB builder in tests/unit/test_carto_glb.py
- [x] T070 [P] Unit tests for arrow builder in tests/unit/test_arrow_builder.py
- [x] T071 [P] Unit tests for legend builder in tests/unit/test_legend_builder.py
- [x] T072 [P] Unit tests for LAT vectors in tests/unit/test_lat_vectors.py
- [x] T073 [P] Unit tests for vertex color bake in tests/unit/test_vertex_color_bake.py
- [x] T074 [P] Unit tests for GLB builder in tests/unit/test_glb_builder.py
- [x] T075 [P] Unit tests for gallery in tests/unit/test_gallery.py
- [x] T076 [P] Unit tests for DICOM reader in tests/unit/test_dicom_reader.py
- [x] T077 [P] Unit tests for methods in tests/unit/test_methods.py
- [x] T078 [P] Unit tests for mesh processing in tests/unit/test_mesh_processing.py
- [x] T079 [P] Unit tests for DICOM pipeline in tests/unit/test_pipeline_dicom.py
- [x] T080 [P] Unit tests for compression in tests/unit/test_compress.py
- [x] T081 [P] Unit tests for CLI wizard in tests/unit/test_cli_wizard.py
- [x] T082 Integration test: CLI in tests/integration/test_cli.py
- [x] T083 Integration test: DICOM pipeline in tests/integration/test_pipeline.py
- [x] T084 Integration test: CARTO pipeline in tests/integration/test_carto_pipeline.py
- [x] T085 Integration test: animated cache in tests/integration/test_animated_cache.py
- [x] T086 Integration test: real DICOM in tests/integration/test_dicom_real.py
