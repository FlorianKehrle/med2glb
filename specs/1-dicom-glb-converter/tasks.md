# Tasks: DICOM to GLB Converter

**Input**: Design documents from `/specs/1-dicom-glb-converter/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-contract.md

**Tests**: Not explicitly requested in spec. Tests included in Polish phase for validation.

**Organization**: Tasks grouped by user story. US1 (3D echo animated) and US4 (method switching) are both P1 and tightly coupled — combined into one phase.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Project initialization, directory structure, build configuration

- [ ] T001 Create project directory structure per plan.md (src/med2glb/ with io/, methods/, mesh/, glb/, core/ subdirectories, tests/unit/, tests/integration/)
- [ ] T002 Create pyproject.toml with project metadata, dependencies (pydicom, numpy, scipy, scikit-image, trimesh, pyvista, pygltflib, typer, rich), [ai] optional extra (totalsegmentator, torch, monai), [dev] extra (pytest, pytest-cov), and `med2glb` CLI entry point
- [ ] T003 [P] Create .gitignore with Python patterns (__pycache__/, *.pyc, .venv/, dist/, *.egg-info/, .env*)
- [ ] T004 [P] Create src/med2glb/__init__.py with package version and src/med2glb/__main__.py for `python -m med2glb` support

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures, interfaces, and shared infrastructure that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 [P] Implement core types (MeshData, MaterialConfig, MethodParams, ThresholdLayer, ConversionResult, AnimatedResult) in src/med2glb/core/types.py per data-model.md
- [ ] T006 [P] Implement DicomVolume and TemporalSequence dataclasses in src/med2glb/core/volume.py per data-model.md
- [ ] T007 Implement ConversionMethod ABC with `convert(volume, params) -> ConversionResult` and `supports_animation() -> bool` in src/med2glb/methods/base.py
- [ ] T008 Implement method registry with @register_method decorator, list_methods(), get_method(), and availability checking in src/med2glb/methods/registry.py
- [ ] T009 Implement DICOM reader: load directory, group by Series Instance UID, detect input type (single/volume/temporal), assemble DicomVolume with metadata (pixel spacing, slice thickness, orientation) in src/med2glb/io/dicom_reader.py
- [ ] T010 [P] Implement mesh processing: Taubin smoothing (configurable iterations), quadric edge collapse decimation (target face count), hole filling, normal recalculation in src/med2glb/mesh/processing.py
- [ ] T011 [P] Implement PBR material definitions with cardiac structure color map (LV red, RV blue, LA pink, RA light blue, myocardium tan, aorta red, etc.) per data-model.md in src/med2glb/glb/materials.py
- [ ] T012 Implement basic GLB builder: create glTF scene, add meshes with PBR materials (alphaMode BLEND, metallicFactor 0.0, roughnessFactor 0.7), export via pygltflib in src/med2glb/glb/builder.py
- [ ] T013 Implement CLI skeleton with typer: INPUT_PATH argument, -o/--output, -m/--method, -f/--format, --animate, --threshold, --smoothing, --faces, --alpha, --multi-threshold, --series, --list-methods, --list-series, -v/--verbose flags per contracts/cli-contract.md in src/med2glb/cli.py
- [ ] T014 [P] Create __init__.py files for all subpackages: io/, methods/, mesh/, glb/, core/

**Checkpoint**: Foundation ready — method registry, DICOM reading, mesh processing, GLB export, and CLI skeleton all functional

---

## Phase 3: User Story 1 + User Story 4 - 3D Echo Animated GLB + Method Switching (Priority: P1)

**Goal**: Convert 3D echo DICOM data to animated GLB with morph targets showing the beating heart. User can switch between methods via --method flag.

**Independent Test**: Provide 3D echo DICOM files, run `med2glb ./echo/ -o heart.glb --animate`, open resulting GLB in AR viewer to see smooth cardiac loop. Run `med2glb --list-methods` to see available methods. Switch methods with `--method marching-cubes` vs `--method classical`.

### Implementation

- [ ] T015 [US1] Implement vendor-specific 3D echo DICOM reader for Philips and GE formats (reference SlicerHeart) with temporal frame detection via Temporal Position Index / Instance Number in src/med2glb/io/echo_reader.py
- [ ] T016 [P] [US1] Implement marching-cubes method: isosurface extraction at configurable threshold using scikit-image marching_cubes, then Taubin smoothing + decimation in src/med2glb/methods/marching_cubes.py
- [ ] T017 [P] [US1] Implement classical method: Gaussian volume smoothing (scipy), adaptive thresholding, morphological ops (scikit-image), marching cubes, Taubin smoothing, quadric decimation, hole filling in src/med2glb/methods/classical.py
- [ ] T018 [US1] Implement temporal mesh smoothing: vertex position smoothing across time frames using weighted moving average to prevent animation flickering in src/med2glb/mesh/temporal.py
- [ ] T019 [US1] Implement consistent mesh topology for animation: extract mesh from first frame, deform vertices for subsequent frames via nearest-surface-point correspondence to ensure identical vertex count across all morph targets in src/med2glb/mesh/temporal.py
- [ ] T020 [US1] Implement morph target animation: create glTF morph targets from per-frame vertex displacements, build AnimationChannel targeting node weights, build AnimationSampler with keyframe times, configure looping in src/med2glb/glb/animation.py
- [ ] T021 [P] [US1] Implement multi-format exporter: GLB (animated via pygltflib), STL (trimesh), OBJ (trimesh) with format selection in src/med2glb/io/exporters.py
- [ ] T022 [US1] Wire CLI --animate flag: detect temporal data, run per-frame conversion, apply temporal smoothing, build morph targets, export animated GLB with progress bar in src/med2glb/cli.py
- [ ] T023 [US1] Wire CLI --list-methods: query registry, display method name, description, recommended data types, dependency status (installed/not installed) per contracts/cli-contract.md in src/med2glb/cli.py
- [ ] T024 [US1] Wire CLI --method switching: validate method exists, check optional deps available, graceful error with install instructions if AI method missing in src/med2glb/cli.py
- [ ] T025 [US1] Implement end-to-end pipeline integration: DICOM dir → detect type → select method → convert (per frame if temporal) → mesh processing → temporal smoothing → GLB export with progress indication in src/med2glb/cli.py

**Checkpoint**: 3D echo → animated GLB works end-to-end. Methods switchable via --method. --list-methods shows available options.

---

## Phase 4: User Story 2 - Cardiac CT/MRI Segmented GLB (Priority: P2)

**Goal**: Convert cardiac CT to GLB with individually segmented, colored, transparent cardiac structures via TotalSegmentator.

**Independent Test**: Provide cardiac CT DICOM directory, run `med2glb ./ct/ -o heart.glb --method totalseg`, verify GLB contains separate colored structures (LV, RV, LA, RA, aorta, myocardium) with transparency.

### Implementation

- [ ] T026 [US2] Implement TotalSegmentator method: wrap totalsegmentator API, run cardiac segmentation, extract per-structure binary masks, generate mesh per structure with assigned cardiac color/material in src/med2glb/methods/totalseg.py
- [ ] T027 [US2] Implement multi-threshold extraction: parse --multi-threshold CLI arg ("val:label:alpha,..."), extract isosurface per threshold, assign material per layer in src/med2glb/methods/marching_cubes.py
- [ ] T028 [US2] Wire multi-structure GLB export: multiple meshes in single GLB scene, each with distinct PBR material (color + alpha from cardiac color map) in src/med2glb/glb/builder.py
- [ ] T029 [US2] Wire CLI --multi-threshold flag and --series flag for series selection in src/med2glb/cli.py

**Checkpoint**: Cardiac CT → segmented GLB with colored structures. Multi-threshold layered output works.

---

## Phase 5: User Story 3 - Single DICOM Slice to GLB (Priority: P3)

**Goal**: Convert a single DICOM image to a GLB textured plane viewable in AR.

**Independent Test**: Provide single .dcm file, run `med2glb image.dcm -o output.glb`, verify GLB shows the image as a textured 3D plane with correct aspect ratio.

### Implementation

- [ ] T030 [US3] Implement single-slice detection and handling: detect single DICOM file, extract pixel data and metadata, create textured quad mesh with correct aspect ratio from pixel spacing in src/med2glb/io/dicom_reader.py
- [ ] T031 [US3] Implement textured plane GLB export: create quad mesh, embed DICOM image as PNG texture in GLB, apply as baseColorTexture material in src/med2glb/glb/builder.py

**Checkpoint**: Single DICOM → textured GLB plane works.

---

## Phase 6: User Story 5 - Documentation and Install Experience (Priority: P2)

**Goal**: Comprehensive README, polished CLI help, and smooth pip install experience.

**Independent Test**: Follow README on a clean Python environment, successfully install and convert a dataset within 10 minutes.

### Implementation

- [ ] T032 [US5] Write comprehensive README.md: project overview, installation (pip install med2glb, pip install med2glb[ai]), quick start examples, method comparison table (marching-cubes vs classical vs totalseg vs medsam2 with recommended data types), CLI reference, AR viewer compatibility matrix (Android Scene Viewer, model-viewer web component), output format comparison, contributing guide
- [ ] T033 [US5] Polish CLI --help output: add rich formatting, usage examples in epilog, method descriptions, version display per contracts/cli-contract.md in src/med2glb/cli.py
- [ ] T034 [US5] Validate pyproject.toml: verify pip install works in clean venv, verify med2glb entry point resolves, verify [ai] extra installs correctly, verify [dev] extra works

**Checkpoint**: README complete. Clean pip install works. CLI help is informative.

---

## Phase 7: MedSAM2 AI Method (Priority: P1 extension)

**Goal**: Add MedSAM2-based segmentation for 3D echo, including per-frame segmentation for animated segmented output.

**Independent Test**: Run `med2glb ./echo/ -o heart.glb --method medsam2 --animate`, verify segmented animated output with distinct structures.

### Implementation

- [ ] T035 [US1] Implement MedSAM2 method: wrap MedSAM2 API, run cardiac segmentation on echo volumes, extract per-structure masks, generate meshes with cardiac materials in src/med2glb/methods/medsam2.py
- [ ] T036 [US1] Implement per-frame AI segmentation: run MedSAM2 on each temporal frame, maintain structure consistency across frames, combine with temporal smoothing and morph target animation in src/med2glb/methods/medsam2.py

**Checkpoint**: MedSAM2 segmentation produces multi-structure animated GLB from 3D echo.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Testing, error handling, edge cases, and quality improvements

- [ ] T037 [P] Create test fixtures: synthetic DICOM data (small 3D volume, temporal sequence, single slice) using pydicom in tests/conftest.py
- [ ] T038 [P] Unit tests for DICOM reader (series grouping, type detection, metadata extraction) in tests/unit/test_dicom_reader.py
- [ ] T039 [P] Unit tests for mesh processing (Taubin smoothing, decimation, hole filling) in tests/unit/test_mesh_processing.py
- [ ] T040 [P] Unit tests for GLB builder (structure validation, material assignment, morph targets) in tests/unit/test_glb_builder.py
- [ ] T041 [P] Unit tests for method registry (registration, listing, availability) in tests/unit/test_methods.py
- [ ] T042 Integration test: end-to-end DICOM directory → GLB file with validation (valid glTF, correct mesh stats) in tests/integration/test_pipeline.py
- [ ] T043 Integration test: CLI argument parsing, --list-methods output, error messages for missing deps in tests/integration/test_cli.py
- [ ] T044 Implement error handling for edge cases: no DICOM found (exit 4), missing AI deps (exit 3), mixed series warning, inconsistent slice spacing warning, memory errors in src/med2glb/cli.py
- [ ] T045 Add --list-series command: scan directory, display series UIDs with metadata (modality, description, slice count) in src/med2glb/cli.py
- [ ] T046 Code cleanup: add type hints to all public functions, add docstrings to public API, ensure consistent naming

**Checkpoint**: All tests pass. Edge cases handled. CLI polished.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1+US4 (Phase 3)**: Depends on Foundational — core functionality, MVP
- **US2 (Phase 4)**: Depends on Foundational — can run in parallel with Phase 3
- **US3 (Phase 5)**: Depends on Foundational — can run in parallel with Phase 3/4
- **US5 (Phase 6)**: Depends on Phase 3+4+5 completion (docs reference all features)
- **MedSAM2 (Phase 7)**: Depends on Phase 3 (extends animated pipeline)
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **US1+US4 (P1)**: Can start after Foundational — no dependencies on other stories
- **US2 (P2)**: Can start after Foundational — independent from US1 but benefits from shared GLB builder
- **US3 (P3)**: Can start after Foundational — fully independent
- **US5 (P2)**: Depends on US1+US2+US3 being complete (README documents all features)

### Within Each User Story

- Core data flow tasks before integration tasks
- Method implementations can be parallelized [P]
- CLI wiring tasks depend on underlying method implementations
- End-to-end integration task is always last in each phase

### Parallel Opportunities

- T003, T004 can run in parallel with T001/T002
- T005, T006, T010, T011, T014 can all run in parallel (different files)
- T016, T017 can run in parallel (different method files)
- T021 can run in parallel with T018/T019/T020
- T037-T041 (test files) can all run in parallel
- US2, US3 can run in parallel with US1 after Foundational completes

---

## Parallel Example: Phase 3 (US1+US4)

```bash
# Launch parallel method implementations:
Task T016: "Implement marching-cubes method in src/med2glb/methods/marching_cubes.py"
Task T017: "Implement classical method in src/med2glb/methods/classical.py"

# Launch parallel after T016/T017 complete:
Task T021: "Implement multi-format exporter in src/med2glb/io/exporters.py"
# While also working on:
Task T018: "Implement temporal mesh smoothing in src/med2glb/mesh/temporal.py"
```

---

## Implementation Strategy

### MVP First (Phase 1 + 2 + 3)

1. Complete Phase 1: Setup (project structure, pyproject.toml)
2. Complete Phase 2: Foundational (DICOM reader, mesh processing, GLB builder, CLI)
3. Complete Phase 3: US1+US4 (3D echo animated GLB + method switching)
4. **STOP and VALIDATE**: Convert real 3D echo data, view in AR
5. This is the MVP — animated beating heart from echo data

### Incremental Delivery

1. Setup + Foundational → Project skeleton
2. US1+US4 → Animated echo GLB with method switching (MVP!)
3. US2 → Cardiac CT segmentation → Deploy/Demo
4. US3 → Single slice support → Deploy/Demo
5. US5 → Documentation → Deploy/Demo
6. MedSAM2 → AI echo segmentation → Deploy/Demo
7. Polish → Tests, edge cases → Release

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- MedSAM2 (Phase 7) is separated from Phase 3 because it requires AI deps and can be added after MVP validation
