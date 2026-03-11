# Tasks: Wizard-First UX

**Input**: Design documents from `/specs/004-wizard-first-ux/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Extract shared utilities needed by multiple user stories

- [ ] T001 Move `_fmt_duration()` from `src/med2glb/glb/carto_builder.py` to a new shared module `src/med2glb/_utils.py` and update all existing imports (carto_builder.py, vertex_color_bake.py) to use the shared version
- [ ] T002 Run existing test suite (`pytest tests/unit/ -x -q`) to verify the refactor introduced no regressions

**Checkpoint**: Shared utility available — user story implementation can begin

---

## Phase 2: User Story 1 — Simplified CLI (Priority: P1) 🎯 MVP

**Goal**: Hide 15 pipeline-specific flags from `--help`, keep them functional, print hint when used directly

**Independent Test**: Run `med2glb --help` and verify ≤10 flags shown. Run `med2glb ./test_data/ --method classical` and verify hint is printed + conversion proceeds.

### Implementation for User Story 1

- [ ] T003 [US1] Add `hidden=True` to all 15 pipeline-specific `typer.Option()` declarations in `src/med2glb/cli.py` per the cli-contract.md hidden flags table (--method, --animate, --no-animate, --threshold, --smoothing, --faces, --alpha, --multi-threshold, --series, --coloring, --subdivide, --vectors, --gallery, --columns, --format)
- [ ] T004 [US1] Add hidden-flag detection logic in `src/med2glb/cli.py`: after option parsing, check if any hidden param was explicitly provided using `_was_option_provided(ctx, p)`. If so, print a Rich hint (`💡 Tip: run without flags to use the interactive wizard`) and skip the wizard per the behavior matrix in cli-contract.md
- [ ] T005 [US1] Update `_has_pipeline_flags()` in `src/med2glb/cli.py` to include all 15 hidden params in the check list (currently only checks 9)
- [ ] T006 [US1] Add unit tests in `tests/unit/test_cli_wizard.py` to verify: (a) hidden flags are accepted and their values flow through, (b) hint is printed when hidden flags are provided on TTY, (c) no hint is printed on non-TTY
- [ ] T007 [US1] Run test suite (`pytest tests/unit/ -x -q`) to verify no regressions from flag visibility changes

**Checkpoint**: `--help` shows ≤10 flags, hidden flags work with hint. User Story 1 independently testable.

---

## Phase 3: User Story 2 — Equivalent Command (Priority: P1)

**Goal**: After wizard-guided conversion, display and log a runnable CLI command that reproduces the same result

**Independent Test**: Run a CARTO conversion through the wizard, copy the printed equivalent command, run it non-interactively, verify identical output files.

### Implementation for User Story 2

- [ ] T008 [P] [US2] Create `build_carto_equiv_command()` function in `src/med2glb/cli_wizard.py` that takes a `CartoConfig` and output path, returns a platform-aware quoted command string using `os.name` detection and `shlex.quote()` on Unix per research.md R-003. Handle batch mode (single `--batch` command with shared settings).
- [ ] T009 [P] [US2] Create `build_dicom_equiv_command()` function in `src/med2glb/cli_wizard.py` that takes a `DicomConfig` and output path, returns a platform-aware quoted command string with --method, --smoothing, --faces, --series, --animate flags as applicable
- [ ] T010 [US2] Integrate equivalent command display in `src/med2glb/_pipeline_carto.py`: after wizard-guided CARTO conversion completes, call `build_carto_equiv_command()` and print the result to console using Rich formatting (`💡 Equivalent command:`)
- [ ] T011 [US2] Integrate equivalent command display in `src/med2glb/_pipeline_dicom.py`: after wizard-guided DICOM conversion completes, call `build_dicom_equiv_command()` and print the result to console using Rich formatting
- [ ] T012 [P] [US2] Add `equivalent_command: str | None = None` parameter to `append_carto_entry()` in `src/med2glb/io/conversion_log.py` and format it as an indented block in the log output per research.md R-006
- [ ] T013 [P] [US2] Add `equivalent_command: str | None = None` parameter to `append_dicom_entry()` in `src/med2glb/io/conversion_log.py` and format it as an indented block in the log output
- [ ] T014 [US2] Update all call sites of `append_carto_entry()` in `src/med2glb/_pipeline_carto.py` to pass the `equivalent_command` string
- [ ] T015 [US2] Update all call sites of `append_dicom_entry()` in `src/med2glb/_pipeline_dicom.py` to pass the `equivalent_command` string
- [ ] T016 [US2] Ensure all durations in log entries use `fmt_duration()` from `src/med2glb/_utils.py` (FR-015) — update `conversion_log.py` to import and use it for `elapsed_seconds` formatting
- [ ] T017 [US2] Add unit tests in `tests/unit/test_cli_wizard.py` for `build_carto_equiv_command()` and `build_dicom_equiv_command()`: verify correct flags, platform quoting, batch mode, and path escaping
- [ ] T018 [US2] Add unit tests in a new `tests/unit/test_conversion_log.py` to verify equivalent command appears in log output for both CARTO and DICOM entries
- [ ] T019 [US2] Run test suite (`pytest tests/unit/ -x -q`) to verify no regressions

**Checkpoint**: Wizard prints equivalent command, log file includes it. User Story 2 independently testable.

---

## Phase 4: User Story 3 — Enhanced DICOM Wizard Summary (Priority: P2)

**Goal**: Enrich the DICOM wizard summary table with volume dimensions, pixel spacing, file count, and estimated processing time

**Independent Test**: Run the wizard on a 3D DICOM series and verify the summary table includes at least 8 columns including Dimensions, Spacing, Files, and Est Time.

### Implementation for User Story 3

- [ ] T020 [P] [US3] Add `spacing: str | None = None` and `est_time: str | None = None` fields to the `SeriesInfo` dataclass in `src/med2glb/core/types.py`
- [ ] T021 [US3] Extract pixel spacing (`PixelSpacing`) and slice thickness (`SliceThickness`) from DICOM headers during series analysis in `src/med2glb/io/dicom_reader.py` and populate the new `spacing` field on `SeriesInfo` (format: "X × Y × Z mm"). Display "—" when metadata is unavailable.
- [ ] T022 [US3] Add DICOM processing time estimation heuristic in `src/med2glb/io/dicom_reader.py` or `src/med2glb/cli_wizard.py`: estimate based on method + volume size (slice count × resolution), populate `est_time` field on `SeriesInfo` using `fmt_duration()` from `_utils.py`
- [ ] T023 [US3] Enhance the DICOM wizard summary table in `src/med2glb/cli_wizard.py` `run_dicom_wizard()`: add 4 new Rich Table columns — "Dimensions" (from `info.dimensions`), "Spacing" (from `info.spacing` or "—"), "Files" (from `info.file_count`), "Est Time" (from `info.est_time` or "—")
- [ ] T024 [US3] Add unit tests in `tests/unit/test_cli_wizard.py` for the enhanced DICOM table: verify all new columns are present, verify "—" display for missing metadata, verify spacing formatting
- [ ] T025 [US3] Run test suite (`pytest tests/unit/ -x -q`) to verify no regressions

**Checkpoint**: DICOM wizard shows ≥8 columns of metadata. User Story 3 independently testable.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, final cleanup

- [ ] T026 [P] Update README.md CLI section to reflect the wizard-first UX: document the 10 visible flags, explain the wizard flow, mention equivalent command feature, update examples
- [ ] T027 [P] Update `specs/med2glb/contracts/cli-contract.md` (the main spec CLI contract) to reflect hidden flags and equivalent command
- [ ] T028 Run full test suite (`pytest tests/ -x -q`) and verify all tests pass
- [ ] T029 Run quickstart.md validation: execute the steps from `specs/004-wizard-first-ux/quickstart.md` and verify each works

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **US1 (Phase 2)**: Depends on Setup — core CLI change that enables wizard-first flow
- **US2 (Phase 3)**: Depends on Setup — uses shared `fmt_duration()`. Can run in parallel with US1.
- **US3 (Phase 4)**: Depends on Setup — uses shared `fmt_duration()`. Can run in parallel with US1 and US2.
- **Polish (Phase 5)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Setup → US1 — No dependencies on US2 or US3
- **User Story 2 (P1)**: Setup → US2 — No dependencies on US1 or US3 (equiv command works regardless of flag visibility)
- **User Story 3 (P2)**: Setup → US3 — No dependencies on US1 or US2

### Within Each User Story

- Implementation tasks before integration tasks
- Builder functions (T008, T009) before pipeline integration (T010-T015)
- All code before tests
- Tests before checkpoint validation

### Parallel Opportunities

- T008 + T009 can run in parallel (different functions, same file but no overlap)
- T012 + T013 can run in parallel (different functions in conversion_log.py)
- T020 can run in parallel with T008/T009 (different files entirely)
- US1, US2, US3 can all proceed in parallel after Setup completes
- T026 + T027 can run in parallel (different documentation files)

---

## Parallel Example: User Story 2

```
# Launch builder functions in parallel:
Task T008: "Create build_carto_equiv_command() in src/med2glb/cli_wizard.py"
Task T009: "Create build_dicom_equiv_command() in src/med2glb/cli_wizard.py"

# Launch log format changes in parallel:
Task T012: "Add equivalent_command param to append_carto_entry() in conversion_log.py"
Task T013: "Add equivalent_command param to append_dicom_entry() in conversion_log.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: User Story 1 (T003-T007)
3. **STOP and VALIDATE**: Run `med2glb --help` — should show ≤10 flags
4. Hidden flags should still work with hint message

### Incremental Delivery

1. Setup → shared utility extracted
2. Add US1 → CLI simplified, wizard is primary → Test independently
3. Add US2 → Equivalent command in console + log → Test independently
4. Add US3 → DICOM wizard enriched → Test independently
5. Polish → README + specs updated → Final validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- All 3 user stories can proceed in parallel after Phase 1 Setup
- Commit after each phase or logical task group
- Stop at any checkpoint to validate the story independently
- Total: 29 tasks (2 setup, 5 US1, 12 US2, 6 US3, 4 polish)
