# Implementation Plan: Wizard-First UX

**Branch**: `004-wizard-first-ux` | **Date**: 2026-03-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-wizard-first-ux/spec.md`

## Summary

Refactor the med2glb CLI to make the interactive wizard the primary user interface. Hide 15 pipeline-specific flags from `--help` (keeping them functional for scripting/equivalent-command reproducibility). Add "Equivalent command" output to console and log file after every wizard-guided conversion. Enhance the DICOM wizard summary table to match CARTO's richness (add dimensions, spacing, file count, estimated time).

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: Typer/Click (CLI framework), Rich (terminal UI), pydicom (DICOM metadata)  
**Storage**: File-based (`med2glb_log.txt`)  
**Testing**: pytest (266 unit tests + integration tests)  
**Target Platform**: Windows, macOS, Linux — all terminal environments  
**Project Type**: Single Python package (src layout)  
**Performance Goals**: Wizard startup <2s, equivalent command generation <1ms  
**Constraints**: Hidden flags must remain backward-compatible; equivalent command must reproduce identical output  
**Scale/Scope**: 4 files modified (cli.py, cli_wizard.py, conversion_log.py, _pipeline_carto.py/_pipeline_dicom.py), ~500 LOC changed

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Pipeline-First, CLI-Driven | ⚠️ ATTENTION | We are changing CLI behavior: hiding flags from `--help`. However, all flags remain functional — no breaking change. The equivalent command ensures scriptability. |
| II. Deterministic Medical Data | ✅ PASS | No data processing changes. Equivalent command ensures reproducible output. |
| III. Test Coverage Non-Negotiable | ✅ PASS | Tests required for: hidden flag detection, equivalent command generation, DICOM metadata extraction, log format. |
| IV. Output Quality and Performance | ✅ PASS | No changes to GLB output quality. Minor time estimation additions. |
| V. Simplicity, Typing, Documentation | ✅ PASS | README must be updated with new CLI UX. Public APIs keep type hints. |

**Principle I justification**: Hiding flags from `--help` is a UX simplification, not a breaking change. All flags remain accepted. The equivalent command mechanism preserves the CLI-driven contract — any wizard-guided conversion can be reproduced via flags. This is the standard pattern (git, docker, kubectl all have hidden flags).

## Project Structure

### Documentation (this feature)

```text
specs/004-wizard-first-ux/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── cli-contract.md  # Updated CLI contract (visible + hidden flags)
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/med2glb/
├── cli.py               # MODIFIED: hide flags, detect hidden-flag usage, print hint
├── cli_wizard.py         # MODIFIED: enhance DICOM table, return equiv command params
├── _pipeline_carto.py    # MODIFIED: print+log equivalent command after conversion
├── _pipeline_dicom.py    # MODIFIED: print+log equivalent command after conversion
└── io/
    └── conversion_log.py # MODIFIED: accept+format equivalent command field

tests/unit/
├── test_cli_wizard.py    # MODIFIED: test DICOM table enrichment, equiv command
└── test_conversion_log.py # NEW: test equiv command in log entries
```

**Structure Decision**: No new modules needed. Changes are surgical modifications to 5 existing source files plus 1 new test file.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | — | — |
