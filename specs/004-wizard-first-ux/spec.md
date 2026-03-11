# Feature Specification: Wizard-First UX

**Feature Branch**: `004-wizard-first-ux`  
**Created**: 2025-07-16  
**Status**: Clarified  
**Input**: User description: "Wizard-first UX refactor with CLI simplification, equivalent command logging, and DICOM wizard enhancement"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Simplified CLI With Wizard as Primary Interface (Priority: P1)

A new user runs `med2glb ./my-data/` for the first time. Instead of being overwhelmed by 25+ CLI flags, they are greeted by an interactive wizard that detects the data type, presents clear options, and guides them through the conversion. Advanced users who need scripted/CI pipelines can still pass a minimal set of non-interactive flags (`-o`, `--batch`, `--compress`).

**Why this priority**: The wizard already exists and is the preferred user path. Removing redundant flags eliminates confusion, reduces maintenance burden, and makes the tool more approachable.

**Independent Test**: Run `med2glb --help` and verify only essential flags are listed. Then run `med2glb ./test-data/` interactively and confirm the wizard launches without needing any flags.

**Acceptance Scenarios**:

1. **Given** a terminal with TTY, **When** the user runs `med2glb ./input/` with no flags, **Then** the interactive wizard launches and guides the user through all conversion choices.
2. **Given** the CLI help output, **When** the user runs `med2glb --help`, **Then** only the essential flags are shown: `input_path`, `-o`, `--batch`, `--compress`, `--max-size`, `--strategy`, `--list-methods`, `--list-series`, `-v`, `--version`.
3. **Given** a non-TTY environment (piped or CI), **When** the user runs `med2glb ./input/ -o out.glb`, **Then** the tool runs with sensible defaults (auto-detect data type, classical method for DICOM, all colorings for CARTO) without prompting.
4. **Given** a user who used the wizard, **When** the conversion completes, **Then** the console displays an "Equivalent command" line showing the exact flags that would reproduce the same result.

---

### User Story 2 — Equivalent Command for Reproducibility (Priority: P1)

After completing a wizard-guided conversion, the user wants to reproduce the exact same conversion later (in CI, in a script, or by sharing with a colleague). The wizard should display and log the equivalent CLI command so the result can be replicated without re-running the wizard.

**Why this priority**: Reproducibility is essential for clinical and research workflows. Without it, users cannot automate or share their exact configurations.

**Independent Test**: Run a CARTO conversion through the wizard, then verify the printed equivalent command. Run that command in a non-TTY environment and confirm the output is identical.

**Acceptance Scenarios**:

1. **Given** a completed CARTO wizard conversion, **When** the result is displayed, **Then** an "Equivalent command" line appears showing the full `med2glb` command with all chosen parameters.
2. **Given** a completed DICOM wizard conversion, **When** the result is displayed, **Then** an "Equivalent command" line appears with the chosen method, series, and output path.
3. **Given** a conversion of any type, **When** the result is logged to `med2glb_log.txt`, **Then** the equivalent command is included in the log entry.
4. **Given** the equivalent command, **When** the user runs it, **Then** the output is identical to the wizard-guided result (deterministic parameters only).

---

### User Story 3 — Enhanced DICOM Wizard Summary (Priority: P2)

A user loads a DICOM dataset and the wizard presents a summary table. Currently the DICOM wizard shows basic metadata (Modality, Description, Type, Recommended Method). The user wants to see richer data — volume dimensions, pixel spacing, file count, and estimated processing time — to make informed decisions before starting the conversion.

**Why this priority**: The CARTO wizard already shows this level of detail. Bringing the DICOM wizard up to the same standard creates a consistent, professional experience.

**Independent Test**: Run the wizard on a 3D DICOM series and verify the summary table includes volume dimensions, spacing, file count, and estimated processing time.

**Acceptance Scenarios**:

1. **Given** a 3D DICOM series, **When** the wizard shows the summary, **Then** the table includes "Dimensions" showing the volume shape (e.g., "512 × 512 × 120").
2. **Given** a 3D DICOM series with spacing info, **When** the wizard shows the summary, **Then** the table includes "Spacing" showing pixel spacing and slice thickness (e.g., "0.5 × 0.5 × 1.0 mm").
3. **Given** a DICOM directory with multiple files, **When** the wizard shows the summary, **Then** the table includes "Files" showing the number of DICOM files per series.
4. **Given** any DICOM series, **When** the wizard shows the summary, **Then** the table includes "Est Time" showing a rough processing time estimate.

---

### Edge Cases

- What happens when a user passes a hidden flag (e.g., `--method`)? The CLI accepts it, prints a hint suggesting the wizard for interactive use, then proceeds with the provided value. The flag works but is not shown in `--help`.
- What happens when running non-interactively without specifying critical parameters? The system should use sensible defaults and log which defaults were applied.
- What happens when the DICOM metadata is incomplete (no spacing info, no series description)? The wizard should display "—" for unknown fields rather than crashing.
- What happens when the equivalent command contains special characters or paths with spaces? The command uses platform-aware quoting matching the current OS shell conventions.
- What happens in batch mode? The equivalent command is a single `--batch` command with shared settings, not per-dataset commands.

## Requirements *(mandatory)*

### Functional Requirements

**CLI Simplification:**

- **FR-001**: The public CLI MUST only expose essential non-interactive flags: `input_path`, `-o/--output`, `--batch`, `--compress`, `--max-size`, `--strategy`, `--list-methods`, `--list-series`, `-v/--verbose`, `--version`.
- **FR-002**: Removed flags (`--method`, `--animate`, `--threshold`, `--smoothing`, `--faces`, `--alpha`, `--multi-threshold`, `--series`, `--coloring`, `--subdivide`, `--vectors`, `--gallery`, `--columns`, `--no-animate`, `--format`) MUST be hidden from `--help` output but still accepted by the CLI. When a user explicitly provides a hidden flag, the system MUST print a hint suggesting the wizard for interactive use, then proceed with the provided value.
- **FR-003**: In interactive mode (TTY), the wizard MUST always launch when no hidden flags are provided. If hidden flags are provided, the system MUST skip the wizard and use the flag values directly.
- **FR-004**: In non-interactive mode, the system MUST use sensible auto-detected defaults for all removed parameters.

**Equivalent Command:**

- **FR-005**: After a wizard-guided conversion, the system MUST display an "Equivalent command" line in the console output.
- **FR-006**: The equivalent command MUST be written to the `med2glb_log.txt` log file for both CARTO and DICOM conversions.
- **FR-007**: The equivalent command MUST reproduce the same conversion result when run non-interactively.
- **FR-008**: The equivalent command MUST use platform-aware path quoting that matches the current OS shell conventions (e.g., backslash paths with double quotes on Windows, forward slashes on Unix).

**DICOM Wizard Enhancement:**

- **FR-009**: The DICOM wizard summary table MUST include volume dimensions for 3D series (e.g., "512 × 512 × 120").
- **FR-010**: The DICOM wizard summary table MUST include pixel spacing for series that have it (e.g., "0.5 × 0.5 × 1.0 mm").
- **FR-011**: The DICOM wizard summary table MUST include the number of DICOM files per series.
- **FR-012**: The DICOM wizard summary table MUST include an estimated processing time.
- **FR-013**: Missing metadata fields MUST display "—" rather than causing errors.

**Conversion Log:**

- **FR-014**: The `med2glb_log.txt` file MUST include the equivalent command for every conversion (both CARTO and DICOM).
- **FR-015**: All time durations in the log file MUST use human-friendly format (s, m, h).

### Key Entities

- **Wizard Configuration**: The set of parameters chosen by the user during the wizard flow (method, colorings, subdivision, animation, etc.).
- **Equivalent Command**: A reconstructed CLI command string derived from the wizard configuration that would produce the same output.
- **Conversion Log Entry**: A structured record in `med2glb_log.txt` documenting one conversion run, including input/output paths, parameters, timings, and equivalent command.
- **DICOM Series Summary**: Metadata extracted from DICOM files for display in the wizard (modality, dimensions, spacing, file count, processing estimate).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The `--help` output lists 10 or fewer flags (down from 25+), making the CLI immediately comprehensible.
- **SC-002**: Users can complete a full conversion using only the wizard without needing to know any flags.
- **SC-003**: Every wizard-guided conversion produces a reproducible equivalent command in both console and log file.
- **SC-004**: The DICOM wizard summary table shows at least 8 columns of useful metadata (matching or approaching the CARTO wizard's level of detail).
- **SC-005**: Running the equivalent command in a non-interactive environment produces the same output as the wizard-guided run.
- **SC-006**: All time-related output (console and log) uses human-friendly duration format (s, m, h).

## Assumptions

- The wizard-first approach assumes most users run the tool interactively. Power users and CI pipelines remain supported through non-interactive defaults and the equivalent command mechanism.
- Removed flags are kept as **hidden flags** — accepted by the CLI but not shown in `--help`. The equivalent command uses these hidden flags to reproduce wizard-guided results. This is the standard CLI pattern (e.g., git, docker).
- DICOM processing time estimation is inherently rough (varies by method, hardware, and data complexity). A ±50% estimate is acceptable as user guidance.
- The conversion log file (`med2glb_log.txt`) is a plain-text, human-readable file — not a structured format like JSON. This is intentional for quick inspection.
- In batch mode, the equivalent command is a single `--batch` command with shared settings, not per-dataset commands.
