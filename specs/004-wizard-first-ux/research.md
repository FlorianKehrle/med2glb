# Research: Wizard-First UX

**Branch**: `004-wizard-first-ux` | **Date**: 2026-03-11

## R-001: Hiding Typer Flags from `--help`

**Decision**: Use Typer's `hidden=True` parameter on `typer.Option()`.

**Rationale**: Typer (via Click) natively supports `hidden=True` which removes the option from `--help` output while keeping it fully functional. This is zero-effort, well-tested, and requires no custom help formatting.

**Alternatives considered**:
- Custom help formatter overriding Click's `format_help` — over-engineered, fragile.
- Separate hidden commands — breaks the single-entry-point pattern.

**Implementation**: Change each hidden flag from `typer.Option(...)` to `typer.Option(..., hidden=True)`.

## R-002: Detecting Hidden-Flag Usage and Printing Hints

**Decision**: Reuse the existing `_was_option_provided(ctx, param_name)` function. After option parsing but before wizard launch, check all hidden flags. If any was explicitly provided, print a Rich hint and skip the wizard.

**Rationale**: `_was_option_provided` already uses Click's `ParameterSource.COMMANDLINE` detection. This is the exact mechanism needed — no new code required for detection.

**Implementation**:
```python
hidden_params = ["method", "coloring", "animate", ...]
provided_hidden = [p for p in hidden_params if _was_option_provided(ctx, p)]
if provided_hidden:
    console.print("[dim]💡 Tip: run without flags to use the interactive wizard[/dim]")
    # proceed with provided flag values (no wizard)
```

## R-003: Equivalent Command Construction

**Decision**: Build the equivalent command from `CartoConfig` / `DicomConfig` dataclasses returned by the wizard. Use `shlex.quote()` on Unix, manual double-quoting on Windows.

**Rationale**: The wizard already produces typed config objects with all user choices. Reconstructing the command from these is straightforward — each config field maps to a known CLI flag.

**Alternatives considered**:
- Recording Click's raw argv — fragile, doesn't capture wizard-derived values.
- Serializing config to JSON — doesn't produce a runnable command.

**Implementation**: Add a `build_equivalent_command()` function in `cli.py` (or a new `_command_builder.py`) that takes a config object and input/output paths, returns a command string.

**Platform quoting**: Use `os.name == 'nt'` to switch between:
- Windows: `"C:\path with spaces\file"` (double quotes, backslash paths)
- Unix: `'/path with spaces/file'` (shlex.quote, forward slashes)

## R-004: Wizard → CLI Flag Mapping Gaps

**Decision**: Accept known gaps; equivalent command is "best effort" for parameters that have no exact CLI equivalent.

**Rationale**: Three wizard features have no CLI equivalent:
1. **Mesh selection** (wizard picks meshes 1,3 of 5) — no `--meshes` flag exists.
2. **Vector "only" mode** — CLI `--vectors` is boolean, wizard has "yes/no/only".
3. **Quality presets** (draft/standard/high) — CLI uses raw `--smoothing` + `--faces` values.

For (1), adding `--meshes` is out of scope — the equivalent command will convert all meshes. For (2), adding a `--vectors only` option would be a minor CLI extension. For (3), the equivalent command outputs the raw numeric values.

**Alternatives considered**:
- Adding new hidden flags for every wizard-only feature — scope creep, deferred to future.
- Marking the equivalent command as "approximate" — confusing for users who expect reproducibility.

**Implementation**: The equivalent command will include the closest CLI flags. A comment in the log explains any deviation.

## R-005: DICOM Wizard Table Enhancement

**Decision**: Add 4 new columns to the DICOM wizard table: Dimensions, Spacing, Files, Est Time. Data is already available in `SeriesInfo` (dimensions, file_count) and pydicom headers (PixelSpacing, SliceThickness).

**Rationale**: `SeriesInfo` already stores `file_count` and `dimensions` but the wizard table doesn't display them. Spacing needs extraction from the DICOM headers during `analyze_series()`. Time estimation is new logic.

**Implementation**:
- `file_count`: Already in `SeriesInfo` — just add column.
- `dimensions`: Already in `SeriesInfo` — just add column.
- `spacing`: Extract `PixelSpacing` + `SliceThickness` during series analysis, add to `SeriesInfo`.
- `est_time`: Heuristic based on method + volume size (similar to CARTO's `_estimate_xatlas_time`).

## R-006: Conversion Log Format for Equivalent Command

**Decision**: Add an `equivalent_command` keyword argument to both `append_carto_entry()` and `append_dicom_entry()`. Output it as an indented line in the log block.

**Rationale**: The log is plain text, human-readable. Adding one more indented line is consistent with the existing format.

**Format**:
```
  Equivalent command:
    med2glb "/path/to/input" --method classical --animate -o "/path/to/output.glb"
```

## R-007: Human-Friendly Duration Format

**Decision**: Reuse the existing `_fmt_duration()` from `carto_builder.py`. Move it to a shared location (e.g., `_utils.py` or `io/conversion_log.py`) since it's needed by both CARTO and DICOM log formatting.

**Rationale**: The function already exists and handles s/m/h formatting correctly. Duplication is the only alternative, which violates DRY.

**Implementation**: Move `_fmt_duration()` to a shared module, import from both `carto_builder.py` and `conversion_log.py`.
