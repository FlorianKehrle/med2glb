# Feature Specification: xatlas Progress Timer

**Feature Branch**: `003-xatlas-timer`  
**Created**: 2025-07-16  
**Status**: Clarified  
**Input**: User description: "Add live progress timer during xatlas UV unwrap showing elapsed time and ETA with human-friendly duration format"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Live Progress During UV Unwrap (Priority: P1)

A clinician or researcher converts a CARTO 3 electrophysiology map to GLB format. The xatlas UV unwrap step can take several minutes on high-resolution meshes (100K+ faces after Loop subdivision). Today the user sees a static "UV unwrapping… (estimated ~3 min)" message with no indication whether the process is still running or has frozen. The user wants a continuously updating timer that shows elapsed time and a rough ETA so they can gauge when the operation will complete.

**Why this priority**: Without live feedback, users may kill the process thinking it has hung — wasting compute and forcing a restart.

**Independent Test**: Run a CARTO conversion with `--subdivide 2` on a large mesh and confirm the terminal shows a live-updating spinner with elapsed time and percentage that refreshes every second.

**Acceptance Scenarios**:

1. **Given** a CARTO mesh with 100K+ faces, **When** xatlas UV unwrap starts, **Then** the terminal shows a spinner that updates every second with elapsed time and estimated percentage.
2. **Given** an xatlas operation completing in under 5 seconds, **When** the unwrap finishes, **Then** the spinner stops and a summary line shows actual elapsed time.
3. **Given** a mesh estimated to take ~3 minutes, **When** the user watches the progress, **Then** durations are displayed in human-friendly format (e.g., "1m 23s / ~3 min (46%)").

---

### User Story 2 — Human-Friendly Duration Format (Priority: P2)

Elapsed and estimated times should be displayed using s/m/h units rather than raw seconds. Reading "2m 45s" is more natural than "165s", especially for operations taking several minutes.

**Why this priority**: Improves readability of all time-related console output, not just the xatlas step.

**Independent Test**: Convert a mesh where xatlas takes over 60 seconds and verify the displayed time uses "Xm Ys" format, not raw seconds.

**Acceptance Scenarios**:

1. **Given** an elapsed time under 60 seconds, **When** displayed, **Then** the format is "{N}s" (e.g., "45s").
2. **Given** an elapsed time between 1 and 60 minutes, **When** displayed, **Then** the format is "{M}m {S}s" (e.g., "2m 45s").
3. **Given** an elapsed time over 60 minutes, **When** displayed, **Then** the format is "{H}h {M}m {S}s" (e.g., "1h 5m 30s").

---

### Edge Cases

- What happens when the xatlas ETA is zero (very small mesh)? The timer should still show elapsed time without a percentage.
- What happens when xatlas crashes inside the background thread? The exception must be re-raised on the main thread so the user sees a clear error.
- What happens when the actual time exceeds the ETA? The percentage should cap at 99% to avoid confusing ">100%" displays.
- What happens when output is piped or redirected (non-TTY)? The in-place spinner is skipped entirely; a single static "UV unwrapping…" status line is printed instead.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display a live-updating spinner with elapsed time during the xatlas UV unwrap operation.
- **FR-002**: System MUST show an estimated percentage based on the power-law time model when the ETA is non-zero.
- **FR-003**: The live display MUST update at least once per second.
- **FR-004**: System MUST use human-friendly duration format (s, m, h) for all elapsed and estimated times.
- **FR-005**: The xatlas operation MUST run in a background thread so the main thread can update the display.
- **FR-006**: If xatlas raises an exception in the background thread, the system MUST re-raise it on the main thread.
- **FR-007**: The progress display MUST use an in-place terminal update (no repeated console lines).
- **FR-008**: The percentage display MUST be capped at 99% to avoid showing >100% when the estimate is inaccurate.
- **FR-009**: In non-TTY environments (piped or redirected output), the system MUST skip the in-place spinner and print a single static status line instead.

### Key Entities

- **xatlas Timer Wrapper**: Runs xatlas in a thread, calls a tick callback every N seconds with `(elapsed, eta)`.
- **Duration Formatter**: Converts seconds to human-readable "Xs", "Xm Ys", or "Xh Xm Xs" strings.
- **ETA Estimator**: Power-law model `t = k × n_faces^exp` calibrated from real CARTO mesh data.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users see a live-updating elapsed time during xatlas UV unwrap that refreshes at least once per second.
- **SC-002**: All time durations displayed in the console use the human-friendly s/m/h format, not raw seconds.
- **SC-003**: The background thread does not block the main thread's display updates during the xatlas operation.
- **SC-004**: When xatlas completes, the final summary line shows the actual elapsed time in human-friendly format.

## Assumptions

- The xatlas C binding (`atlas.generate()`) provides no progress callback, so threading is the only mechanism to show live updates.
- The power-law ETA model has ~±50% accuracy. This is acceptable — the primary value is showing that the process is still alive, not providing a precise countdown.
- The timer applies to the CARTO pipeline only; the DICOM pipeline's xatlas calls are typically fast enough (<5s) that a timer is unnecessary. No face-count threshold is used — CARTO always gets the timer, DICOM never does.
