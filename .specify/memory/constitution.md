# med2glb Constitution

## Core Principles

### I. Pipeline-First, CLI-Driven
All conversion features must be implemented as reusable Python modules under `src/med2glb/` and exposed through the Typer CLI. CLI behavior is a stable user contract: flags, defaults, and output semantics must remain backward compatible unless a documented breaking change is approved.

### II. Deterministic Medical Data Handling
Input parsing and mapping for DICOM and CARTO data must be deterministic and auditable. Transformations must avoid silent data loss, preserve important metadata where practical, and fail with actionable error messages when required input data is missing or malformed.

### III. Test Coverage Is Non-Negotiable
Every behavior change requires tests. Unit tests are required for algorithmic logic (mapping, meshing, animation, compression), and integration tests are required for end-to-end pipeline behavior. Bug fixes must include a regression test that fails before the fix and passes after it.

### IV. Output Quality and Performance Budgets
Generated GLB output must prioritize correctness first, then size and runtime performance. Changes that significantly increase file size, processing time, or memory use must include justification and, when possible, benchmark evidence on representative data.

### V. Simplicity, Typing, and Documentation
Public APIs must use type hints and clear docstrings. Prefer simple, explicit implementations over clever abstractions. User-visible changes (CLI options, workflows, outputs) must be reflected in `README.md` in the same change set.

## Technical Standards

- Python 3.10+ is the baseline runtime.
- Core stack: Typer/Rich for CLI, NumPy/SciPy/scikit-image for volume processing, trimesh/pyvista for mesh operations, pygltflib for GLB generation.
- Optional AI segmentation dependencies must remain optional and must not break non-AI workflows.
- New dependencies require clear rationale and must not duplicate existing capabilities.
- Logging and error messages must help users diagnose data and pipeline failures quickly.

## Development Workflow and Quality Gates

1. Define scope in specs before major features or architectural changes.
2. Implement in small, reviewable increments with tests.
3. Run relevant test suites (`pytest` and targeted integration tests) before merge.
4. Update user documentation for any CLI or behavior change.
5. Reject changes that weaken determinism, reduce test coverage, or introduce undocumented breaking behavior.

## Governance
This constitution supersedes ad hoc local practices. In case of conflict, this file and accepted feature specs are authoritative.

- Amendments require a documented rationale and impact summary in the associated planning artifact.
- Reviews must explicitly verify constitution compliance, especially testing, deterministic behavior, and README alignment.
- Any temporary exception must include an owner, a reason, and a follow-up task to restore compliance.
- `CLAUDE.md` provides operational implementation guidance and must stay aligned with this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-03-11 | **Last Amended**: 2026-03-11
