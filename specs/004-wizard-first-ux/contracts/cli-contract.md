# CLI Contract: Wizard-First UX

**Branch**: `004-wizard-first-ux` | **Date**: 2026-03-11

## Visible Flags (shown in `--help`)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `INPUT_PATH` | Path (arg) | required | Path to DICOM/CARTO data or GLB file |
| `-o, --output` | Path | auto | Output file path |
| `--batch` | bool | false | Batch mode: find all CARTO exports and convert |
| `--compress` | bool | false | Compress an existing GLB file |
| `--max-size` | int | 25 | Target size in MB for --compress |
| `--strategy` | str | "ktx2" | Compression strategy: ktx2, draco, downscale, jpeg |
| `--list-methods` | bool | false | List available conversion methods and exit |
| `--list-series` | bool | false | List DICOM series found in input and exit |
| `-v, --verbose` | bool | false | Show detailed processing information |
| `--version` | bool | false | Show version and exit |

## Hidden Flags (accepted but not in `--help`)

| Flag | Type | Default | Used by Equiv Command | Notes |
|------|------|---------|----------------------|-------|
| `-m, --method` | str | "classical" | ✅ DICOM | Conversion method |
| `--animate` | bool | false | ✅ Both | Enable animation |
| `--no-animate` | bool | false | ✅ Both | Force static output |
| `--threshold` | float | None | ✅ DICOM | Isosurface threshold |
| `--smoothing` | int | 15 | ✅ DICOM | Taubin smoothing iterations |
| `--faces` | int | 80000 | ✅ DICOM | Target face count |
| `--alpha` | float | 1.0 | ✅ DICOM | Global transparency |
| `--multi-threshold` | str | None | ✅ DICOM | Multi-threshold config |
| `--series` | str | None | ✅ DICOM | Series UID filter |
| `--coloring` | str | "all" | ✅ CARTO | Coloring scheme |
| `--subdivide` | int | 2 | ✅ CARTO | Mesh subdivision level |
| `--vectors` | bool | false | ✅ CARTO | LAT streamline arrows |
| `--gallery` | bool | false | ❌ | Gallery mode |
| `--columns` | int | 6 | ❌ | Gallery column count |
| `-f, --format` | str | "glb" | ❌ | Output format |

## Behavior Matrix

| TTY | Hidden Flags | Behavior |
|-----|-------------|----------|
| Yes | None | Wizard launches, prints equivalent command after conversion |
| Yes | Any provided | Hint printed ("💡 use wizard for interactive mode"), wizard skipped, flags used directly |
| No  | None | Sensible defaults applied, conversion runs silently |
| No  | Any provided | Flags used directly, no wizard, no hint |

## Equivalent Command Output

**Console (Rich formatted)**:
```
💡 Equivalent command:
   med2glb "C:\data\CARTO" --coloring lat --subdivide 2 --animate --vectors -o "C:\output"
```

**Log file (plain text)**:
```
  Equivalent command:
    med2glb "C:\data\CARTO" --coloring lat --subdivide 2 --animate --vectors -o "C:\output"
```

## Backward Compatibility

- All existing flags continue to work identically.
- Scripts using `--method`, `--coloring`, etc. will see a hint but function correctly.
- The only visible change is `--help` output showing fewer flags.
- No environment variable or config file mechanism needed.
