# Quickstart: Wizard-First UX

**Branch**: `004-wizard-first-ux` | **Date**: 2026-03-11

## Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### 1. Hide CLI Flags

In `src/med2glb/cli.py`, add `hidden=True` to each flag being hidden:

```python
# Before
method: str = typer.Option("classical", "-m", "--method", help="...")

# After
method: str = typer.Option("classical", "-m", "--method", help="...", hidden=True)
```

### 2. Test Hidden Flags

```bash
# Verify flag is hidden from help
med2glb --help  # should NOT show --method

# Verify flag still works
med2glb ./test_data/ --method classical  # should work + print hint
```

### 3. Run Tests

```bash
# Unit tests
pytest tests/unit/ -x -q

# Specific test files
pytest tests/unit/test_cli_wizard.py -x -q
pytest tests/unit/test_conversion_log.py -x -q
```

### 4. Key Files to Modify

| File | Change |
|------|--------|
| `src/med2glb/cli.py` | Hide 15 flags, add hint detection, pass equiv command |
| `src/med2glb/cli_wizard.py` | Enhance DICOM table, build equiv command from config |
| `src/med2glb/io/conversion_log.py` | Accept + format `equivalent_command` parameter |
| `src/med2glb/_pipeline_carto.py` | Print equiv command after conversion |
| `src/med2glb/_pipeline_dicom.py` | Print equiv command after conversion |

### 5. Verify End-to-End

```bash
# Interactive wizard — should show enhanced DICOM table + equivalent command
med2glb ./test_data/

# Non-interactive — should work with hidden flags
med2glb ./test_data/ --method classical -o output.glb

# Check log file
cat med2glb_log.txt  # should contain equivalent command
```
