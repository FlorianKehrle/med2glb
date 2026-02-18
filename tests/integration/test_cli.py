"""Integration test: CLI argument parsing and commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from med2glb.cli import _make_output_path, _parse_selection, app
from med2glb.core.types import SeriesInfo

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_list_methods():
    result = runner.invoke(app, ["--list-methods"])
    assert result.exit_code == 0
    assert "marching-cubes" in result.output
    assert "classical" in result.output


def test_missing_input():
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_nonexistent_input():
    result = runner.invoke(app, ["/nonexistent/path"])
    assert result.exit_code != 0


def test_full_pipeline(dicom_directory, tmp_path):
    output = tmp_path / "output.glb"
    result = runner.invoke(
        app,
        [
            str(dicom_directory),
            "-o", str(output),
            "-m", "marching-cubes",
            "--threshold", "250",
            "--smoothing", "3",
            "--faces", "5000",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert output.exists()


# --- _parse_selection tests ---


def _make_series_info(uid: str, desc: str = "", data_type: str = "3D volume") -> SeriesInfo:
    return SeriesInfo(
        series_uid=uid,
        modality="CT",
        description=desc,
        file_count=10,
        data_type=data_type,
        detail="10 slices",
        dimensions="32x32x10",
        recommended_method="marching-cubes",
        recommended_output="3D mesh",
    )


def test_parse_selection_single():
    series_list = [_make_series_info("A"), _make_series_info("B"), _make_series_info("C")]
    result = _parse_selection("1", series_list)
    assert len(result) == 1
    assert result[0].series_uid == "A"


def test_parse_selection_multiple():
    series_list = [_make_series_info("A"), _make_series_info("B"), _make_series_info("C")]
    result = _parse_selection("1,3", series_list)
    assert len(result) == 2
    assert result[0].series_uid == "A"
    assert result[1].series_uid == "C"


def test_parse_selection_all():
    series_list = [_make_series_info("A"), _make_series_info("B"), _make_series_info("C")]
    result = _parse_selection("all", series_list)
    assert len(result) == 3


def test_make_output_path():
    base = Path("output.glb")
    # Single series — unchanged
    assert _make_output_path(base, 0, 1) == base
    # Multiple series — indexed
    assert _make_output_path(base, 0, 3) == Path("output_1.glb")
    assert _make_output_path(base, 1, 3) == Path("output_2.glb")
    assert _make_output_path(base, 2, 3) == Path("output_3.glb")


def test_multi_series_non_tty_auto_selects(dicom_multi_series_directory, tmp_path):
    """CliRunner (non-tty) auto-selects best series without prompting."""
    output = tmp_path / "output.glb"
    result = runner.invoke(
        app,
        [
            str(dicom_multi_series_directory),
            "-o", str(output),
            "-m", "marching-cubes",
            "--threshold", "250",
            "--smoothing", "3",
            "--faces", "5000",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert output.exists()


def test_list_series_enriched(dicom_multi_series_with_multiframe_directory):
    """--list-series shows enriched table with data type and recommendations."""
    result = runner.invoke(
        app,
        [
            str(dicom_multi_series_with_multiframe_directory),
            "--list-series",
        ],
    )
    assert result.exit_code == 0
    assert "Data Type" in result.output
    assert "Recommended" in result.output
