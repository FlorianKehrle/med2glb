"""Integration test: CLI argument parsing and commands."""

from __future__ import annotations

from typer.testing import CliRunner

from dicom2glb.cli import app

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
