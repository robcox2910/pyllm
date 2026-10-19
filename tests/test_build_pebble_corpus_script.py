"""Tests for the `scripts/build_pebble_corpus.py` build step."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_pebble_corpus.py"


def _load_script() -> ModuleType:
    """Import the build script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("build_pebble_corpus", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_docs_path_is_the_sibling_pebble_checkout():
    """With no option the script mines the sibling ../pebble-lang/docs folder."""
    script = _load_script()
    args = script.parse_args([])
    assert args.pebble_docs == _SCRIPT.parents[2] / "pebble-lang" / "docs"
    assert args.harvest is True


def test_missing_docs_folder_fails_loudly(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A docs folder that does not exist stops the build with a clear message."""
    script = _load_script()
    missing = tmp_path / "no-such-docs"
    out = tmp_path / "corpus.txt"
    with pytest.raises(SystemExit) as stop:
        script.main(["--pebble-docs", str(missing), "--out", str(out)])
    assert stop.value.code != 0
    err = capsys.readouterr().err
    assert "no-such-docs" in err
    assert "--no-harvest" in err
    assert not out.exists()


def test_no_harvest_opts_out_of_the_docs_folder(tmp_path: Path):
    """--no-harvest skips mining on purpose, so a missing folder is fine."""
    script = _load_script()
    args = script.parse_args(["--no-harvest", "--pebble-docs", str(tmp_path / "nope")])
    assert script.harvest_paths(args) == []


def test_existing_docs_folder_is_harvested(tmp_path: Path):
    """An existing docs folder is handed to the corpus builder."""
    script = _load_script()
    args = script.parse_args(["--pebble-docs", str(tmp_path)])
    assert script.harvest_paths(args) == [tmp_path]
