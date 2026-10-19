"""Tests for cli pebble."""

from pathlib import Path

import pytest

from pyllm.cli import main


def test_pebble_command_generates(capsys: pytest.CaptureFixture[str]):
    """Pebble command generates."""
    code = main(["pebble", "--max-new-tokens", "40", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(out.strip()) > 0


def test_pebble_command_score_reports_rate_when_available(capsys: pytest.CaptureFixture[str]):
    """Pebble command score reports rate when available."""
    import pytest

    pytest.importorskip("pebble")
    code = main(["pebble", "--max-new-tokens", "80", "--seed", "1", "--score"])
    out = capsys.readouterr().out
    assert code == 0
    assert "parse rate" in out.lower()


def test_gen_corpus_writes_a_file(tmp_path: Path):
    """Gen corpus writes a file."""
    dest = tmp_path / "corpus.txt"
    code = main(["gen-corpus", "--num-generated", "10", "--out", str(dest), "--seed", "0"])
    assert code == 0
    assert dest.exists()
    assert dest.read_text(encoding="utf-8").count("let ") >= 10


def test_constrained_pebble_always_parses(capsys: pytest.CaptureFixture[str]):
    """Constrained pebble always parses."""
    import pytest

    pytest.importorskip("pebble")
    code = main(["pebble", "--constrained", "--max-new-tokens", "150", "--seed", "0", "--score"])
    out = capsys.readouterr().out
    assert code == 0
    assert "parse rate: 100.0%" in out


def test_constrained_pebble_rejects_a_rule_breaking_prompt(capsys: pytest.CaptureFixture[str]):
    """Constrained pebble rejects a rule breaking prompt."""
    code = main(["pebble", "--constrained", "--prompt", "}", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "referee" in out.lower()


def test_gen_corpus_missing_harvest_folder_fails_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Gen corpus refuses a --harvest folder that does not exist (no silent skip)."""
    dest = tmp_path / "corpus.txt"
    missing = tmp_path / "no-such-docs"
    code = main(
        ["gen-corpus", "--num-generated", "2", "--harvest", str(missing), "--out", str(dest)]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "no-such-docs" in out
    assert not dest.exists()
