from pyllm.cli import main


def test_pebble_command_generates(capsys):
    code = main(["pebble", "--max-new-tokens", "40", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(out.strip()) > 0


def test_pebble_command_score_reports_rate_when_available(capsys):
    import pytest

    pytest.importorskip("pebble")
    code = main(["pebble", "--max-new-tokens", "80", "--seed", "1", "--score"])
    out = capsys.readouterr().out
    assert code == 0
    assert "parse rate" in out.lower()


def test_gen_corpus_writes_a_file(tmp_path):
    dest = tmp_path / "corpus.txt"
    code = main(
        ["gen-corpus", "--num-generated", "10", "--out", str(dest), "--seed", "0"]
    )
    assert code == 0
    assert dest.exists()
    assert dest.read_text(encoding="utf-8").count("let ") >= 10


def test_constrained_pebble_always_parses(capsys):
    import pytest

    pytest.importorskip("pebble")
    code = main(
        ["pebble", "--constrained", "--max-new-tokens", "150", "--seed", "0", "--score"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "parse rate: 100.0%" in out


def test_constrained_pebble_rejects_a_rule_breaking_prompt(capsys):
    code = main(["pebble", "--constrained", "--prompt", "}", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "referee" in out.lower()
