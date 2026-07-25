from pyllm.cli import main


def test_tokenize_prints_ids(capsys):
    code = main(["tokenize", "--text", "abc"])
    out = capsys.readouterr().out
    assert code == 0
    # each character appears next to a number
    assert "a" in out and "b" in out and "c" in out
    assert any(ch.isdigit() for ch in out)


def test_default_command_generates_from_bundled_checkpoint(capsys):
    code = main(["--max-new-tokens", "20", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(out.strip()) > 0


def test_train_writes_a_checkpoint(tmp_path):
    dest = tmp_path / "my.npz"
    code = main(
        [
            "train",
            "--model",
            "bigram",
            "--steps",
            "5",
            "--out",
            str(dest),
            "--seed",
            "0",
        ]
    )
    assert code == 0
    assert dest.exists()


def test_generate_is_reproducible_with_seed(capsys):
    main(["--max-new-tokens", "30", "--seed", "42"])
    first = capsys.readouterr().out
    main(["--max-new-tokens", "30", "--seed", "42"])
    second = capsys.readouterr().out
    assert first == second
