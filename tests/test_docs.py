from pathlib import Path


def test_autograd_concept_doc_exists_and_covers_key_ideas():
    doc = Path("docs/concepts/autograd.md")
    assert doc.exists(), "RULE #1: every plan ships a kid-friendly concept doc"
    text = doc.read_text().lower()
    # The big ideas a child should walk away understanding.
    for idea in ["breadcrumb", "gradient", "backward", "analogy"]:
        assert idea in text, f"concept doc should explain '{idea}'"


def test_plan2_concept_docs_exist_and_cover_key_ideas():
    from pathlib import Path

    checks = {
        "docs/concepts/tokens.md": ["token", "chunk", "analogy"],
        "docs/concepts/embeddings.md": ["embedding", "meaning", "analogy"],
        "docs/concepts/attention.md": ["attention", "look back", "analogy"],
    }
    for path, ideas in checks.items():
        doc = Path(path)
        assert doc.exists(), f"RULE #1: missing concept doc {path}"
        text = doc.read_text().lower()
        for idea in ideas:
            assert idea in text, f"{path} should explain '{idea}'"


def test_plan3_concept_docs_exist_and_cover_key_ideas():
    from pathlib import Path

    checks = {
        "docs/concepts/how-a-model-learns.md": ["loss", "gradient", "analogy"],
        "docs/concepts/sampling.md": ["temperature", "top-k", "analogy"],
        "docs/concepts/transformer.md": ["attention", "block", "analogy"],
    }
    for path, ideas in checks.items():
        doc = Path(path)
        assert doc.exists(), f"RULE #1: missing concept doc {path}"
        text = doc.read_text().lower()
        for idea in ideas:
            assert idea in text, f"{path} should explain '{idea}'"


def test_plan4_concept_doc_exists_and_covers_key_ideas():
    from pathlib import Path

    doc = Path("docs/concepts/grow-your-own-data.md")
    assert doc.exists(), "RULE #1: missing concept doc grow-your-own-data.md"
    text = doc.read_text().lower()
    for idea in ["harvest", "generate", "grade", "parser", "analogy"]:
        assert idea in text, f"grow-your-own-data.md should explain '{idea}'"
