from pathlib import Path


def test_autograd_concept_doc_exists_and_covers_key_ideas():
    doc = Path("docs/concepts/autograd.md")
    assert doc.exists(), "RULE #1: every plan ships a kid-friendly concept doc"
    text = doc.read_text().lower()
    # The big ideas a child should walk away understanding.
    for idea in ["breadcrumb", "gradient", "backward", "analogy"]:
        assert idea in text, f"concept doc should explain '{idea}'"
