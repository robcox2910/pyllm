"""Sibling projects are pinned to release tags, never to a moving branch or folder."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
GITHUB = "https://github.com/robcox2910"
# Packages whose GitHub repo has a different name.
REPOS = {"pyos-learn": "py-os", "pydocdb": "pydoc-db"}
TAG_PATTERN = re.compile(r"v\d+\.\d+\.\d+")


def _sources() -> dict[str, dict[str, str]]:
    """Read the sibling sources table from pyproject."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["uv"]["sources"]


def test_every_sibling_source_is_a_tagged_git_source():
    """Every sibling source is a tagged git source."""
    for name, source in _sources().items():
        assert set(source) == {"git", "tag"}, name
        assert source["git"] == f"{GITHUB}/{REPOS.get(name, name)}", name
        assert TAG_PATTERN.fullmatch(source["tag"]), name
