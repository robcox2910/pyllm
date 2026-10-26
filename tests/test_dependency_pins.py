"""Sibling projects come from their PyPI releases, never from a branch, a folder or a git tag."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
SIBLING_PREFIX = "fromscratch-"
# A sibling requirement: the package name and a minimum released version.
SIBLING_REQUIREMENT = re.compile(r"fromscratch-[a-z]+>=\d+\.\d+\.\d+")


def _config() -> dict[str, object]:
    """Read pyproject.toml."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _requirements() -> list[str]:
    """Return every requirement, required and optional."""
    project = _config()["project"]
    assert isinstance(project, dict)
    found: list[str] = list(project.get("dependencies", []))  # type: ignore[arg-type]
    for extra in project.get("optional-dependencies", {}).values():  # type: ignore[union-attr]
        found.extend(extra)  # type: ignore[arg-type]
    return found


def test_no_sibling_is_pulled_from_git_or_a_folder():
    """There is no ``[tool.uv.sources]`` table: every sibling comes from PyPI."""
    tool = _config().get("tool", {})
    assert isinstance(tool, dict)
    assert "sources" not in tool.get("uv", {})  # type: ignore[operator]


def test_every_sibling_names_a_released_version():
    """Each sibling requirement is a ``fromscratch-*`` package with a ``>=X.Y.Z`` floor."""
    siblings = [r for r in _requirements() if r.startswith(SIBLING_PREFIX)]
    assert siblings
    for requirement in siblings:
        assert SIBLING_REQUIREMENT.fullmatch(requirement), requirement
