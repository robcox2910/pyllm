# PyLLM

<!-- standard:start -->
## Code Style

- All public functions, methods, classes, and modules **must** have a docstring
- Use imperative mood: "Return the result." not "Returns the result."
- One-line docstrings for simple functions; multi-line with summary + details for complex ones
- Follow the Google docstring convention (D211, D212 selected via ruff)

## Linting

- Ruff enforces a comprehensive rule set -- run `uv run ruff check .` before committing
- Pyright runs in strict mode -- run `uv run pyright src tests` before committing
- All violations must be fixed, not ignored. Exceptions are the repo-owned `ignore` and `per-file-ignores` lists in `pyproject.toml`, which belong to this repo; the series sync never edits them

## Testing

- TDD workflow: Red-Green-Refactor
- Coverage is enforced by pytest-cov; the floor in `pyproject.toml` is repo-owned
- Test classes and methods also require docstrings
- Use named constants instead of magic values (PLR2004)
- Run `uv run pytest` before committing
- Property-based tests (Hypothesis) live in `tests/test_properties.py` and `tests/test_model.py`; the shared profiles are in the root `conftest.py` (owned by `series sync`). Run `uv run pytest --hypothesis-profile=thorough` before a release.

## Commits

- Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`
- All changes go via PR from a feature branch (main is protected); merge when CI is green

## Dependencies

- Package manager: uv
- Run `uv sync --all-extras` after modifying dependencies
- `uv.lock` is committed; CI and pre-commit audit it with pip-audit through `uvx`, so pip-audit is not a project dependency
- Sibling projects, where used, are pinned release tags in `[tool.uv.sources]` (git + tag); never use `path` sources

## Releasing

- A release is a `vX.Y.Z` tag on main after the PR is merged
- Docs-only and tooling-only changes do not get a release tag
- Docs are built with `mkdocs build --strict` and deployed to GitHub Pages by the docs workflow
<!-- standard:end -->

## Project notes

- **Numpy autograd.** There is no torch: `src/pyllm/autograd` is a small reverse-mode
  autodiff engine on numpy arrays (`Tensor`). Layers live in `src/pyllm/nn`, models in
  `src/pyllm/models`, and `pyllm._types` holds the shared `Array` alias. Each op stores a
  backward recipe via `Tensor.set_backward`; check new ops with
  `pyllm.autograd.gradcheck.numerical_grad`.
- **Bundled checkpoints.** `src/pyllm/data/pokemon.npz` and `src/pyllm/data/pebble.npz` are
  trained models committed to the repo (and force-included in the wheel). Regenerate them with
  `uv run python scripts/train_pokemon.py` and `uv run python scripts/train_pebble.py`. Both
  use fixed seeds, but training changes whenever the model code changes, so rerun them and
  commit the new `.npz` files if you touch the models, optimizers or training loop.
- **The Pebble referee.** `src/pyllm/pebble/referee.py` is a hand-written, character-by-character
  checker for the canonical slice of Pebble the corpus uses (4-space indents, single spaces
  around operators). `viable(text)` means "could still become valid" and `complete(text)` means
  "valid right now". It must stay a strict subset of real Pebble so anything it accepts, Pebble's
  own parser accepts too (tests check this against `pebble-lang`). `generate(..., referee=...)`
  uses it to veto illegal characters.
- **Regenerating the Pebble corpus.** `src/pyllm/data/pebble_corpus.txt` is generated, not
  hand-written: `uv run python scripts/build_pebble_corpus.py` grows random valid programs
  (`pebble/generator.py`), mixes in snippets harvested from a sibling `../pebble-lang/docs`
  checkout (kept only if they parse) and canonicalises everything with Pebble's formatter. The
  same recipe is available as `pyllm gen-corpus`. It needs the optional `pebble` extra
  (`uv sync --all-extras`). Retrain `pebble.npz` afterwards.
