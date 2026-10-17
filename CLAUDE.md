# PyLLM

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
