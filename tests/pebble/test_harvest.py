from pyllm.pebble.harvest import harvest_dir
from pyllm.pebble.harvest import harvest_text

_MD = """# Title

Some prose.

```pebble
let x = 1
print(x)
```

More prose.

```python
print("not pebble")
```

```pebble
fn f() { return 1 }
```
"""


def test_harvest_text_extracts_only_pebble_blocks():
    blocks = harvest_text(_MD)
    assert len(blocks) == 2
    assert blocks[0] == "let x = 1\nprint(x)"
    assert blocks[1] == "fn f() { return 1 }"


def test_harvest_text_no_blocks_returns_empty():
    assert harvest_text("just prose, no code") == []


def test_harvest_dir_reads_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("```pebble\nlet a = 1\n```\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("```pebble\nlet b = 2\n```\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("```pebble\nlet c = 3\n```\n", encoding="utf-8")
    blocks = harvest_dir(tmp_path)
    assert blocks == ["let a = 1", "let b = 2"]  # only .md, sorted
