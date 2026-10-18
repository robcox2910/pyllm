"""The referee: a rule-checker that watches Pebble being written, one letter at a time.

Imagine a football referee who knows every rule of the game. They don't tell the
players what to do -- they only blow the whistle when a move is *illegal*. This
module is that referee for Pebble code. Give it some half-written text and it
answers two questions:

- `viable(text)`: "could this still grow into a valid program?" (no whistle yet)
- `complete(text)`: "is this a finished, valid program right now?" (full time)

It only knows the canonical slice of Pebble our corpus is written in (4-space
indents, single spaces around operators), which is a strict subset of real
Pebble -- so anything the referee lets through, Pebble's own parser accepts too.

Programs are separated by a blank line, just like in the training corpus. Every
earlier program must be complete; only the last one may still be in progress.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

_INDENT = "    "
# Every word Pebble reserves (so it can never be a name), plus `print` and
# `range`, which our tidy slice of Pebble only uses in their special spots.
_KEYWORDS = frozenset(
    [
        "const",
        "let",
        "if",
        "else",
        "while",
        "for",
        "in",
        "fn",
        "return",
        "yield",
        "break",
        "continue",
        "try",
        "catch",
        "finally",
        "throw",
        "match",
        "case",
        "struct",
        "class",
        "enum",
        "extends",
        "super",
        "import",
        "from",
        "true",
        "false",
        "null",
        "async",
        "await",
        "and",
        "or",
        "not",
        "print",
        "range",
    ]
)
_VALUES = frozenset(["true", "false", "null"])
_NAME_START = "abcdefghijklmnopqrstuvwxyz_"
_NAME_CHARS = _NAME_START + "0123456789"
_DIGITS = "0123456789"
_STRING_BANNED = '"\\\n{}'
_COMPARE_OPS = (" == ", " != ", " < ", " <= ", " > ", " >= ")
# How many brackets, calls, `not`s and blocks may be open at once. The parser
# calls itself once per level, so without a limit very deep nesting runs out of
# Python's stack (a RecursionError) instead of getting the whistle. Real code
# never comes close; a model stuck typing "(((((" does.
_MAX_NESTING = 50


class _NeedMoreError(Exception):
    """The text stopped mid-move: not wrong yet, just unfinished."""


class _FoulError(Exception):
    """The text broke a rule: no amount of extra typing can fix it."""


class _Parser:
    """A recursive-descent parser that reads characters left to right.

    Each `_parse_*` method reads one grammar shape. Running out of text raises
    `_NeedMoreError`; reading something the rules forbid raises `_FoulError`.
    """

    def __init__(self, text: str) -> None:
        """Start reading `text` from the first character."""
        self.text = text
        self.pos = 0
        self.nesting = 0

    # -- nesting ---------------------------------------------------------------

    def _open(self) -> None:
        """Step one level deeper (a bracket, call, `not` or block); too deep is a foul."""
        self.nesting += 1
        if self.nesting > _MAX_NESTING:
            raise _FoulError

    def _close(self) -> None:
        """Step back out of one level."""
        self.nesting -= 1

    # -- reading helpers -------------------------------------------------------

    def _at_end(self) -> bool:
        return self.pos >= len(self.text)

    def _peek(self) -> str:
        if self._at_end():
            raise _NeedMoreError
        return self.text[self.pos]

    def _expect(self, literal: str) -> None:
        """Read exactly `literal`, letter by letter."""
        for char in literal:
            if self._peek() != char:
                raise _FoulError
            self.pos += 1

    def _choose(self, options: Sequence[str]) -> str | None:
        """Read whichever option comes next; None if none of them do.

        If the text ends part-way through an option ("a" of " and "), we can't
        tell yet, so we ask for more.
        """
        rest = self.text[self.pos :]
        if not rest:
            return None  # the text simply ends here; the caller decides if that's ok
        for option in options:
            if rest.startswith(option):
                self.pos += len(option)
                return option
        if any(option.startswith(rest) for option in options):
            raise _NeedMoreError
        return None

    def _word(self) -> tuple[str, bool]:
        """Read a run of name letters. Returns (word, touched_the_end)."""
        start = self.pos
        while not self._at_end() and self.text[self.pos] in _NAME_CHARS:
            self.pos += 1
        return self.text[start : self.pos], self._at_end()

    def _name(self) -> str:
        """Read a new name (for let/for/fn/params): not a keyword, not empty."""
        if self._peek() not in _NAME_START:
            raise _FoulError
        word, at_end = self._word()
        if at_end:
            raise _NeedMoreError  # "le" might grow into "level"
        if word in _KEYWORDS:
            raise _FoulError
        return word

    # -- programs and blocks ---------------------------------------------------

    def parse_program(self) -> None:
        """Read a program: statements on their own lines, nothing indented."""
        self._parse_statement(depth=0, in_fn=False)
        while not self._at_end():
            self._expect("\n")
            if self._at_end():
                return  # one trailing newline is fine
            self._parse_statement(depth=0, in_fn=False)

    def _parse_block(self, depth: int, in_fn: bool) -> None:
        """`{` already read: one or more indented statements, then `}`."""
        inner = _INDENT * (depth + 1)
        self._open()
        self._expect("\n" + inner)
        self._parse_statement(depth + 1, in_fn)
        while True:
            self._expect("\n")
            self._peek()  # the block can't simply stop: it needs a `}`
            step = self._choose([inner, _INDENT * depth + "}"])
            if step is None:
                raise _FoulError
            if step != inner:
                self._close()
                return
            self._parse_statement(depth + 1, in_fn)

    # -- statements ------------------------------------------------------------

    def _parse_statement(self, depth: int, in_fn: bool) -> None:
        char = self._peek()
        if char == "#":
            self._parse_comment()
            return
        if char not in _NAME_START:
            raise _FoulError
        word, at_end = self._word()
        if at_end:
            raise _NeedMoreError  # a half-typed keyword or name
        match word:
            case "let":
                self._stmt_let()
            case "print":
                self._stmt_print()
            case "return":
                self._stmt_return(in_fn)
            case "if":
                self._stmt_if(depth, in_fn)
            case "while":
                self._stmt_while(depth, in_fn)
            case "for":
                self._stmt_for(depth, in_fn)
            case "fn":
                self._stmt_fn(depth)
            case _ if word in _KEYWORDS:
                raise _FoulError
            case _:
                self._expect(" = ")
                self._parse_expr()

    def _stmt_let(self) -> None:
        """Read `let name = value`."""
        self._expect(" ")
        self._name()
        self._expect(" = ")
        self._parse_expr()

    def _stmt_print(self) -> None:
        """Read `print(value)`."""
        self._expect("(")
        self._parse_expr()
        self._expect(")")

    def _stmt_return(self, in_fn: bool) -> None:
        """Read `return` with an optional value (only legal inside a function)."""
        if not in_fn:
            raise _FoulError  # you can only hand back a value from a function
        if self._choose([" "]):
            self._parse_expr()

    def _stmt_if(self, depth: int, in_fn: bool) -> None:
        """Read `if test { ... }` with any `else if` and `else` branches."""
        self._expect(" ")
        self._parse_expr()
        self._expect(" {")
        self._parse_block(depth, in_fn)
        while branch := self._choose([" else {", " else if "]):
            if branch == " else {":
                self._parse_block(depth, in_fn)
                break
            self._parse_expr()  # `else if`: another test, another block
            self._expect(" {")
            self._parse_block(depth, in_fn)

    def _stmt_while(self, depth: int, in_fn: bool) -> None:
        """Read `while test { ... }`."""
        self._expect(" ")
        self._parse_expr()
        self._expect(" {")
        self._parse_block(depth, in_fn)

    def _stmt_for(self, depth: int, in_fn: bool) -> None:
        """Read `for name in range(count) { ... }`."""
        self._expect(" ")
        self._name()
        self._expect(" in range(")
        self._parse_expr()
        self._expect(") {")
        self._parse_block(depth, in_fn)

    def _stmt_fn(self, depth: int) -> None:
        """Read `fn name(params) { ... }` (only at the top of the program)."""
        if depth > 0:
            raise _FoulError  # functions live at the top of the program
        self._expect(" ")
        self._name()
        self._expect("(")
        if self._choose([")"]) is None:
            self._name()
            while self._choose([", "]):
                self._name()
            self._expect(")")
        self._expect(" {")
        self._parse_block(depth, in_fn=True)

    def _parse_comment(self) -> None:
        """`#` and then anything up to the end of the line."""
        while not self._at_end() and self.text[self.pos] != "\n":
            self.pos += 1

    # -- expressions, loosest-binding first ------------------------------------

    def _parse_expr(self) -> None:
        self._open()
        self._parse_and()
        while self._choose([" or "]):
            self._parse_and()
        self._close()

    def _parse_and(self) -> None:
        self._parse_not()
        while self._choose([" and "]):
            self._parse_not()

    def _parse_not(self) -> None:
        # Only a full "not " counts: "no" might just be the start of a name.
        if self.text.startswith("not ", self.pos):
            self.pos += len("not ")
            self._open()
            self._parse_not()
            self._close()
        else:
            self._parse_compare()

    def _parse_compare(self) -> None:
        self._parse_sum()
        if self._choose(_COMPARE_OPS):
            self._parse_sum()  # just one comparison: no `a < b < c`

    def _parse_sum(self) -> None:
        self._parse_term()
        while self._choose([" + ", " - "]):
            self._parse_term()

    def _parse_term(self) -> None:
        self._parse_unary()
        while self._choose([" * ", " / ", " % "]):
            self._parse_unary()

    def _parse_unary(self) -> None:
        self._choose(["-"])  # an optional minus sign in front
        self._parse_primary()

    def _parse_string(self) -> None:
        """Read a double-quoted string that avoids the banned characters."""
        self.pos += 1
        while self._peek() != '"':
            if self._peek() in _STRING_BANNED:
                raise _FoulError
            self.pos += 1
        self.pos += 1

    def _parse_primary(self) -> None:
        char = self._peek()
        if char in _DIGITS:
            while not self._at_end() and self.text[self.pos] in _DIGITS:
                self.pos += 1
        elif char == '"':
            self._parse_string()
        elif char == "(":
            self.pos += 1
            self._parse_expr()
            self._expect(")")
        elif char in _NAME_START:
            word, at_end = self._word()
            if word in _VALUES:
                return
            if word in _KEYWORDS:
                if at_end:
                    raise _NeedMoreError  # "not" might grow into "nothing"
                raise _FoulError
            # A call: `name(` then either `)` straight away or some arguments.
            if self._choose(["("]) and self._choose([")"]) is None:
                self._parse_expr()
                while self._choose([", "]):
                    self._parse_expr()
                self._expect(")")
        else:
            raise _FoulError


def _judge(program: str) -> Literal["complete", "unfinished", "foul"]:
    """'complete', 'unfinished' or 'foul' for a single program's text."""
    try:
        _Parser(program).parse_program()
    except _NeedMoreError:
        return "unfinished"
    except _FoulError:
        return "foul"
    return "complete"


@lru_cache(maxsize=4096)
def _program_complete(program: str) -> bool:
    """Say whether one program is finished (cached, since programs repeat)."""
    return _judge(program) == "complete"


def viable(text: str) -> bool:
    """Say whether `text` could still grow into valid Pebble (no whistle yet)."""
    *finished, current = text.split("\n\n")
    if not all(_program_complete(p) for p in finished):
        return False
    return _judge(current) != "foul"


def complete(text: str) -> bool:
    """Say whether `text` is finished, valid Pebble right now (full time)."""
    return all(_program_complete(p) for p in text.split("\n\n"))
