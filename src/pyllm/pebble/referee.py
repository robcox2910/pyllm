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

from functools import lru_cache

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


class _NeedMore(Exception):
    """The text stopped mid-move: not wrong yet, just unfinished."""


class _Foul(Exception):
    """The text broke a rule: no amount of extra typing can fix it."""


class _Parser:
    """A recursive-descent parser that reads characters left to right.

    Each `_parse_*` method reads one grammar shape. Running out of text raises
    `_NeedMore`; reading something the rules forbid raises `_Foul`.
    """

    def __init__(self, text):
        self.text = text
        self.pos = 0

    # -- reading helpers -------------------------------------------------------

    def _at_end(self):
        return self.pos >= len(self.text)

    def _peek(self):
        if self._at_end():
            raise _NeedMore
        return self.text[self.pos]

    def _expect(self, literal):
        """Read exactly `literal`, letter by letter."""
        for char in literal:
            if self._peek() != char:
                raise _Foul
            self.pos += 1

    def _choose(self, options):
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
            raise _NeedMore
        return None

    def _word(self):
        """Read a run of name letters. Returns (word, touched_the_end)."""
        start = self.pos
        while not self._at_end() and self.text[self.pos] in _NAME_CHARS:
            self.pos += 1
        return self.text[start : self.pos], self._at_end()

    def _name(self):
        """Read a new name (for let/for/fn/params): not a keyword, not empty."""
        if self._peek() not in _NAME_START:
            raise _Foul
        word, at_end = self._word()
        if at_end:
            raise _NeedMore  # "le" might grow into "level"
        if word in _KEYWORDS:
            raise _Foul
        return word

    # -- programs and blocks ---------------------------------------------------

    def parse_program(self):
        """A program: statements on their own lines, nothing indented."""
        self._parse_statement(depth=0, in_fn=False)
        while not self._at_end():
            self._expect("\n")
            if self._at_end():
                return  # one trailing newline is fine
            self._parse_statement(depth=0, in_fn=False)

    def _parse_block(self, depth, in_fn):
        """`{` already read: one or more indented statements, then `}`."""
        inner = _INDENT * (depth + 1)
        self._expect("\n" + inner)
        self._parse_statement(depth + 1, in_fn)
        while True:
            self._expect("\n")
            self._peek()  # the block can't simply stop: it needs a `}`
            step = self._choose([inner, _INDENT * depth + "}"])
            if step is None:
                raise _Foul
            if step != inner:
                return
            self._parse_statement(depth + 1, in_fn)

    # -- statements ------------------------------------------------------------

    def _parse_statement(self, depth, in_fn):
        char = self._peek()
        if char == "#":
            self._parse_comment()
            return
        if char not in _NAME_START:
            raise _Foul
        word, at_end = self._word()
        if at_end:
            raise _NeedMore  # a half-typed keyword or name
        match word:
            case "let":
                self._expect(" ")
                self._name()
                self._expect(" = ")
                self._parse_expr()
            case "print":
                self._expect("(")
                self._parse_expr()
                self._expect(")")
            case "return":
                if not in_fn:
                    raise _Foul  # you can only hand back a value from a function
                if self._choose([" "]):
                    self._parse_expr()
            case "if":
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
            case "while":
                self._expect(" ")
                self._parse_expr()
                self._expect(" {")
                self._parse_block(depth, in_fn)
            case "for":
                self._expect(" ")
                self._name()
                self._expect(" in range(")
                self._parse_expr()
                self._expect(") {")
                self._parse_block(depth, in_fn)
            case "fn":
                if depth > 0:
                    raise _Foul  # functions live at the top of the program
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
            case _ if word in _KEYWORDS:
                raise _Foul
            case _:
                self._expect(" = ")
                self._parse_expr()

    def _parse_comment(self):
        """`#` and then anything up to the end of the line."""
        while not self._at_end() and self.text[self.pos] != "\n":
            self.pos += 1

    # -- expressions, loosest-binding first ------------------------------------

    def _parse_expr(self):
        self._parse_and()
        while self._choose([" or "]):
            self._parse_and()

    def _parse_and(self):
        self._parse_not()
        while self._choose([" and "]):
            self._parse_not()

    def _parse_not(self):
        # Only a full "not " counts: "no" might just be the start of a name.
        if self.text.startswith("not ", self.pos):
            self.pos += len("not ")
            self._parse_not()
        else:
            self._parse_compare()

    def _parse_compare(self):
        self._parse_sum()
        if self._choose(_COMPARE_OPS):
            self._parse_sum()  # just one comparison: no `a < b < c`

    def _parse_sum(self):
        self._parse_term()
        while self._choose([" + ", " - "]):
            self._parse_term()

    def _parse_term(self):
        self._parse_unary()
        while self._choose([" * ", " / ", " % "]):
            self._parse_unary()

    def _parse_unary(self):
        self._choose(["-"])  # an optional minus sign in front
        self._parse_primary()

    def _parse_primary(self):
        char = self._peek()
        if char in _DIGITS:
            while not self._at_end() and self.text[self.pos] in _DIGITS:
                self.pos += 1
        elif char == '"':
            self.pos += 1
            while self._peek() != '"':
                if self._peek() in _STRING_BANNED:
                    raise _Foul
                self.pos += 1
            self.pos += 1
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
                    raise _NeedMore  # "not" might grow into "nothing"
                raise _Foul
            # A call: `name(` then either `)` straight away or some arguments.
            if self._choose(["("]) and self._choose([")"]) is None:
                self._parse_expr()
                while self._choose([", "]):
                    self._parse_expr()
                self._expect(")")
        else:
            raise _Foul


def _judge(program):
    """'complete', 'unfinished' or 'foul' for a single program's text."""
    try:
        _Parser(program).parse_program()
    except _NeedMore:
        return "unfinished"
    except _Foul:
        return "foul"
    return "complete"


@lru_cache(maxsize=4096)
def _program_complete(program):
    return _judge(program) == "complete"


def viable(text):
    """True if `text` could still grow into valid Pebble (no whistle yet)."""
    *finished, current = text.split("\n\n")
    if not all(_program_complete(p) for p in finished):
        return False
    return _judge(current) != "foul"


def complete(text):
    """True if `text` is finished, valid Pebble right now (full time)."""
    return all(_program_complete(p) for p in text.split("\n\n"))
