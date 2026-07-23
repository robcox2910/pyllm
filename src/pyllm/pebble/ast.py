"""A tiny AST for a slice of the Pebble language -- the shapes we can generate.

An AST ("abstract syntax tree") is a program drawn as a family tree: a `+` node
with two children, an `if` node with a condition and a body, and so on. We build
these trees at random (always following the rules) and then `render` turns a tree
back into text you could paste into Pebble. This is our own little copy of
Pebble's grammar -- just the parts we teach the model to write.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Num:
    """A whole number, like 42."""

    value: int


@dataclass(frozen=True)
class Bool:
    """A yes/no value: true or false."""

    value: bool


@dataclass(frozen=True)
class Str:
    """A piece of text in quotes."""

    value: str


@dataclass(frozen=True)
class Var:
    """A name that stands for a value, like `count`."""

    name: str


@dataclass(frozen=True)
class Bin:
    """Two values joined by an operator, like `a + b` or `x > 0`."""

    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Unary:
    """One value with an operator in front, like `-x` or `not done`."""

    op: str
    operand: Expr


@dataclass(frozen=True)
class Call:
    """Calling a function by name with some arguments, like `step(x)`."""

    name: str
    args: list


type Expr = Num | Bool | Str | Var | Bin | Unary | Call


@dataclass(frozen=True)
class Let:
    """Make a new named box and put a value in it: `let x = 1`."""

    name: str
    value: Expr


@dataclass(frozen=True)
class Assign:
    """Put a new value in an existing box: `x = 2`."""

    name: str
    value: Expr


@dataclass(frozen=True)
class Print:
    """Show a value on the screen: `print(x)`."""

    value: Expr


@dataclass(frozen=True)
class Return:
    """Hand a value back out of a function: `return x` (or bare `return`)."""

    value: Expr | None


@dataclass(frozen=True)
class If:
    """Do one thing if a test is true, optionally another thing otherwise."""

    cond: Expr
    body: list
    orelse: list | None


@dataclass(frozen=True)
class While:
    """Keep doing something while a test stays true."""

    cond: Expr
    body: list


@dataclass(frozen=True)
class For:
    """Repeat once for each number in a range: `for i in range(3) { ... }`."""

    var: str
    count: Expr
    body: list


@dataclass(frozen=True)
class Func:
    """Define a reusable function: `fn add(a, b) { ... }`."""

    name: str
    params: list
    body: list


type Stmt = Let | Assign | Print | Return | If | While | For | Func


@dataclass(frozen=True)
class Program:
    """A whole program: a list of statements, top to bottom."""

    statements: list
