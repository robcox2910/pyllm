"""Turn a Pebble AST back into Pebble source text (the reverse of parsing).

Walking the family tree and writing out the text at each node. We wrap every
maths/logic expression in parentheses so the meaning can never get scrambled by
operator precedence -- Pebble's own formatter tidies away the extra brackets
afterwards.
"""

from pyllm.pebble.ast import (
    Assign,
    Bin,
    Bool,
    Call,
    Expr,
    For,
    Func,
    If,
    Let,
    Num,
    Print,
    Program,
    Return,
    Stmt,
    Str,
    Unary,
    Var,
    While,
)

_INDENT = "    "


def render(program: Program) -> str:
    """Render a whole Program to Pebble source (ends with a newline)."""
    lines: list[str] = []
    for stmt in program.statements:
        lines.extend(_render_stmt(stmt, 0))
    return "\n".join(lines) + "\n"


def _render_block(body: list[Stmt], depth: int) -> list[str]:
    """Render each statement of a block, one level deeper."""
    lines: list[str] = []
    for stmt in body:
        lines.extend(_render_stmt(stmt, depth))
    return lines


def _render_braced(header: str, body: list[Stmt], depth: int) -> list[str]:
    """Render `header {`, the indented body, then the closing `}`."""
    pad = _INDENT * depth
    return [f"{pad}{header} {{", *_render_block(body, depth + 1), f"{pad}}}"]


def _render_stmt(stmt: Stmt, depth: int) -> list[str]:
    """Render one statement (and anything nested inside it) as lines of text."""
    pad = _INDENT * depth
    match stmt:
        case Let(name, value):
            lines = [f"{pad}let {name} = {_render_expr(value)}"]
        case Assign(name, value):
            lines = [f"{pad}{name} = {_render_expr(value)}"]
        case Print(value):
            lines = [f"{pad}print({_render_expr(value)})"]
        case Return(value):
            lines = [f"{pad}return"] if value is None else [f"{pad}return {_render_expr(value)}"]
        case If(cond, body, orelse):
            lines = [f"{pad}if {_render_expr(cond)} {{"]
            lines.extend(_render_block(body, depth + 1))
            if orelse:
                lines.append(f"{pad}}} else {{")
                lines.extend(_render_block(orelse, depth + 1))
            lines.append(f"{pad}}}")
        case While(cond, body):
            lines = _render_braced(f"while {_render_expr(cond)}", body, depth)
        case For(var, count, body):
            lines = _render_braced(f"for {var} in range({_render_expr(count)})", body, depth)
        case Func(name, params, body):
            lines = _render_braced(f"fn {name}({', '.join(params)})", body, depth)
        case _:
            msg = f"cannot render statement: {stmt!r}"
            raise TypeError(msg)
    return lines


def _render_expr(expr: Expr) -> str:
    """Render one expression as text, with brackets so precedence can't bite."""
    match expr:
        case Num(value):
            text = str(value)
        case Bool(value):
            text = "true" if value else "false"
        case Str(value):
            text = f'"{value}"'
        case Var(name):
            text = name
        case Bin(op, left, right):
            text = f"({_render_expr(left)} {op} {_render_expr(right)})"
        case Unary("not", operand):
            text = f"(not {_render_expr(operand)})"
        case Unary(op, operand):
            text = f"({op}{_render_expr(operand)})"
        case Call(name, args):
            rendered = ", ".join(_render_expr(a) for a in args)
            text = f"{name}({rendered})"
        case _:
            msg = f"cannot render expression: {expr!r}"
            raise TypeError(msg)
    return text
