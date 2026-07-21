"""Turn a Pebble AST back into Pebble source text (the reverse of parsing).

Walking the family tree and writing out the text at each node. We wrap every
maths/logic expression in parentheses so the meaning can never get scrambled by
operator precedence -- Pebble's own formatter tidies away the extra brackets
afterwards.
"""

from pyllm.pebble.ast import Assign
from pyllm.pebble.ast import Bin
from pyllm.pebble.ast import Bool
from pyllm.pebble.ast import Call
from pyllm.pebble.ast import For
from pyllm.pebble.ast import Func
from pyllm.pebble.ast import If
from pyllm.pebble.ast import Let
from pyllm.pebble.ast import Num
from pyllm.pebble.ast import Print
from pyllm.pebble.ast import Return
from pyllm.pebble.ast import Str
from pyllm.pebble.ast import Unary
from pyllm.pebble.ast import Var
from pyllm.pebble.ast import While

_INDENT = "    "


def render(program):
    """Render a whole Program to Pebble source (ends with a newline)."""
    lines = []
    for stmt in program.statements:
        lines.extend(_render_stmt(stmt, 0))
    return "\n".join(lines) + "\n"


def _render_block(body, depth):
    lines = []
    for stmt in body:
        lines.extend(_render_stmt(stmt, depth))
    return lines


def _render_stmt(stmt, depth):
    pad = _INDENT * depth
    match stmt:
        case Let(name, value):
            return [f"{pad}let {name} = {_render_expr(value)}"]
        case Assign(name, value):
            return [f"{pad}{name} = {_render_expr(value)}"]
        case Print(value):
            return [f"{pad}print({_render_expr(value)})"]
        case Return(None):
            return [f"{pad}return"]
        case Return(value):
            return [f"{pad}return {_render_expr(value)}"]
        case If(cond, body, orelse):
            lines = [f"{pad}if {_render_expr(cond)} {{"]
            lines.extend(_render_block(body, depth + 1))
            if orelse:
                lines.append(f"{pad}}} else {{")
                lines.extend(_render_block(orelse, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        case While(cond, body):
            lines = [f"{pad}while {_render_expr(cond)} {{"]
            lines.extend(_render_block(body, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        case For(var, count, body):
            lines = [f"{pad}for {var} in range({_render_expr(count)}) {{"]
            lines.extend(_render_block(body, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        case Func(name, params, body):
            lines = [f"{pad}fn {name}({', '.join(params)}) {{"]
            lines.extend(_render_block(body, depth + 1))
            lines.append(f"{pad}}}")
            return lines
    raise TypeError(f"cannot render statement: {stmt!r}")


def _render_expr(expr):
    match expr:
        case Num(value):
            return str(value)
        case Bool(value):
            return "true" if value else "false"
        case Str(value):
            return f'"{value}"'
        case Var(name):
            return name
        case Bin(op, left, right):
            return f"({_render_expr(left)} {op} {_render_expr(right)})"
        case Unary("not", operand):
            return f"(not {_render_expr(operand)})"
        case Unary(op, operand):
            return f"({op}{_render_expr(operand)})"
        case Call(name, args):
            rendered = ", ".join(_render_expr(a) for a in args)
            return f"{name}({rendered})"
    raise TypeError(f"cannot render expression: {expr!r}")
