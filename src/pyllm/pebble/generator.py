"""Grow random-but-valid Pebble programs, following the grammar rules.

Like a robot that writes little programs by rolling dice at every choice: "should
this be a loop or a print? add or multiply? which variable?" Because it only ever
picks moves the grammar allows, every program it writes is syntactically valid --
which is exactly how we manufacture unlimited, always-correct training data when
there's no real Pebble code to be found.
"""

from pyllm.pebble.ast import Assign
from pyllm.pebble.ast import Bin
from pyllm.pebble.ast import Call
from pyllm.pebble.ast import For
from pyllm.pebble.ast import Func
from pyllm.pebble.ast import If
from pyllm.pebble.ast import Let
from pyllm.pebble.ast import Num
from pyllm.pebble.ast import Print
from pyllm.pebble.ast import Program
from pyllm.pebble.ast import Return
from pyllm.pebble.ast import Unary
from pyllm.pebble.ast import Var
from pyllm.pebble.ast import While

_VARS = ["x", "y", "z", "a", "b", "c", "n", "total", "count", "result", "value"]
_FUNCS = ["compute", "helper", "calc", "combine", "step", "make"]
_CMPOPS = ["==", "!=", "<", "<=", ">", ">="]


def _choice(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _rand_value_expr(rng, names, depth):
    """A value-producing expression (number, variable, or arithmetic)."""
    if depth <= 0 or rng.random() < 0.4:
        if names and rng.random() < 0.5:
            return Var(_choice(rng, names))
        return Num(int(rng.integers(0, 100)))
    left = _rand_value_expr(rng, names, depth - 1)
    right = _rand_value_expr(rng, names, depth - 1)
    return Bin(_choice(rng, ["+", "-", "*", "%"]), left, right)


def _rand_condition(rng, names, depth):
    """A truthy expression for `if`/`while` -- a comparison or a boolean combo."""
    if depth > 0 and rng.random() < 0.3:
        op = _choice(rng, ["and", "or"])
        return Bin(op, _rand_condition(rng, names, depth - 1),
                   _rand_condition(rng, names, depth - 1))
    if rng.random() < 0.15:
        return Unary("not", _rand_condition(rng, names, 0))
    left = _rand_value_expr(rng, names, depth)
    right = _rand_value_expr(rng, names, depth)
    return Bin(_choice(rng, _CMPOPS), left, right)


def _rand_call(rng, names):
    n_args = int(rng.integers(0, 3))
    args = [_rand_value_expr(rng, names, 1) for _ in range(n_args)]
    return Call(_choice(rng, _FUNCS), args)


def _rand_simple_stmt(rng, names):
    """A leaf statement that never nests: let / reassignment / print."""
    roll = rng.random()
    if roll < 0.4 or not names:
        name = _choice(rng, _VARS)
        value = (
            _rand_call(rng, names)
            if rng.random() < 0.2
            else _rand_value_expr(rng, names, 2)
        )
        names.append(name)
        return Let(name, value)
    if roll < 0.7:
        return Assign(_choice(rng, names), _rand_value_expr(rng, names, 2))
    return Print(_choice(rng, [Var(_choice(rng, names)), _rand_call(rng, names)]))


def _rand_block(rng, names, depth, size):
    """A non-empty block of statements (bodies must not be empty)."""
    return [_rand_stmt(rng, names, depth) for _ in range(max(1, size))]


def _rand_stmt(rng, names, depth):
    """Any statement; control flow only appears while we still have depth budget."""
    if depth <= 0 or rng.random() < 0.55:
        return _rand_simple_stmt(rng, names)
    roll = rng.random()
    if roll < 0.4:
        orelse = _rand_block(rng, names, depth - 1, 1) if rng.random() < 0.5 else None
        return If(_rand_condition(rng, names, 1),
                  _rand_block(rng, names, depth - 1, 2), orelse)
    if roll < 0.7:
        return While(_rand_condition(rng, names, 1),
                     _rand_block(rng, names, depth - 1, 2))
    var = _choice(rng, ["i", "j", "k"])
    return For(var, Num(int(rng.integers(1, 6))),
               _rand_block(rng, names + [var], depth - 1, 2))


def random_program(rng, num_statements=8, max_depth=2):
    """Build one random valid Pebble program of roughly `num_statements` steps."""
    names = []
    statements = []
    # Seed a couple of variables so later statements have something to reference.
    for _ in range(2):
        statements.append(_rand_simple_stmt(rng, names))
    # Optionally define a small function up top.
    if rng.random() < 0.4:
        params = [
            _choice(rng, ["a", "b", "n"]) for _ in range(int(rng.integers(1, 3)))
        ]
        body = _rand_block(rng, list(params), 1, 2)
        body.append(Return(_rand_value_expr(rng, list(params), 1)))
        statements.append(Func(_choice(rng, _FUNCS), params, body))
    while len(statements) < num_statements:
        statements.append(_rand_stmt(rng, names, max_depth))
    return Program(statements)
