"""Grow random-but-valid Pebble programs, following the grammar rules.

Like a robot that writes little programs by rolling dice at every choice: "should
this be a loop or a print? add or multiply? which variable?" Because it only ever
picks moves the grammar allows, every program it writes is syntactically valid --
which is exactly how we manufacture unlimited, always-correct training data when
there's no real Pebble code to be found.
"""

from collections.abc import Sequence

import numpy as np

from pyllm.pebble.ast import (
    Assign,
    Bin,
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
    Unary,
    Var,
    While,
)

_VARS = ["x", "y", "z", "a", "b", "c", "n", "total", "count", "result", "value"]
_FUNCS = ["compute", "helper", "calc", "combine", "step", "make"]
_CMPOPS = ["==", "!=", "<", "<=", ">", ">="]

# How likely each random choice is. Each is a roll of `rng.random()` compared to it.
_STOP_EARLY = 0.4  # a value expression stops growing and becomes a leaf
_USE_VARIABLE = 0.5  # a leaf is a variable rather than a number
_JOIN_CONDITIONS = 0.3  # a condition is two conditions joined by and/or
_NEGATE = 0.15  # a condition is `not` something
_CALL_AS_VALUE = 0.2  # a `let` stores a function call instead of arithmetic
_MAKE_LET = 0.4  # a simple statement is a `let` (below this)
_MAKE_ASSIGN = 0.7  # ...or an assignment (below this), else a `print`
_STAY_SIMPLE = 0.55  # a statement is a leaf rather than control flow
_MAKE_IF = 0.4  # control flow is an `if` (below this)
_MAKE_WHILE = 0.7  # ...or a `while` (below this), else a `for`
_ADD_ELSE = 0.5  # an `if` has an `else` branch
_ADD_FUNCTION = 0.4  # the program defines a function up top


def _choice[T](rng: np.random.Generator, seq: Sequence[T]) -> T:
    """Pick one item from `seq` at random."""
    return seq[int(rng.integers(0, len(seq)))]


def _rand_value_expr(rng: np.random.Generator, names: list[str], depth: int) -> Expr:
    """Build a value-producing expression (number, variable, or arithmetic)."""
    if depth <= 0 or rng.random() < _STOP_EARLY:
        if names and rng.random() < _USE_VARIABLE:
            return Var(_choice(rng, names))
        return Num(int(rng.integers(0, 100)))
    left = _rand_value_expr(rng, names, depth - 1)
    right = _rand_value_expr(rng, names, depth - 1)
    return Bin(_choice(rng, ["+", "-", "*", "%"]), left, right)


def _rand_condition(rng: np.random.Generator, names: list[str], depth: int) -> Expr:
    """Build a truthy expression for `if`/`while` -- a comparison or a boolean combo."""
    if depth > 0 and rng.random() < _JOIN_CONDITIONS:
        op = _choice(rng, ["and", "or"])
        return Bin(
            op,
            _rand_condition(rng, names, depth - 1),
            _rand_condition(rng, names, depth - 1),
        )
    if rng.random() < _NEGATE:
        return Unary("not", _rand_condition(rng, names, 0))
    left = _rand_value_expr(rng, names, depth)
    right = _rand_value_expr(rng, names, depth)
    return Bin(_choice(rng, _CMPOPS), left, right)


def _rand_call(rng: np.random.Generator, names: list[str]) -> Call:
    n_args = int(rng.integers(0, 3))
    args = [_rand_value_expr(rng, names, 1) for _ in range(n_args)]
    return Call(_choice(rng, _FUNCS), args)


def _rand_simple_stmt(rng: np.random.Generator, names: list[str]) -> Stmt:
    """Build a leaf statement that never nests: let / reassignment / print."""
    roll = rng.random()
    if roll < _MAKE_LET or not names:
        name = _choice(rng, _VARS)
        value = (
            _rand_call(rng, names)
            if rng.random() < _CALL_AS_VALUE
            else _rand_value_expr(rng, names, 2)
        )
        names.append(name)
        return Let(name, value)
    if roll < _MAKE_ASSIGN:
        return Assign(_choice(rng, names), _rand_value_expr(rng, names, 2))
    return Print(_choice(rng, [Var(_choice(rng, names)), _rand_call(rng, names)]))


def _rand_block(rng: np.random.Generator, names: list[str], depth: int, size: int) -> list[Stmt]:
    """Build a non-empty block of statements (bodies must not be empty)."""
    return [_rand_stmt(rng, names, depth) for _ in range(max(1, size))]


def _rand_stmt(rng: np.random.Generator, names: list[str], depth: int) -> Stmt:
    """Any statement; control flow only appears while we still have depth budget."""
    if depth <= 0 or rng.random() < _STAY_SIMPLE:
        return _rand_simple_stmt(rng, names)
    roll = rng.random()
    if roll < _MAKE_IF:
        orelse = _rand_block(rng, names, depth - 1, 1) if rng.random() < _ADD_ELSE else None
        return If(
            _rand_condition(rng, names, 1),
            _rand_block(rng, names, depth - 1, 2),
            orelse,
        )
    if roll < _MAKE_WHILE:
        return While(_rand_condition(rng, names, 1), _rand_block(rng, names, depth - 1, 2))
    var = _choice(rng, ["i", "j", "k"])
    return For(var, Num(int(rng.integers(1, 6))), _rand_block(rng, [*names, var], depth - 1, 2))


def random_program(
    rng: np.random.Generator, num_statements: int = 8, max_depth: int = 2
) -> Program:
    """Build one random valid Pebble program of roughly `num_statements` steps."""
    names: list[str] = []
    # Seed a couple of variables so later statements have something to reference.
    statements: list[Stmt] = [_rand_simple_stmt(rng, names) for _ in range(2)]
    # Optionally define a small function up top.
    if rng.random() < _ADD_FUNCTION:
        params = [_choice(rng, ["a", "b", "n"]) for _ in range(int(rng.integers(1, 3)))]
        body = _rand_block(rng, list(params), 1, 2)
        body.append(Return(_rand_value_expr(rng, list(params), 1)))
        statements.append(Func(_choice(rng, _FUNCS), params, body))
    while len(statements) < num_statements:
        statements.append(_rand_stmt(rng, names, max_depth))
    return Program(statements)
