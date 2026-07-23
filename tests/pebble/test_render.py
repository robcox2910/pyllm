from pyllm.pebble.ast import (
    Assign,
    Bin,
    Bool,
    Call,
    For,
    Func,
    If,
    Let,
    Num,
    Print,
    Program,
    Return,
    Unary,
    Var,
    While,
)
from pyllm.pebble.render import render


def test_render_let_and_binop():
    prog = Program([Let("x", Bin("+", Num(1), Num(2)))])
    assert render(prog) == "let x = (1 + 2)\n"


def test_render_reassignment_and_print():
    prog = Program([Assign("x", Num(5)), Print(Var("x"))])
    assert render(prog) == "x = 5\nprint(x)\n"


def test_render_bool_and_unary():
    prog = Program([Let("b", Unary("not", Bool(True)))])
    assert render(prog) == "let b = (not true)\n"


def test_render_if_else_block_is_brace_delimited():
    prog = Program(
        [
            If(Bin(">", Var("x"), Num(0)), [Print(Var("x"))], [Print(Num(0))]),
        ]
    )
    assert render(prog) == ("if (x > 0) {\n    print(x)\n} else {\n    print(0)\n}\n")


def test_render_while_and_for_and_call():
    prog = Program(
        [
            While(Bool(True), [Assign("x", Call("step", [Var("x")]))]),
            For("i", Num(3), [Print(Var("i"))]),
        ]
    )
    assert render(prog) == (
        "while true {\n    x = step(x)\n}\nfor i in range(3) {\n    print(i)\n}\n"
    )


def test_render_function_def_and_return():
    prog = Program([Func("add", ["a", "b"], [Return(Bin("+", Var("a"), Var("b")))])])
    assert render(prog) == ("fn add(a, b) {\n    return (a + b)\n}\n")
