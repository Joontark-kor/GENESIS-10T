"""Tools the core hands to a model instead of letting it do the work in its head.

``calculate`` evaluates one arithmetic expression exactly (Python floats) and
refuses everything else: no names, no calls, no attributes. A model that writes
the formula but cannot divide reliably can therefore be paired with this.
"""
from __future__ import annotations

import ast
import operator
import re

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 12
_MAX_LENGTH = 400


class CalculationError(ValueError):
    """The expression is not plain arithmetic, or cannot be evaluated."""


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise CalculationError("exponent too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculationError("division by zero") from exc
    raise CalculationError(f"not plain arithmetic: {type(node).__name__}")


_TRANSLATE = str.maketrans({"×": "*", "÷": "/", "−": "-", "△": "-", "（": "(", "）": ")", "，": ","})


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _percent(match: re.Match) -> str:
    """'3%' means 3/100 inside a formula; a trailing '100%' is just the 'x 100' of a ratio and stays 100."""
    number = match.group(1)
    return number if float(number) == 100 else f"({number}/100)"


def clean(expression: str) -> str:
    """Normalize what models tend to write: thousands separators, × ÷ −, ^ as a power, a trailing '= result'."""
    text = expression.translate(_TRANSLATE).split("=")[0].replace(",", "")
    text = _PERCENT.sub(_percent, text)
    return text.replace("%", "").replace("^", "**").strip()  # models write 2^3 for a power (Python would read XOR)


def calculate(expression: str) -> float:
    text = clean(expression)
    if not text or len(text) > _MAX_LENGTH:
        raise CalculationError("empty or too long expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"cannot parse: {text[:60]!r}") from exc
    value = _eval(tree)
    if value != value or value in (float("inf"), float("-inf")):
        raise CalculationError("result is not finite")
    return float(value)


def format_number(value: float, digits: int = 2) -> str:
    """Round and print without trailing zeros: 11.0 -> '11', 61.97 -> '61.97'."""
    text = f"{round(value, digits):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text
