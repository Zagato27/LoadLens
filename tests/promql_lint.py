"""Offline syntax check for the PromQL subset the query packs use.

Covers vector selectors with matchers, range vectors, function calls, aggregations with
by/without (before or after the arguments), binary operators with bool / on / ignoring /
group_left / group_right, unary signs, numbers, strings, durations and subqueries ``[range:step]``.
Offsets and @ modifiers are reported as unsupported: the packs do not use them. This is a guard against typos
(unbalanced braces, a missing comma, an unknown function), not a replacement for Prometheus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

AGGREGATIONS = {
    "sum", "avg", "min", "max", "count", "group", "stddev", "stdvar",
    "topk", "bottomk", "quantile", "count_values", "limitk", "limit_ratio",
}
PARAM_AGGREGATIONS = {"topk", "bottomk", "quantile", "count_values", "limitk", "limit_ratio"}
# name -> (min args, max args or None, positions that must be range vectors, positions that must be strings)
FUNCTIONS: dict[str, tuple[int, Optional[int], set[int], set[int]]] = {
    **{name: (1, 1, {0}, set()) for name in (
        "rate", "irate", "increase", "delta", "idelta", "deriv", "changes", "resets",
        "avg_over_time", "min_over_time", "max_over_time", "sum_over_time", "count_over_time",
        "last_over_time", "stddev_over_time", "present_over_time",
    )},
    "quantile_over_time": (2, 2, {1}, set()),
    "histogram_quantile": (2, 2, set(), set()),
    "histogram_sum": (1, 1, set(), set()),
    "histogram_count": (1, 1, set(), set()),
    "histogram_avg": (1, 1, set(), set()),
    "label_replace": (5, 5, set(), {1, 2, 3, 4}),
    "label_join": (3, None, set(), set()),
    "vector": (1, 1, set(), set()),
    "scalar": (1, 1, set(), set()),
    "absent": (1, 1, set(), set()),
    "time": (0, 0, set(), set()),
    "round": (1, 2, set(), set()),
    "clamp": (3, 3, set(), set()),
    "clamp_min": (2, 2, set(), set()),
    "clamp_max": (2, 2, set(), set()),
    **{name: (1, 1, set(), set()) for name in ("abs", "ceil", "floor", "sqrt", "exp", "ln", "log2", "log10", "sgn", "sort", "sort_desc")},
}
KEYWORDS = {"by", "without", "on", "ignoring", "group_left", "group_right", "bool", "and", "or", "unless", "offset", "atan2"}
LEVELS: list[tuple[set[str], bool]] = [
    ({"or"}, False),
    ({"and", "unless"}, False),
    ({"==", "!=", ">=", "<=", ">", "<"}, True),
    ({"+", "-"}, False),
    ({"*", "/", "%", "atan2"}, False),
]
MATCH_OPS = {"=", "!=", "=~", "!~"}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
   |(?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`[^`]*`)
   |(?P<duration>\d+(?:ms|[smhdwy])(?:\d+(?:ms|[smhdwy]))*(?![a-zA-Z0-9_]))
   |(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
   |(?P<ident>[a-zA-Z_][a-zA-Z0-9_:]*)
   |(?P<op>==|!=|>=|<=|=~|!~|[-+*/%^<>=,(){}\[\]@:])
    """,
    re.X,
)


class PromQLSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


def tokenize(query: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(query):
        match = _TOKEN_RE.match(query, pos)
        if not match:
            raise PromQLSyntaxError(f"unexpected character {query[pos]!r} at {pos}")
        kind = match.lastgroup or ""
        if kind != "ws":
            tokens.append(Token(kind, match.group(0), pos))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, query: str) -> None:
        self.query = query
        self.tokens = tokenize(query)
        self.index = 0

    # ---- token helpers ----
    def peek(self, offset: int = 0) -> Optional[Token]:
        position = self.index + offset
        return self.tokens[position] if position < len(self.tokens) else None

    def fail(self, message: str) -> PromQLSyntaxError:
        token = self.peek()
        where = f"at {token.pos} ({token.text!r})" if token else "at the end"
        return PromQLSyntaxError(f"{message} {where}")

    def next(self) -> Token:
        token = self.peek()
        if token is None:
            raise self.fail("unexpected end of query")
        self.index += 1
        return token

    def is_op(self, *texts: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "op" and token.text in texts

    def is_ident(self, *texts: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "ident" and token.text in texts

    def expect_op(self, text: str) -> None:
        if not self.is_op(text):
            raise self.fail(f"expected {text!r}")
        self.next()

    # ---- grammar ----
    def parse(self) -> None:
        self.expr()
        if self.peek() is not None:
            raise self.fail("unexpected token")

    def expr(self, level: int = 0) -> str:
        if level == len(LEVELS):
            return self.power()
        ops, allow_bool = LEVELS[level]
        kind = self.expr(level + 1)
        while self.peek() is not None and self.peek().text in ops and self.peek().kind in ("op", "ident"):
            self._no_range(kind)
            self.next()
            if allow_bool and self.is_ident("bool"):
                self.next()
            self.modifiers()
            self._no_range(self.expr(level + 1))
            kind = "instant"
        return kind

    def _no_range(self, kind: str) -> None:
        if kind == "range":
            raise self.fail("binary operators take instant vectors or scalars, not range vectors")

    def power(self) -> str:
        kind = self.unary()
        if self.is_op("^"):
            self._no_range(kind)
            self.next()
            self.modifiers()
            self._no_range(self.power())
            return "instant"
        return kind

    def unary(self) -> str:
        if self.is_op("-", "+"):
            self.next()
            self._no_range(self.unary())
            return "instant"
        return self.postfix()

    def postfix(self) -> str:
        kind = self.primary()
        if self.is_op("["):
            self.next()
            token = self.next()
            if token.kind != "duration":
                raise PromQLSyntaxError(f"expected a duration at {token.pos}, got {token.text!r}")
            if self.is_op(":"):
                # Subquery: [range:resolution] after any instant expression; the resolution is optional.
                self._no_range(kind)
                self.next()
                if not self.is_op("]"):
                    step = self.next()
                    if step.kind != "duration":
                        raise PromQLSyntaxError(f"expected a subquery resolution at {step.pos}, got {step.text!r}")
            elif kind != "selector":
                raise self.fail("range only after a vector selector")
            self.expect_op("]")
            kind = "range"
        if self.is_ident("offset") or self.is_op("@"):
            raise self.fail("offset and @ are not supported by this check")
        return kind

    def modifiers(self) -> None:
        if self.is_ident("on", "ignoring"):
            self.next()
            self.label_list()
            if self.is_ident("group_left", "group_right"):
                self.next()
                if self.is_op("("):
                    self.label_list()
        elif self.is_ident("group_left", "group_right"):
            raise self.fail("group_left/group_right need on() or ignoring()")

    def label_list(self) -> None:
        self.expect_op("(")
        while not self.is_op(")"):
            token = self.next()
            if token.kind != "ident":
                raise PromQLSyntaxError(f"expected a label name at {token.pos}, got {token.text!r}")
            if self.is_op(","):
                self.next()
            elif not self.is_op(")"):
                raise self.fail("expected ',' or ')' in a label list")
        self.expect_op(")")

    def matchers(self, require_one: bool) -> None:
        self.expect_op("{")
        count = 0
        while not self.is_op("}"):
            name = self.next()
            if name.kind != "ident":
                raise PromQLSyntaxError(f"expected a label name at {name.pos}, got {name.text!r}")
            op = self.next()
            if op.kind != "op" or op.text not in MATCH_OPS:
                raise PromQLSyntaxError(f"expected a matcher operator at {op.pos}, got {op.text!r}")
            value = self.next()
            if value.kind != "string":
                raise PromQLSyntaxError(f"expected a quoted value at {value.pos}, got {value.text!r}")
            count += 1
            if self.is_op(","):
                self.next()
            elif not self.is_op("}"):
                raise self.fail("expected ',' or '}' after a matcher")
        self.expect_op("}")
        if require_one and not count:
            raise self.fail("a selector without a metric name needs a matcher")

    def args(self) -> list[str]:
        self.expect_op("(")
        kinds: list[str] = []
        while not self.is_op(")"):
            kinds.append(self.expr())
            if self.is_op(","):
                self.next()
                if self.is_op(")"):
                    raise self.fail("trailing comma in arguments")
            elif not self.is_op(")"):
                raise self.fail("expected ',' or ')' in arguments")
        self.expect_op(")")
        return kinds

    def primary(self) -> str:
        token = self.peek()
        if token is None:
            raise self.fail("unexpected end of query")
        if token.kind == "number":
            self.next()
            return "number"
        if token.kind == "string":
            self.next()
            return "string"
        if token.kind == "op" and token.text == "(":
            self.next()
            kind = self.expr()
            self.expect_op(")")
            return "instant" if kind == "range" else kind
        if token.kind == "op" and token.text == "{":
            self.matchers(require_one=True)
            return "selector"
        if token.kind != "ident":
            raise self.fail("unexpected token")
        following = self.peek(1)
        if token.text in AGGREGATIONS and following is not None and (
            (following.kind == "op" and following.text == "(") or following.text in ("by", "without")
        ):
            return self.aggregation()
        if following is not None and following.kind == "op" and following.text == "(":
            return self.function()
        if token.text in KEYWORDS:
            raise self.fail("keyword where an expression was expected")
        self.next()
        if self.is_op("{"):
            self.matchers(require_one=False)
        return "selector"

    def aggregation(self) -> str:
        name = self.next().text
        grouped = False
        if self.is_ident("by", "without"):
            self.next()
            self.label_list()
            grouped = True
        kinds = self.args()
        expected = 2 if name in PARAM_AGGREGATIONS else 1
        if len(kinds) != expected:
            raise PromQLSyntaxError(f"{name}() takes {expected} argument(s), got {len(kinds)}")
        if "range" in kinds:
            raise PromQLSyntaxError(f"{name}() got a range vector")
        if not grouped and self.is_ident("by", "without"):
            self.next()
            self.label_list()
        return "instant"

    def function(self) -> str:
        token = self.next()
        spec = FUNCTIONS.get(token.text)
        if spec is None:
            raise PromQLSyntaxError(f"unknown function {token.text}() at {token.pos}")
        minimum, maximum, ranges, strings = spec
        kinds = self.args()
        if len(kinds) < minimum or (maximum is not None and len(kinds) > maximum):
            raise PromQLSyntaxError(f"{token.text}() got {len(kinds)} argument(s)")
        for position, kind in enumerate(kinds):
            if position in ranges and kind != "range":
                raise PromQLSyntaxError(f"{token.text}() argument {position + 1} must be a range vector")
            if position not in ranges and kind == "range":
                raise PromQLSyntaxError(f"{token.text}() argument {position + 1} must not be a range vector")
            if position in strings and kind != "string":
                raise PromQLSyntaxError(f"{token.text}() argument {position + 1} must be a string")
        return "instant"


def check_promql(query: str) -> None:
    """Raises PromQLSyntaxError when the query is not valid in the supported subset."""
    _Parser(query).parse()


__all__ = ["PromQLSyntaxError", "check_promql", "tokenize"]
