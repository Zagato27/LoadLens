# SPDX-License-Identifier: Apache-2.0
"""Cyrillic user-facing literals per file: the ratchet of the localization (plans/i18n-plan.md, section 2.4).

Scanned files:
- ``templates/*.html``: text between markup, quotes and Jinja delimiters, line by line; ``{# #}`` and ``<!-- -->``
  comments and the JS/CSS comments of inline ``<script>``/``<style>`` blocks are skipped;
- ``static/js/*.js`` (not ``static/vendor``): string, template and regex literals; comments are skipped;
- Python modules of ``AI/`` (``query_packs`` included), ``loadlens_app/``, ``confluence_manager/``,
  ``data_collectors/`` and ``update_page.py`` through the AST: ``str`` constants and f-strings, without docstrings and
  bare string statements, arguments of ``logger.*``/``print``/``log`` calls, ``if __name__ == "__main__":`` blocks
  and command-line code (CLI output is not localized).

Literals listed in ``locales/allowlist.json`` (``{path: [{"literal": ..., "reason": ...}]}``, ``"*"`` for every file)
do not count. ``locales/baseline.json`` stores the counts; tests/test_i18n_guard.py fails when a file's count grows
or a file without an entry gets one.

    python tools/i18n_extract.py [PATH ...]                    path:line: literal for every counted literal
    python tools/i18n_extract.py --summary [PATH ...]          counts per file next to the baseline
    python tools/i18n_extract.py --write-baseline [PATH ...]   store the counts: only the listed files (lowering
                                                               only, unless --force), every file without paths
"""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = ROOT / "locales" / "allowlist.json"
BASELINE_PATH = ROOT / "locales" / "baseline.json"

CYRILLIC = re.compile("[" + chr(0x0400) + "-" + chr(0x04FF) + "]")
BACKSLASH = chr(92)

PYTHON_ROOTS = ("AI", "loadlens_app", "confluence_manager", "data_collectors")
PYTHON_FILES = ("update_page.py",)
# Command-line tools and manual scripts: what they print is not localized.
CLI_MODULES = frozenset({
    "AI/query_packs/__main__.py",
    "AI/test_timescale_write.py",
    "loadlens_app/auth/cli.py",
})
# Command-line entry points inside modules that are scanned otherwise.
CLI_FUNCTIONS = {
    "loadlens_app/migrations.py": frozenset({"main"}),
}
LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal"})
_LOG_NAME = re.compile(r"_*(\w+_)?(log|logger|logging)", re.I)


@dataclass(frozen=True)
class Literal:
    path: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.text}"


def _display(text: str) -> str:
    """One-line form of a literal (also the form matched against the allowlist)."""
    return text.replace("\r\n", "\n").replace("\n", BACKSLASH + "n").replace("\t", BACKSLASH + "t")


class _Lines:
    def __init__(self, text: str) -> None:
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset)


# ---- JavaScript ---------------------------------------------------------------------------------------------

_JS_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_JS_NUMBER = re.compile(r"\.?\d[\w.]*")
# After these words a "/" starts a regular expression, after other identifiers it divides.
_JS_EXPRESSION_KEYWORDS = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do", "else", "yield", "await",
})


class _JsScanner:
    """Literals and comment spans of a JS source (strings, template literals with nested ``${}``, regexes)."""

    def __init__(self, source: str) -> None:
        self.src = source
        self.n = len(source)
        self.found: list[tuple[int, str]] = []
        self.comments: list[tuple[int, int]] = []

    def run(self) -> "_JsScanner":
        self._code(0, nested=False)
        return self

    def _code(self, i: int, nested: bool) -> int:
        src, n = self.src, self.n
        depth = 0
        division = False  # a "/" here divides (after a value) instead of starting a regex
        while i < n:
            c = src[i]
            if c in " \t\r\n":
                i += 1
                continue
            if c == "/" and i + 1 < n and src[i + 1] == "/":
                end = src.find("\n", i)
                end = n if end < 0 else end
                self.comments.append((i, end))
                i = end
                continue
            if c == "/" and i + 1 < n and src[i + 1] == "*":
                end = src.find("*/", i + 2)
                end = n if end < 0 else end + 2
                self.comments.append((i, end))
                i = end
                continue
            if c in "'\"":
                end, body = self._string(i, c)
                if CYRILLIC.search(body):
                    self.found.append((i, body))
                i, division = end, True
                continue
            if c == "`":
                i, division = self._template(i), True
                continue
            if c == "/":
                if division:
                    i, division = i + 1, False
                    continue
                end = self._regex_end(i)
                body = src[i:end]
                if CYRILLIC.search(body):
                    self.found.append((i, body))
                i, division = end, True
                continue
            match = _JS_IDENT.match(src, i)
            if match:
                i = match.end()
                division = match.group(0) not in _JS_EXPRESSION_KEYWORDS
                continue
            match = _JS_NUMBER.match(src, i)
            if match:
                i, division = match.end(), True
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                if nested and depth == 0:
                    return i + 1
                depth -= 1
            division = c in ")]"
            i += 1
        return i

    def _string(self, i: int, quote: str) -> tuple[int, str]:
        src, n = self.src, self.n
        j = i + 1
        while j < n:
            ch = src[j]
            if ch == BACKSLASH:
                j += 2
                continue
            if ch == quote:
                return j + 1, src[i + 1:j]
            if ch == "\n":  # unterminated: stop at the line end
                return j, src[i + 1:j]
            j += 1
        return n, src[i + 1:n]

    def _regex_end(self, i: int) -> int:
        src, n = self.src, self.n
        j = i + 1
        in_class = False
        while j < n:
            ch = src[j]
            if ch == BACKSLASH:
                j += 2
                continue
            if ch == "\n":
                return j
            if in_class:
                if ch == "]":
                    in_class = False
            elif ch == "[":
                in_class = True
            elif ch == "/":
                j += 1
                while j < n and (src[j].isalpha()):
                    j += 1
                return j
            j += 1
        return n

    def _template(self, i: int) -> int:
        src, n = self.src, self.n
        j = i + 1
        static: list[str] = []
        segment = j
        while j < n:
            ch = src[j]
            if ch == BACKSLASH:
                j += 2
                continue
            if ch == "`":
                static.append(src[segment:j])
                if CYRILLIC.search("".join(static)):
                    self.found.append((i, src[i + 1:j]))
                return j + 1
            if ch == "$" and j + 1 < n and src[j + 1] == "{":
                static.append(src[segment:j])
                j = self._code(j + 2, nested=True)
                segment = j
                continue
            j += 1
        static.append(src[segment:n])
        if CYRILLIC.search("".join(static)):
            self.found.append((i, src[i + 1:n]))
        return n


def strip_js_comments(source: str) -> str:
    """The source with comments blanked out (line breaks kept, so line numbers stay)."""
    scanner = _JsScanner(source).run()
    chars = list(source)
    for start, end in scanner.comments:
        for k in range(start, end):
            if chars[k] != "\n":
                chars[k] = " "
    return "".join(chars)


def js_literals(path: str, source: str) -> list[Literal]:
    lines = _Lines(source)
    scanner = _JsScanner(source).run()
    return [Literal(path, lines.line(offset), _display(body)) for offset, body in sorted(scanner.found)]


# ---- templates ----------------------------------------------------------------------------------------------

_TEMPLATE_COMMENT = re.compile(r"\{#.*?#\}|<!--.*?-->", re.S)
_INLINE_BLOCK = re.compile(r"(<(script|style)\b[^>]*>)(.*?)(</\2\s*>)", re.S | re.I)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_SEGMENT = re.compile(r"[^<>{}\"'\n]+")


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def template_literals(path: str, source: str) -> list[Literal]:
    text = _TEMPLATE_COMMENT.sub(lambda m: _blank(m.group(0)), source)

    def strip_block(match: re.Match) -> str:
        body = match.group(3)
        if match.group(2).lower() == "script":
            body = strip_js_comments(body)
        else:
            body = _CSS_COMMENT.sub(lambda m: _blank(m.group(0)), body)
        return match.group(1) + body + match.group(4)

    text = _INLINE_BLOCK.sub(strip_block, text)
    lines = _Lines(text)
    out = []
    for match in _SEGMENT.finditer(text):
        piece = " ".join(match.group(0).split())
        if piece and CYRILLIC.search(piece):
            out.append(Literal(path, lines.line(match.start()), piece))
    return out


# ---- Python -------------------------------------------------------------------------------------------------


def _is_log_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "print" or bool(_LOG_NAME.fullmatch(func.id))
    if isinstance(func, ast.Attribute):
        if func.attr == "log":
            return True
        if func.attr in LOG_METHODS:
            receiver = func.value
            name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, "id", "")
            return bool(_LOG_NAME.fullmatch(name or ""))
    return False


def _is_main_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


class _PythonLiterals(ast.NodeVisitor):
    def __init__(self, path: str, skip_functions: frozenset[str]) -> None:
        self.path = path
        self.skip_functions = skip_functions
        self.out: list[Literal] = []

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return  # docstring or a bare string statement
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in self.skip_functions:
            return
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_log_call(node.func):
            self.visit(node.func)
            return
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        parts: list[str] = []
        static: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                static.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + ast.unparse(value.value) + "}")
        if CYRILLIC.search("".join(static)):
            self.out.append(Literal(self.path, node.lineno, _display("".join(parts))))
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self.visit(value.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and CYRILLIC.search(node.value):
            self.out.append(Literal(self.path, node.lineno, _display(node.value)))


def python_literals(path: str, source: str) -> list[Literal]:
    visitor = _PythonLiterals(path, CLI_FUNCTIONS.get(path, frozenset()))
    visitor.visit(ast.parse(source, filename=path))
    return sorted(visitor.out, key=lambda item: item.line)


# ---- files, allowlist, baseline -----------------------------------------------------------------------------


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def scanned_files() -> list[str]:
    files = [_rel(p) for p in sorted((ROOT / "templates").glob("*.html"))]
    files += [_rel(p) for p in sorted((ROOT / "static" / "js").glob("*.js"))]
    for folder in PYTHON_ROOTS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = _rel(path)
            if rel not in CLI_MODULES:
                files.append(rel)
    files += [name for name in PYTHON_FILES if (ROOT / name).is_file()]
    return files


def literals_of(path: str) -> list[Literal]:
    """Every Cyrillic literal of one scanned file (allowlisted ones included)."""
    source = (ROOT / path).read_text(encoding="utf-8")
    if path.endswith(".html"):
        return template_literals(path, source)
    if path.endswith(".js"):
        return js_literals(path, source)
    if path.endswith(".py"):
        return python_literals(path, source)
    raise ValueError(f"not a scanned file type: {path}")


def load_allowlist(path: Optional[Path] = None) -> dict[str, set[str]]:
    path = path or ALLOWLIST_PATH
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: {entry["literal"] for entry in entries} for name, entries in data.items()}


def load_baseline(path: Optional[Path] = None) -> dict[str, int]:
    path = path or BASELINE_PATH
    if not path.is_file():
        return {}
    return {name: int(count) for name, count in json.loads(path.read_text(encoding="utf-8")).items()}


def counted(path: str, allowlist: Optional[dict[str, set[str]]] = None) -> list[Literal]:
    """Literals of a file that are not allowlisted."""
    allow = load_allowlist() if allowlist is None else allowlist
    own = allow.get(path, set())
    shared = allow.get("*", set())
    return [item for item in literals_of(path) if item.text not in own and item.text not in shared]


def counts(paths: Optional[Iterable[str]] = None) -> dict[str, int]:
    allow = load_allowlist()
    return {path: len(counted(path, allow)) for path in (scanned_files() if paths is None else paths)}


def dump_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _normalize_paths(raw: Iterable[str]) -> list[str]:
    known = set(scanned_files())
    out = []
    for item in raw:
        candidate = Path(item)
        rel = _rel(candidate if candidate.is_absolute() else ROOT / candidate)
        if rel not in known:
            raise SystemExit(f"not a scanned file: {item}")
        out.append(rel)
    return out


def write_baseline(paths: Optional[list[str]], force: bool = False) -> int:
    """Rewrites the baseline; returns the number of refused entries (a listed file may only go down)."""
    baseline = load_baseline()
    refused = 0
    if paths is None:
        current = counts()
        updated = {path: count for path, count in current.items() if count > 0 or path in baseline}
    else:
        updated = dict(baseline)
        for path, count in counts(paths).items():
            old = baseline.get(path, 0)
            if count > old and not force:
                print(f"refused: {path} would grow from {old} to {count} (translate it, allowlist it, or --force)")
                refused += 1
                continue
            updated[path] = count
    BASELINE_PATH.write_bytes(dump_json(updated).encode("utf-8"))
    return refused


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cyrillic user-facing literals per file against locales/baseline.json.")
    parser.add_argument("paths", nargs="*", help="files to scan (default: every scanned file)")
    parser.add_argument("--summary", action="store_true", help="counts per file next to the baseline")
    parser.add_argument("--write-baseline", action="store_true", help="store the counts of the listed files (all without paths)")
    parser.add_argument("--force", action="store_true", help="with --write-baseline: allow a listed file to grow")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    paths = _normalize_paths(args.paths) if args.paths else None

    if args.write_baseline:
        refused = write_baseline(paths, force=args.force)
        print(f"baseline written: {BASELINE_PATH.relative_to(ROOT).as_posix()}")
        return 1 if refused else 0

    if args.summary:
        baseline = load_baseline()
        current = counts(paths)
        total = 0
        for path, count in sorted(current.items()):
            if count or baseline.get(path):
                print(f"{count:6d} {baseline.get(path, 0):6d}  {path}")
            total += count
        print(f"{total:6d} {sum(baseline.get(p, 0) for p in current):6d}  total (current, baseline) in {len(current)} files")
        return 0

    allow = load_allowlist()
    for path in paths or scanned_files():
        for item in counted(path, allow):
            print(item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
