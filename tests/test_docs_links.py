# SPDX-License-Identifier: Apache-2.0
"""The English user documentation under docs/ (plans/open-source-plan.md, section 5).

Offline: reads the Markdown files and parses settings.example.py with ``ast`` (nothing is imported or executed).

- Every relative link and image of the English pages and of docs/README.md points at an existing file or
  directory (anchors are stripped; http(s) and mailto links are not checked).
- The English pages contain no Cyrillic; docs/README.md may quote the Russian titles of the design notes inside
  guillemets.
- Every settings key that docs/configuration.md names as code exists in settings.example.py: the first cell of
  the ``Key`` / ``Section`` tables, and every dotted path whose first segment is a top-level or second-level key
  of the example (``storage.timescale.pool_max``, ``timescale.pool_max``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
SETTINGS_EXAMPLE = ROOT_DIR / "settings.example.py"

ENGLISH_DOCS = (
    "README.md",
    "getting-started.md",
    "configuration.md",
    "data-sources.md",
    "ai-analysis.md",
    "running-load-tests.md",
    "api.md",
    "deployment.md",
    "development.md",
)
# docs/README.md quotes the Russian titles of the design notes inside guillemets; nothing else may be Cyrillic.
CYRILLIC_ALLOWED_IN_QUOTES = {"README.md"}

CYRILLIC = re.compile("[Ѐ-ӿ]")
GUILLEMETS = re.compile("«[^»]*»")
# [text](target) and ![alt](target); the target ends at the first space or closing parenthesis.
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^```.*?^```", re.M | re.S)
KEY_PATH = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
TABLE_CELL_KEY = re.compile(r"^\|\s*`([^`|]+)`\s*(?:,\s*`[^`|]+`\s*)*\|")


def _read(name: str) -> str:
    return (DOCS_DIR / name).read_text(encoding="utf-8")


def _without_fences(text: str) -> str:
    return FENCE.sub("", text)


# ---- links --------------------------------------------------------------------------------------------------------


def _links(text: str) -> list[str]:
    return [match.group(1) for match in MARKDOWN_LINK.finditer(text)]


@pytest.mark.parametrize("name", ENGLISH_DOCS)
def test_relative_links_resolve(name: str) -> None:
    text = _read(name)
    broken = []
    for target in _links(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:  # a pure anchor into the same page
            continue
        resolved = (DOCS_DIR / path_part).resolve()
        if path_part.endswith("/"):
            ok = resolved.is_dir()
        else:
            ok = resolved.exists()
        if not ok:
            broken.append(target)
    assert not broken, f"docs/{name}: links to missing files: {broken}"


def test_index_lists_every_english_page() -> None:
    index = _read("README.md")
    linked = {target.split("#", 1)[0] for target in _links(index)}
    missing = [name for name in ENGLISH_DOCS if name != "README.md" and name not in linked]
    assert not missing, f"docs/README.md does not link: {missing}"
    for note in sorted(DOCS_DIR.glob("*.md")):
        if note.name not in ENGLISH_DOCS:
            assert note.name in linked, f"docs/README.md does not list {note.name}"


# ---- language -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ENGLISH_DOCS)
def test_english_docs_have_no_cyrillic(name: str) -> None:
    text = _read(name)
    if name in CYRILLIC_ALLOWED_IN_QUOTES:
        text = GUILLEMETS.sub("", text)
    offending = [
        f"{number}: {line.strip()[:100]}"
        for number, line in enumerate(text.splitlines(), 1)
        if CYRILLIC.search(line)
    ]
    assert not offending, f"docs/{name} contains Cyrillic:\n" + "\n".join(offending)


@pytest.mark.parametrize("name", ENGLISH_DOCS)
def test_english_docs_are_utf8_lf_without_bom(name: str) -> None:
    data = (DOCS_DIR / name).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), f"docs/{name} starts with a BOM"
    # LF in the repository; a Windows checkout with core.autocrlf turns every line into CRLF, never only some.
    assert data.count(b"\r\n") in (0, data.count(b"\n")), f"docs/{name} mixes CRLF and LF"
    assert b"\r" not in data.replace(b"\r\n", b""), f"docs/{name} has stray CR characters"
    data.decode("utf-8")


# ---- settings keys of configuration.md ----------------------------------------------------------------------------------


def _dict_paths(node: ast.AST, prefix: tuple[str, ...], out: set[tuple[str, ...]]) -> None:
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                path = prefix + (key.value,)
                out.add(path)
                _dict_paths(value, path, out)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            _dict_paths(item, prefix, out)


def _example_key_paths() -> set[tuple[str, ...]]:
    """Every key path of CONFIG in settings.example.py, at every depth."""
    tree = ast.parse(SETTINGS_EXAMPLE.read_text(encoding="utf-8"), filename=str(SETTINGS_EXAMPLE))
    paths: set[tuple[str, ...]] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CONFIG" for target in statement.targets
        ):
            _dict_paths(statement.value, (), paths)
    assert paths, "settings.example.py: no CONFIG dict found"
    return paths


def _is_subpath(needle: tuple[str, ...], paths: set[tuple[str, ...]]) -> bool:
    """``needle`` is a contiguous slice of some key path of the example."""
    size = len(needle)
    for path in paths:
        if len(path) < size:
            continue
        for start in range(len(path) - size + 1):
            if path[start:start + size] == needle:
                return True
    return False


def _table_header_keys(text: str) -> list[str]:
    """First cells of the rows of every table whose header starts with ``Key`` or ``Section``."""
    cells: list[str] = []
    header_active = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            header_active = False
            continue
        first = stripped.split("|")[1].strip()
        if first in ("Key", "Section"):
            header_active = True
            continue
        if not header_active or set(first) <= {"-", ":", " "}:
            continue
        match = TABLE_CELL_KEY.match(stripped)
        if not match:
            continue
        # A cell may list several keys: `a`, `b`, `c`
        cells.extend(re.findall(r"`([^`]+)`", stripped.split("|")[1]))
    return cells


def test_configuration_keys_exist_in_settings_example() -> None:
    paths = _example_key_paths()
    top_and_second = {path[0] for path in paths} | {path[1] for path in paths if len(path) > 1}
    text = _without_fences(_read("configuration.md"))

    unknown: list[str] = []
    for cell in _table_header_keys(text):
        if not KEY_PATH.match(cell):
            continue
        if not _is_subpath(tuple(cell.split(".")), paths):
            unknown.append(cell)

    for code in INLINE_CODE.findall(text):
        if "." not in code or not KEY_PATH.match(code):
            continue
        parts = tuple(code.split("."))
        if parts[0] not in top_and_second:
            continue  # a module, a cookie or a file name, not a settings path
        if not _is_subpath(parts, paths):
            unknown.append(code)

    assert not unknown, "docs/configuration.md names settings keys that settings.example.py does not have: " + ", ".join(
        sorted(set(unknown))
    )


def test_configuration_documents_every_top_level_section() -> None:
    paths = _example_key_paths()
    top_level = {path[0] for path in paths if len(path) == 1}
    text = _read("configuration.md")
    missing = sorted(key for key in top_level if f"`{key}`" not in text)
    assert not missing, f"docs/configuration.md does not mention the sections: {missing}"
