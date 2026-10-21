# SPDX-License-Identifier: Apache-2.0
"""Add or check the SPDX license header of the project's own Python and JavaScript files.

    python tools/license_headers.py --check            # exit 1 and list the in-scope files without the header
    python tools/license_headers.py --write [paths]    # add the header to every in-scope file, or to the given files
    python tools/license_headers.py --list             # print the in-scope files

Scope: every ``*.py`` file of the repository and ``static/js/*.js`` that git tracks or would track (untracked files
are included, ignored ones such as settings.py and metrics_config.py are not). Not in scope: static/vendor/,
locales/, tests/fixtures/ and the generated files in EXCLUDED. Without git the files are found by walking the tree
with the same exclusions plus the names of the secret configuration files.

Header: ``# SPDX-License-Identifier: Apache-2.0`` for Python, placed after a shebang and an encoding line and before
the module docstring (comments may precede ``from __future__`` imports); ``// SPDX-License-Identifier: Apache-2.0``
as the first line of JavaScript. No copyright line: NOTICE carries it. Each file keeps its line endings and BOM.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SPDX = "SPDX-License-Identifier: Apache-2.0"
HEADERS = {"py": "# " + SPDX, "js": "// " + SPDX}
# Path prefixes (repository-relative, "/"-separated) that are never in scope.
EXCLUDED_PREFIXES = ("static/vendor/", "locales/", "tests/fixtures/", "docker/dev-stack/gatling/bundle/")
# Generated or foreign files inside the scope directories (repository-relative paths).
EXCLUDED = frozenset()
# Fallback for a checkout without git: directories and secret files never to touch.
WALK_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode", ".claude", ".deepeval",
                  ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist", "htmlcov"}
WALK_SKIP_FILES = {"settings.py", "metrics_config.py", "AI/config.py"}
_BOM = b"\xef\xbb\xbf"
_CODING = re.compile(rb"^#.*coding[:=]")


def kind_of(path: str) -> str | None:
    """``"py"`` / ``"js"`` for an in-scope path, ``None`` otherwise."""
    rel = path.replace("\\", "/")
    if rel.startswith(EXCLUDED_PREFIXES) or rel in EXCLUDED or "/__pycache__/" in rel:
        return None
    if rel.endswith(".py"):
        return "py"
    if rel.endswith(".js") and rel.startswith("static/js/") and rel.count("/") == 2:
        return "js"
    return None


def _git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py", "static/js/*.js"],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def _walk_files(root: Path) -> list[str]:
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in WALK_SKIP_DIRS)
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), root).replace("\\", "/")
            if rel in WALK_SKIP_FILES or ".local." in name or ".secrets." in name:
                continue
            found.append(rel)
    return found


def in_scope_files(root: Path = ROOT) -> list[str]:
    """Repository-relative paths of the files that must carry the header, sorted."""
    candidates = _git_files(root)
    if candidates is None:
        candidates = _walk_files(root)
    return sorted({p for p in candidates if kind_of(p) and (root / p).is_file()})


def has_header(data: bytes, kind: str) -> bool:
    head = data[len(_BOM):] if data.startswith(_BOM) else data
    return any(SPDX.encode() in line for line in head.split(b"\n", 4)[:4])


def with_header(data: bytes, kind: str) -> bytes:
    """``data`` with the header inserted; unchanged when it is already there. Keeps BOM and line endings."""
    if has_header(data, kind):
        return data
    bom = _BOM if data.startswith(_BOM) else b""
    body = data[len(bom):]
    newline = b"\r\n" if b"\r\n" in body.split(b"\n", 1)[0] + b"\n" else b"\n"
    lines = body.split(b"\n")  # every line keeps its own "\r", so the lines are re-joined with "\n" only
    index = 0
    if kind == "py":
        if lines and lines[0].startswith(b"#!"):
            index = 1
        if len(lines) > index and _CODING.match(lines[index].rstrip(b"\r")):
            index += 1
    header = HEADERS[kind].encode()
    if not body:
        return bom + header + newline
    prefix = b"\n".join(lines[:index]) + b"\n" if index else b""
    return bom + prefix + header + newline + b"\n".join(lines[index:])


def missing_headers(root: Path = ROOT, paths: Iterable[str] | None = None) -> list[str]:
    """In-scope files (or the given ones) that do not start with the header."""
    targets = list(paths) if paths is not None else in_scope_files(root)
    missing = []
    for rel in targets:
        kind = kind_of(rel)
        if kind and not has_header((root / rel).read_bytes(), kind):
            missing.append(rel)
    return missing


def write_headers(root: Path = ROOT, paths: Iterable[str] | None = None) -> list[str]:
    """Adds the header where it is missing; returns the files that changed."""
    changed = []
    for rel in missing_headers(root, paths):
        path = root / rel
        path.write_bytes(with_header(path.read_bytes(), kind_of(rel)))
        changed.append(rel)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="list the in-scope files without the header; exit 1 if any")
    group.add_argument("--write", action="store_true", help="add the missing headers (to PATHS when given)")
    group.add_argument("--list", action="store_true", help="print the in-scope files")
    parser.add_argument("paths", nargs="*", help="repository-relative files for --write (default: every in-scope file)")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(in_scope_files(ROOT)))
        return 0
    if args.check:
        missing = missing_headers(ROOT)
        for rel in missing:
            print(rel)
        print(f"{len(missing)} file(s) without the SPDX header" if missing else "all in-scope files carry the SPDX header")
        return 1 if missing else 0
    paths = [p.replace("\\", "/") for p in args.paths] or None
    if paths:
        unknown = [p for p in paths if not kind_of(p) or not (ROOT / p).is_file()]
        if unknown:
            print("not in scope or missing: " + ", ".join(unknown), file=sys.stderr)
            return 2
    changed = write_headers(ROOT, paths)
    for rel in changed:
        print(rel)
    print(f"{len(changed)} file(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
