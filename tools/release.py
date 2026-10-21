# SPDX-License-Identifier: Apache-2.0
"""Prepare a LoadLens release: version, changelog, third-party table and license headers.

    python tools/release.py --version 1.1.0   # bump, date the changelog, refresh the notices, print next steps
    python tools/release.py --check           # version.py and CHANGELOG.md agree (CI and tests)
    python tools/release.py --check --tag v1.1.0   # ... and the tag names the same version (release workflow)
    python tools/release.py --notes 1.1.0     # the CHANGELOG section of a version, for the GitHub Release

The script edits files only; it never runs a git command that changes state. After ``--version`` the owner
reviews the diff, commits, tags ``vX.Y.Z`` and pushes the tag, which starts .github/workflows/release.yml.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "loadlens_app" / "version.py"
CHANGELOG = ROOT / "CHANGELOG.md"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
TABLE_TOOL = ROOT / "tools" / "third_party_table.py"
HEADERS_TOOL = ROOT / "tools" / "license_headers.py"

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
VERSION_LINE = re.compile(r'^__version__ = "([^"]+)"$', re.M)
SOURCE_LINE = re.compile(r'^SOURCE_URL = "([^"]+)"$', re.M)
SECTION = re.compile(r"^## \[([^\]]+)\](?: - (\S+))?[ \t]*$", re.M)
# The generated part of THIRD_PARTY_NOTICES.md sits between these markers (tools/third_party_table.py).
TABLE_START = "<!-- python-deps:start -->"
TABLE_END = "<!-- python-deps:end -->"


def read_version() -> str:
    match = VERSION_LINE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{VERSION_FILE}: no __version__ line")
    return match.group(1)


def read_source_url() -> str:
    match = SOURCE_LINE.search(VERSION_FILE.read_text(encoding="utf-8"))
    return match.group(1) if match else ""


def _sections(text: str) -> list[tuple[str, str | None, int, int]]:
    """(version, date, start, end) of every ``## [version] - date`` section; ``end`` is the start of the next one."""
    found = list(SECTION.finditer(text))
    out = []
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else _links_start(text)
        out.append((match.group(1), match.group(2), match.start(), end))
    return out


def _links_start(text: str) -> int:
    """Where the reference links (``[1.0.0]: https://...``) at the end of the file begin."""
    match = re.search(r"^\[[^\]]+\]: \S+[ \t]*$", text, re.M)
    return match.start() if match else len(text)


def _body(text: str, start: int, end: int) -> str:
    """Section text without its heading line."""
    heading_end = text.index("\n", start) + 1
    return text[heading_end:end].strip("\n")


def changelog_section(version: str, text: str | None = None) -> str | None:
    text = CHANGELOG.read_text(encoding="utf-8") if text is None else text
    for name, _date, start, end in _sections(text):
        if name == version:
            return _body(text, start, end)
    return None


def check(tag: str | None = None) -> list[str]:
    """Problems that stop a release: an invalid or inconsistent version, a changelog without the version."""
    problems: list[str] = []
    version = read_version()
    if not SEMVER.match(version):
        problems.append(f"loadlens_app/version.py: __version__ {version!r} is not a semantic version")
    if not CHANGELOG.is_file():
        return problems + ["CHANGELOG.md is missing"]
    text = CHANGELOG.read_text(encoding="utf-8")
    names = [name for name, _date, _s, _e in _sections(text)]
    if "Unreleased" not in names:
        problems.append("CHANGELOG.md: no '## [Unreleased]' section")
    if version not in names:
        problems.append(f"CHANGELOG.md: no '## [{version}]' section for the version in loadlens_app/version.py")
    else:
        body = changelog_section(version, text) or ""
        if not body.strip():
            problems.append(f"CHANGELOG.md: the [{version}] section is empty")
    versions = [name for name in names if name != "Unreleased"]
    if versions and versions[0] != version:
        problems.append(f"CHANGELOG.md: the newest section is [{versions[0]}], loadlens_app/version.py says {version}")
    if tag:
        tag_version = tag[1:] if tag.startswith("v") else tag
        if tag_version != version:
            problems.append(f"tag {tag} does not match __version__ {version}")
        for name, date, _s, _e in _sections(text):
            if name == version and (not date or date == "unreleased"):
                problems.append(f"CHANGELOG.md: [{version}] has no release date; run tools/release.py --version {version}")
    return problems


def bump_version(new: str) -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    old = read_version()
    VERSION_FILE.write_text(VERSION_LINE.sub(f'__version__ = "{new}"', text, count=1), encoding="utf-8", newline="\n")
    return old


def _link_block(text: str, version: str, previous: str | None, source_url: str) -> str:
    """Reference links at the end of the file: Unreleased compares against the new tag, the new tag gets a link."""
    links = text[_links_start(text):]
    lines = [line for line in links.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("[Unreleased]:") and not line.startswith(f"[{version}]:")]
    base = source_url or read_source_url()
    new_lines = [f"[Unreleased]: {base}/compare/v{version}...HEAD"]
    if previous:
        new_lines.append(f"[{version}]: {base}/compare/v{previous}...v{version}")
    else:
        new_lines.append(f"[{version}]: {base}/releases/tag/v{version}")
    return "\n".join(new_lines + lines) + "\n"


def release_changelog(version: str, today: str) -> None:
    """``## [Unreleased]`` entries become ``## [version] - today``; an undated section of that version is dated."""
    text = CHANGELOG.read_text(encoding="utf-8")
    sections = _sections(text)
    by_name = {name: (date, start, end) for name, date, start, end in sections}
    if "Unreleased" not in by_name:
        raise SystemExit("CHANGELOG.md: no '## [Unreleased]' section")
    _u_date, u_start, u_end = by_name["Unreleased"]
    unreleased_body = _body(text, u_start, u_end)
    existing_body = ""
    if version in by_name:
        _v_date, v_start, v_end = by_name[version]
        existing_body = _body(text, v_start, v_end)
        # Drop the old section; its text is merged into the new one below.
        text = text[:v_start] + text[v_end:]
        sections = _sections(text)
        by_name = {name: (date, start, end) for name, date, start, end in sections}
        _u_date, u_start, u_end = by_name["Unreleased"]
    merged = "\n\n".join(part for part in (unreleased_body, existing_body) if part.strip())
    if not merged.strip():
        raise SystemExit(f"CHANGELOG.md: nothing to release: [Unreleased] and [{version}] are empty")
    previous = next((name for name, date, _s, _e in sections if name not in ("Unreleased", version) and date and date != "unreleased"), None)
    head = text[:u_start]
    tail = text[u_end:_links_start(text)].rstrip("\n")
    new_section = f"## [Unreleased]\n\n## [{version}] - {today}\n\n{merged.strip()}\n"
    body = head + new_section + ("\n" + tail + "\n" if tail.strip() else "") + "\n"
    CHANGELOG.write_text(body + _link_block(text, version, previous, read_source_url()), encoding="utf-8", newline="\n")


def refresh_third_party_table() -> str:
    """Regenerates the Python dependency table of THIRD_PARTY_NOTICES.md when tools/third_party_table.py exists."""
    if not NOTICES.is_file():
        return "THIRD_PARTY_NOTICES.md not found: table not refreshed"
    if not TABLE_TOOL.is_file():
        return "tools/third_party_table.py not found: table not refreshed"
    # Loaded by path: tools/ is not a package and -I keeps the script directory off sys.path.
    spec = importlib.util.spec_from_file_location("third_party_table_for_release", TABLE_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    update = getattr(module, "update_notices", None)
    if callable(update):
        changed = update(NOTICES)
        return "THIRD_PARTY_NOTICES.md: Python dependency table " + ("refreshed" if changed else "already current")
    text = NOTICES.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        return f"THIRD_PARTY_NOTICES.md has no {TABLE_START} / {TABLE_END} markers: table not refreshed"
    rows = module.python_dependency_rows()
    render = getattr(module, "render_table", None)
    table = str(render(rows)).strip("\n") if callable(render) else _render_rows(rows)
    start = text.index(TABLE_START) + len(TABLE_START)
    end = text.index(TABLE_END)
    NOTICES.write_text(text[:start] + "\n" + table + "\n" + text[end:], encoding="utf-8", newline="\n")
    return "THIRD_PARTY_NOTICES.md: Python dependency table refreshed"


def _render_rows(rows) -> str:
    """A markdown table from the rows of python_dependency_rows(): dicts or sequences, the first row sets the columns."""
    rows = list(rows)
    if not rows:
        return "_no Python dependencies_"
    if isinstance(rows[0], dict):
        header = list(rows[0].keys())
        cells = [[str(row.get(key, "")) for key in header] for row in rows]
    else:
        header = ["Package", "Version", "License", "Homepage"][: len(rows[0])]
        cells = [[str(value) for value in row] for row in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in cells]
    return "\n".join(lines)


def check_license_headers() -> str:
    if not HEADERS_TOOL.is_file():
        return "tools/license_headers.py not found: headers not checked"
    result = subprocess.run([sys.executable, str(HEADERS_TOOL), "--check"], cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("license headers check failed:\n" + (result.stdout + result.stderr).strip())
    return "license headers: ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", metavar="X.Y.Z", help="prepare the release of this version")
    group.add_argument("--check", action="store_true", help="verify that version.py and CHANGELOG.md agree")
    group.add_argument("--notes", metavar="X.Y.Z", help="print the CHANGELOG section of a version")
    parser.add_argument("--tag", help="with --check: the git tag that must name the same version (vX.Y.Z)")
    parser.add_argument("--date", help="release date for --version (default: today, UTC)")
    args = parser.parse_args(argv)

    if args.check:
        problems = check(args.tag)
        for problem in problems:
            print("ERROR:", problem)
        if not problems:
            print(f"ok: version {read_version()} is in CHANGELOG.md")
        return 1 if problems else 0

    if args.notes:
        body = changelog_section(args.notes)
        if body is None:
            print(f"CHANGELOG.md has no section for {args.notes}", file=sys.stderr)
            return 1
        print(body)
        return 0

    version = args.version
    if not SEMVER.match(version):
        parser.error(f"{version!r} is not a semantic version (X.Y.Z)")
    today = args.date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    old = bump_version(version)
    print(f"loadlens_app/version.py: {old} -> {version}")
    release_changelog(version, today)
    print(f"CHANGELOG.md: [Unreleased] -> [{version}] - {today}")
    print(refresh_third_party_table())
    print(check_license_headers())
    problems = check()
    for problem in problems:
        print("ERROR:", problem)
    if problems:
        return 1
    print(
        "\nNext steps (by hand):\n"
        "  git diff                                   # review the changes\n"
        f"  git commit -am \"Release {version}\"\n"
        f"  git tag -a v{version} -m \"LoadLens {version}\"\n"
        f"  git push origin main v{version}            # the tag starts .github/workflows/release.yml"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
