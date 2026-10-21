# SPDX-License-Identifier: Apache-2.0
"""Table of the Python dependencies of LoadLens with their licenses, for THIRD_PARTY_NOTICES.md.

Reads the direct dependencies of requirements.txt and resolves the license and the homepage of each one from the
metadata of the distribution installed in the current interpreter (``importlib.metadata``); the version column is
the one pinned in requirements.txt. Packages that are not installed, and packages whose metadata only names a
license family ("BSD License"), fall back to ``KNOWN_LICENSES``. No network access, nothing is installed.

    python tools/third_party_table.py            # print the Markdown table
    python tools/third_party_table.py --write    # refresh the table between the markers of THIRD_PARTY_NOTICES.md
    python tools/third_party_table.py --check    # exit 1 when the table in THIRD_PARTY_NOTICES.md is stale

``python_dependency_rows()`` is reused by tools/release.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
START_MARK = "<!-- python-deps:start -->"
END_MARK = "<!-- python-deps:end -->"

# Precise license identifiers and homepages of the known dependencies. Package metadata often carries only a
# classifier ("BSD License") or a free-text field; these values win over such generic answers and cover packages
# that are not installed where the table is generated. Keys are normalized names (lower case, "-" for "_" and ".").
KNOWN_LICENSES: dict[str, tuple[str, str]] = {
    "flask": ("BSD-3-Clause", "https://flask.palletsprojects.com/"),
    "werkzeug": ("BSD-3-Clause", "https://werkzeug.palletsprojects.com/"),
    "gunicorn": ("MIT", "https://gunicorn.org/"),
    "requests": ("Apache-2.0", "https://requests.readthedocs.io/"),
    "urllib3": ("MIT", "https://urllib3.readthedocs.io/"),
    "psycopg2": ("LGPL-3.0-or-later WITH OpenSSL linking exception", "https://www.psycopg.org/"),
    "pandas": ("BSD-3-Clause", "https://pandas.pydata.org/"),
    "numpy": ("BSD-3-Clause", "https://numpy.org/"),
    "tabulate": ("MIT", "https://github.com/astanin/python-tabulate"),
    "matplotlib": ("Matplotlib License (PSF-based)", "https://matplotlib.org/"),
    "pillow": ("MIT-CMU", "https://python-pillow.github.io/"),
    "celery": ("BSD-3-Clause", "https://docs.celeryq.dev/"),
    "atlassian-python-api": ("Apache-2.0", "https://github.com/atlassian-api/atlassian-python-api"),
    "beautifulsoup4": ("MIT", "https://www.crummy.com/software/BeautifulSoup/"),
    "pydantic": ("MIT", "https://docs.pydantic.dev/"),
    "pypdf": ("BSD-3-Clause", "https://pypdf.readthedocs.io/"),
    "python-docx": ("MIT", "https://github.com/python-openxml/python-docx"),
    "langchain-core": ("MIT", "https://github.com/langchain-ai/langchain"),
    "langchain-gigachat": ("MIT", "https://github.com/ai-forever/langchain-gigachat"),
    "paramiko": ("LGPL-2.1-or-later", "https://www.paramiko.org/"),
    "pyyaml": ("MIT", "https://pyyaml.org/"),
    "jinja2": ("BSD-3-Clause", "https://jinja.palletsprojects.com/"),
    "idna": ("BSD-3-Clause", "https://github.com/kjd/idna"),
    "certifi": ("MPL-2.0", "https://github.com/certifi/python-certifi"),
    "h11": ("MIT", "https://github.com/python-hyper/h11"),
}

# Trove classifiers -> SPDX-like identifiers. Families without a clause count stay generic and are refined by
# KNOWN_LICENSES when possible.
CLASSIFIERS = {
    "MIT License": "MIT",
    "BSD License": "BSD",
    "Apache Software License": "Apache-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "ISC License (ISCL)": "ISC",
    "GNU Library or Lesser General Public License (LGPL)": "LGPL",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.1-or-later",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
}
GENERIC = {"", "BSD", "LGPL", "PSF-2.0", "UNKNOWN", "Dual License", "OSI Approved"}
HOMEPAGE_KEYS = ("homepage", "home-page", "home", "source", "repository", "source code", "code", "documentation")
_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[^\]]*\])?\s*(?P<spec>[<>=!~].*)?$")


class Requirement(NamedTuple):
    name: str
    extras: str
    spec: str


class Row(NamedTuple):
    name: str
    version: str
    license: str
    homepage: str
    source: str  # "metadata", "metadata+table", "table" or "unknown"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path: Path = REQUIREMENTS) -> list[Requirement]:
    """Direct dependencies of requirements.txt in file order (comments, blank lines and options skipped)."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT.match(line)
        if not match:
            raise ValueError(f"{path.name}: cannot parse requirement {raw!r}")
        rows.append(Requirement(match.group("name"), match.group("extras") or "", (match.group("spec") or "").strip()))
    return rows


def _declared_version(spec: str) -> str:
    """``==1.2`` -> ``1.2``; any other specifier is shown as written (``>=3.1.6``)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and parts[0].startswith("==") and not parts[0].startswith("==="):
        return parts[0][2:].strip()
    return spec or "(unpinned)"


def _license_from_metadata(meta) -> str:
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License :: "):
            label = classifier.split(" :: ")[-1]
            if label in CLASSIFIERS:
                return CLASSIFIERS[label]
    field = (meta.get("License") or "").strip()
    first = field.splitlines()[0].strip() if field else ""
    if 0 < len(first) <= 40:  # a short identifier such as "MIT" or "Apache 2.0", not a pasted license text
        return first
    return ""


def _homepage_from_metadata(meta) -> str:
    urls = {}
    for item in meta.get_all("Project-URL") or []:
        label, _, url = item.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in HOMEPAGE_KEYS:
        if urls.get(key):
            return urls[key]
    return (meta.get("Home-page") or "").strip()


def python_dependency_rows(requirements_path: Path = REQUIREMENTS) -> list[Row]:
    """One row per direct dependency: name, declared version, license, homepage and where the answer came from."""
    rows = []
    for req in read_requirements(requirements_path):
        key = normalize(req.name)
        known_license, known_home = KNOWN_LICENSES.get(key, ("", ""))
        try:
            meta = metadata.distribution(req.name).metadata
        except metadata.PackageNotFoundError:
            meta = None
        if meta is None:
            license_id, homepage = known_license, known_home
            source = "table" if known_license else "unknown"
        else:
            license_id = _license_from_metadata(meta)
            homepage = known_home or _homepage_from_metadata(meta)
            source = "metadata"
            if known_license and (license_id in GENERIC or license_id != known_license):
                license_id = known_license
                source = "metadata+table"
        rows.append(Row(req.name, _declared_version(req.spec), license_id or "unknown", homepage, source))
    return rows


def render_table(rows: Iterable[Row]) -> str:
    lines = ["| Package | Version | License | Homepage |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row.name} | {row.version} | {row.license} | {row.homepage} |")
    return "\n".join(lines) + "\n"


def _split_notices(text: str) -> tuple[str, str, str]:
    start = text.index(START_MARK) + len(START_MARK)
    end = text.index(END_MARK)
    return text[:start], text[start:end], text[end:]


def current_table(path: Path = NOTICES) -> str:
    """The table between the markers of THIRD_PARTY_NOTICES.md (without the surrounding blank lines)."""
    return _split_notices(path.read_text(encoding="utf-8"))[1].strip("\n") + "\n"


def notices_are_current(path: Path = NOTICES, requirements_path: Path = REQUIREMENTS) -> bool:
    return current_table(path) == render_table(python_dependency_rows(requirements_path))


def update_notices(path: Path = NOTICES, requirements_path: Path = REQUIREMENTS) -> bool:
    """Replaces the table between the markers; returns True when the file changed."""
    text = path.read_text(encoding="utf-8")
    head, _old, tail = _split_notices(text)
    new = head + "\n\n" + render_table(python_dependency_rows(requirements_path)) + "\n" + tail
    if new == text:
        return False
    path.write_bytes(new.encode("utf-8"))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="refresh the table in THIRD_PARTY_NOTICES.md")
    group.add_argument("--check", action="store_true", help="exit 1 when THIRD_PARTY_NOTICES.md is stale")
    parser.add_argument("--verbose", action="store_true", help="also print where each answer came from")
    args = parser.parse_args(argv)
    rows = python_dependency_rows()
    if args.check:
        if notices_are_current():
            print(f"{NOTICES.name}: python dependency table is current")
            return 0
        print(f"{NOTICES.name}: python dependency table is stale; run: python tools/third_party_table.py --write")
        return 1
    if args.write:
        changed = update_notices()
        print(f"{NOTICES.name}: {'updated' if changed else 'already current'}")
        return 0
    sys.stdout.write(render_table(rows))
    if args.verbose:
        for row in rows:
            print(f"  {row.name}: {row.source}")
    unknown = [row.name for row in rows if row.license == "unknown" or not row.homepage]
    if unknown:
        print(f"incomplete rows (add them to KNOWN_LICENSES): {', '.join(unknown)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
