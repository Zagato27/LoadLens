# SPDX-License-Identifier: Apache-2.0
"""License files of the open-source release (plans/open-source-plan.md, sections 2 and 8).

Offline: reads LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, TRADEMARK.md, requirements.txt and the vendor
directory; imports tools/license_headers.py and tools/third_party_table.py by path.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR = ROOT_DIR / "static" / "vendor"
NOTICES = ROOT_DIR / "THIRD_PARTY_NOTICES.md"

# TEMPORARY: files that were being edited by other release tasks while the SPDX headers were added, so the
# header tool skipped them. Add the headers (python tools/license_headers.py --write) and empty this set.
# Files that may lack the SPDX header. Empty: every in-scope file carries it; add a path here only
# temporarily, while a generated or third-party file is being sorted out.
HEADER_ALLOWLIST: frozenset[str] = frozenset()


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


license_headers = _load("license_headers")
third_party_table = _load("third_party_table")


def _allowlisted(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in HEADER_ALLOWLIST)


# ---- LICENSE, NOTICE, TRADEMARK --------------------------------------------------------------------------------


def test_license_is_the_full_apache_2_0_text():
    text = (ROOT_DIR / "LICENSE").read_text(encoding="utf-8")
    head = [line.strip() for line in text.splitlines()[:4]]
    assert head[1] == "Apache License" and head[2] == "Version 2.0, January 2004", head
    assert head[3] == "http://www.apache.org/licenses/"
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text
    assert "END OF TERMS AND CONDITIONS" in text
    assert "APPENDIX: How to apply the Apache License to your work." in text
    assert "Copyright [yyyy] [name of copyright owner]" in text, "the appendix must stay the unfilled template"
    assert text.endswith("limitations under the License.\n")
    assert "\r" not in text and not text.startswith("﻿")


def test_notice_names_the_project_and_the_copyright_holder():
    text = (ROOT_DIR / "NOTICE").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "LoadLens"
    assert re.fullmatch(r"Copyright 2025-\d{4} Pavel Bekker", lines[1]), lines[1]
    assert "THIRD_PARTY_NOTICES.md" in text
    assert "DOMPurify" in text, "the bundled Apache-2.0 component is named in NOTICE"


def test_trademark_policy_exists():
    text = (ROOT_DIR / "TRADEMARK.md").read_text(encoding="utf-8")
    assert "LoadLens" in text and "logo" in text
    assert "not" in text and "Apache License" in text, "the policy says the name and logo are outside the license"
    assert "based on LoadLens" in text


# ---- THIRD_PARTY_NOTICES.md ------------------------------------------------------------------------------------


def _vendor_patterns(text: str) -> list[str]:
    return re.findall(r"static/vendor/[^\s|`)\]]+", text)


def test_every_vendor_file_is_listed_in_third_party_notices():
    text = NOTICES.read_text(encoding="utf-8")
    patterns = _vendor_patterns(text)
    assert patterns
    files = [p.relative_to(ROOT_DIR).as_posix() for p in VENDOR.rglob("*") if p.is_file()]
    assert files, "static/vendor is empty"
    missing = [
        f for f in files
        if f != "static/vendor/fonts/OFL.txt" and not any(fnmatch.fnmatchcase(f, pattern) for pattern in patterns)
    ]
    assert not missing, f"vendor files without a notice: {missing}"


def test_bundled_components_have_their_license_texts():
    text = NOTICES.read_text(encoding="utf-8")
    assert "Chart.js" in text and "marked" in text and "DOMPurify" in text and "Ace" in text
    assert text.count("Permission is hereby granted, free of charge, to any person obtaining a copy") == 2, "two MIT texts"
    assert "Redistribution and use in source and binary forms" in text, "BSD-3-Clause text of Ace"
    assert "SIL OPEN FONT LICENSE Version 1.1" in text
    assert "Apache License, Version 2.0" in text and "MPL-2.0" in text, "DOMPurify dual license and the chosen option"
    assert "python:3.12-slim" in text and "timescale/timescaledb" in text and "redis:7.4" in text


def test_every_requirement_is_in_third_party_notices():
    text = NOTICES.read_text(encoding="utf-8").lower()
    names = [req.name for req in third_party_table.read_requirements(ROOT_DIR / "requirements.txt")]
    assert names
    missing = [name for name in names if f"| {name.lower()} |" not in text]
    assert not missing, f"requirements without a row in THIRD_PARTY_NOTICES.md: {missing}"


def test_python_dependency_table_is_generated_and_current():
    rows = third_party_table.python_dependency_rows(ROOT_DIR / "requirements.txt")
    assert all(row.license != "unknown" and row.homepage for row in rows), [r for r in rows if r.license == "unknown" or not r.homepage]
    assert third_party_table.notices_are_current(NOTICES, ROOT_DIR / "requirements.txt"), (
        "run: python tools/third_party_table.py --write"
    )


def test_third_party_table_parses_requirement_lines(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("# comment\nFlask==3.1.3  # pinned\ncelery[redis]==5.4.0\nJinja2>=3.1.6\n-r other.txt\n", encoding="utf-8")
    rows = third_party_table.read_requirements(req)
    assert [(r.name, r.extras, r.spec) for r in rows] == [
        ("Flask", "", "==3.1.3"), ("celery", "[redis]", "==5.4.0"), ("Jinja2", "", ">=3.1.6"),
    ]
    table = third_party_table.render_table(third_party_table.python_dependency_rows(req))
    assert table.splitlines()[0] == "| Package | Version | License | Homepage |"
    assert "| Flask | 3.1.3 | BSD-3-Clause |" in table and "| Jinja2 | >=3.1.6 |" in table


def test_vendored_fonts_are_golos_text_and_jetbrains_mono():
    fonts = sorted(p.name for p in (VENDOR / "fonts").glob("*.woff2"))
    expected = sorted(
        f"{family}-{subset}-{weight}-normal.woff2"
        for family in ("golos-text", "jetbrains-mono") for subset in ("cyrillic", "latin") for weight in (400, 500, 600)
    )
    assert fonts == expected
    for name in fonts:
        assert (VENDOR / "fonts" / name).read_bytes()[:4] == b"wOF2", name
    css = (ROOT_DIR / "static" / "css" / "app.css").read_text(encoding="utf-8")
    for name in fonts:
        assert f"url('/static/vendor/fonts/{name}')" in css, name
    assert "'Inter'" not in css and "Montserrat" not in css


def test_ofl_is_next_to_the_fonts():
    text = (VENDOR / "fonts" / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE Version 1.1" in text
    assert "The Golos Text Project Authors" in text and "The JetBrains Mono Project Authors" in text
    assert "Inter" not in text and "Montserrat" not in text, "the removed fonts are not named any more"
    assert "PERMISSION & CONDITIONS" in text and text.rstrip().endswith("OTHER DEALINGS IN THE FONT SOFTWARE.")


# ---- SPDX headers ----------------------------------------------------------------------------------------------


def test_in_scope_files_carry_the_spdx_header():
    files = license_headers.in_scope_files(ROOT_DIR)
    assert "update_page.py" in files and "static/js/common.js" in files and "tests/test_license_files.py" in files
    assert not [f for f in files if f.startswith(("static/vendor/", "locales/", "tests/fixtures/"))]
    assert "settings.py" not in files and "metrics_config.py" not in files, "ignored secret files are out of scope"
    missing = [f for f in license_headers.missing_headers(ROOT_DIR, files) if not _allowlisted(f)]
    assert not missing, "files without the SPDX header (run: python tools/license_headers.py --write):\n" + "\n".join(missing)


@pytest.mark.parametrize(
    ("name", "source", "expected"),
    [
        ("plain.py", b'"""Doc."""\nfrom __future__ import annotations\n', b'# SPDX-License-Identifier: Apache-2.0\n"""Doc."""\nfrom __future__ import annotations\n'),
        ("crlf.py", b"import os\r\nprint(os)\r\n", b"# SPDX-License-Identifier: Apache-2.0\r\nimport os\r\nprint(os)\r\n"),
        ("shebang.py", b"#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\"\"\"Doc.\"\"\"\n", b"#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n# SPDX-License-Identifier: Apache-2.0\n\"\"\"Doc.\"\"\"\n"),
        ("bom.py", b"\xef\xbb\xbfx = 1\n", b"\xef\xbb\xbf# SPDX-License-Identifier: Apache-2.0\nx = 1\n"),
        ("empty.py", b"", b"# SPDX-License-Identifier: Apache-2.0\n"),
        ("page.js", b"// Page logic\n(function () {})();\n", b"// SPDX-License-Identifier: Apache-2.0\n// Page logic\n(function () {})();\n"),
    ],
)
def test_with_header_keeps_shebang_encoding_bom_and_line_endings(name, source, expected):
    kind = "js" if name.endswith(".js") else "py"
    assert license_headers.with_header(source, kind) == expected
    assert license_headers.with_header(expected, kind) == expected, "idempotent"
    assert license_headers.has_header(expected, kind) and not license_headers.has_header(source, kind)


def test_header_tool_check_and_write_on_a_sandbox(tmp_path):
    (tmp_path / "static" / "js").mkdir(parents=True)
    (tmp_path / "static" / "vendor").mkdir()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_bytes(b"x = 1\n")
    (tmp_path / "settings.py").write_bytes(b"SECRET = 1\n")
    (tmp_path / "static" / "js" / "a.js").write_bytes(b"var a = 1;\r\n")
    (tmp_path / "static" / "vendor" / "lib.js").write_bytes(b"var lib = 1;\n")
    assert license_headers.in_scope_files(tmp_path) == ["pkg/mod.py", "static/js/a.js"], "no git: walk with the secret files skipped"
    assert license_headers.missing_headers(tmp_path) == ["pkg/mod.py", "static/js/a.js"]
    assert license_headers.write_headers(tmp_path) == ["pkg/mod.py", "static/js/a.js"]
    assert license_headers.missing_headers(tmp_path) == []
    assert (tmp_path / "static" / "js" / "a.js").read_bytes() == b"// SPDX-License-Identifier: Apache-2.0\r\nvar a = 1;\r\n"
    assert (tmp_path / "settings.py").read_bytes() == b"SECRET = 1\n"
    assert (tmp_path / "static" / "vendor" / "lib.js").read_bytes() == b"var lib = 1;\n"


# ---- repository hygiene ----------------------------------------------------------------------------------------


def test_foreign_material_and_root_data_dumps_are_gone():
    assert not (ROOT_DIR / "G-Eval.pdf").exists(), "the G-Eval paper is cited in AI/README.md instead"
    assert not (ROOT_DIR / "body.json").exists()
    assert "arxiv.org/abs/2303.16634" in (ROOT_DIR / "AI" / "README.md").read_text(encoding="utf-8")
    if shutil.which("git") is None or not (ROOT_DIR / ".git").exists():
        pytest.skip("git is not available")
    # Grafana dashboard dumps at the root (k6-load-testing-results*.json) can never be committed.
    ignored = subprocess.run(
        ["git", "-C", str(ROOT_DIR), "check-ignore", "-q", "k6-load-testing-results.json"], capture_output=True,
    )
    assert ignored.returncode == 0, ".gitignore no longer covers k6-load-testing-results*.json"


def test_gitignore_covers_the_secret_and_local_files():
    text = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    rules = {line.strip() for line in text if line.strip() and not line.startswith("#")}
    for required in ("settings.py", "settings_runtime*.json", "metrics_config.py", "metrics_config_runtime.json",
                     ".loadlens_secret_key", "**/.env", "AI/config.py", "k6-load-testing-results*.json"):
        assert required in rules, f".gitignore lost the rule {required!r}"
    assert "!settings.example.py" in rules and "!metrics_config.example.py" in rules
    assert not any("sber" in rule for rule in rules), "no company names in the ignore rules"
