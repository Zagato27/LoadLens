# SPDX-License-Identifier: Apache-2.0
"""README and community files of the open-source release (plans/open-source-plan.md, sections 5 and 8).

Offline: reads the Markdown files, the demo compose file and its env file. README.md has no Cyrillic outside
the link to the Russian README; relative links of the community files point at existing files (or at the
English docs that are written in parallel); the demo command of the README names the real demo files and
credentials; README.ru.md links back to README.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
README = ROOT_DIR / "README.md"
README_RU = ROOT_DIR / "README.ru.md"
DEMO_COMPOSE = "docker/docker-compose.demo.yml"
DEMO_ENV = "docker/demo.env"
DEMO_COMMAND = f"docker compose -f {DEMO_COMPOSE} --env-file {DEMO_ENV} --profile demo up --build"

CYRILLIC = re.compile("[\u0400-\u04ff]")
# The only Cyrillic README.md may contain: the text of the link to README.ru.md.
README_CYRILLIC_ALLOWLIST = ("Русская версия",)

# Files whose relative links are checked.
LINKED_FILES = (
    "README.md",
    "README.ru.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
)
# English user documentation written by the docs task of the release: a link to one of these is valid even
# before the file lands in the tree (plans/open-source-plan.md, section 5).
AGREED_DOCS = frozenset({
    "docs/README.md",
    "docs/getting-started.md",
    "docs/configuration.md",
    "docs/data-sources.md",
    "docs/ai-analysis.md",
    "docs/running-load-tests.md",
    "docs/api.md",
    "docs/deployment.md",
    "docs/development.md",
})

MARKDOWN_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MARKDOWN_IMAGE = re.compile(r"\!\[[^\]]*\]\(([^)\s]+)\)")
HTML_REF = re.compile(r"""(?:href|src)=["']([^"']+)["']""")
FENCE = re.compile(r"```.*?```", re.S)


def _read(rel: str) -> str:
    return (ROOT_DIR / rel).read_text(encoding="utf-8")


def _relative_targets(rel: str) -> list[str]:
    """Every link and image target of a Markdown file that is neither absolute nor an in-page anchor."""
    text = FENCE.sub("", _read(rel))
    targets: list[str] = []
    for pattern in (MARKDOWN_LINK, MARKDOWN_IMAGE, HTML_REF):
        targets.extend(pattern.findall(text))
    out = []
    for target in targets:
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#") or target.startswith("//"):
            continue
        out.append(target.split("#", 1)[0])
    return [t for t in out if t]


# ---- README.md -----------------------------------------------------------------------------------------------


def test_readme_is_english():
    text = README.read_text(encoding="utf-8")
    for allowed in README_CYRILLIC_ALLOWLIST:
        text = text.replace(allowed, "")
    leftovers = sorted({line.strip() for line in text.splitlines() if CYRILLIC.search(line)})
    assert not leftovers, "README.md is the English README; Russian text belongs in README.ru.md:\n" + "\n".join(leftovers)


def test_readme_links_the_russian_readme_and_stays_short():
    text = README.read_text(encoding="utf-8")
    assert "README.ru.md" in text, "README.md must link README.ru.md"
    assert len(text.splitlines()) <= 300, "README.md is the front page: keep it short and link docs/ for depth"
    for heading in ("## Why LoadLens", "## Quick start", "## How it works", "## Configuration", "## Development",
                    "## Contributing", "## Security", "## License", "## Community"):
        assert heading in text, f"README.md lost the section {heading!r}"


def test_readme_demo_command_matches_the_demo_files():
    text = README.read_text(encoding="utf-8")
    assert DEMO_COMMAND in text, f"README.md must show the demo command verbatim: {DEMO_COMMAND}"
    assert DEMO_COMMAND.replace("up --build", "down -v") in text, "README.md must show how to stop the demo"
    assert (ROOT_DIR / DEMO_COMPOSE).is_file() and (ROOT_DIR / DEMO_ENV).is_file()
    compose = _read(DEMO_COMPOSE)
    assert re.search(r'^\s*profiles:\s*\["demo"\]', compose, re.M), "the demo services sit behind the demo profile"
    env = dict(
        line.split("=", 1) for line in _read(DEMO_ENV).splitlines() if line and not line.startswith("#") and "=" in line
    )
    user, password = env["DEMO_ADMIN_USER"].strip(), env["DEMO_ADMIN_PASSWORD"].strip()
    assert f"`{user}` / `{password}`" in text, "README.md must show the demo credentials of docker/demo.env"
    port = env.get("APP_PORT", "5000").strip()
    assert f"http://localhost:{port}" in text


# ---- Links of the community files ----------------------------------------------------------------------------


@pytest.mark.parametrize("rel", LINKED_FILES)
def test_relative_links_point_at_existing_files(rel):
    assert (ROOT_DIR / rel).is_file(), f"{rel} is missing"
    base = (ROOT_DIR / rel).parent
    missing = []
    for target in _relative_targets(rel):
        path = (base / target).resolve()
        try:
            repo_rel = path.relative_to(ROOT_DIR).as_posix()
        except ValueError:
            missing.append(f"{target} (outside the repository)")
            continue
        if path.exists() or repo_rel in AGREED_DOCS:
            continue
        missing.append(target)
    assert not missing, (
        f"{rel}: links to files that do not exist: {', '.join(sorted(set(missing)))}. "
        "Only the English docs of the release plan may be linked before they exist: " + ", ".join(sorted(AGREED_DOCS))
    )


# ---- README.ru.md, CONTRIBUTING.md, SECURITY.md, templates ---------------------------------------------------


def test_russian_readme_links_back_and_keeps_its_content():
    text = README_RU.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:25])
    assert "[README.md](README.md)" in head, "README.ru.md must link README.md near the top"
    assert "## Лицензия" in text and "Apache License 2.0" in text
    assert "t.me/pbekker" in text, "the Telegram contact stays in the Russian README"
    assert "## Быстрый старт" in text and "## REST API" in text, "README.ru.md keeps the full Russian documentation"
    assert "\r" not in text and not text.startswith("\ufeff")


def test_contributing_and_security_cover_the_agreed_points():
    contributing = _read("CONTRIBUTING.md")
    for needle in ("requirements-dev.txt", "python -m pytest -q", "LOADLENS_REQUIRE_NODE=1", "locales/", "i18n_extract.py",
                   "license_headers.py --check", "git commit -s", "Signed-off-by", "tools/release.py", "CHANGELOG.md"):
        assert needle in contributing, f"CONTRIBUTING.md must mention {needle!r}"
    security = _read("SECURITY.md")
    for needle in ("private vulnerability reporting", "7 days", "Supported versions", "settings.py", "docker/.env",
                   "phone home", "LLM provider"):
        assert needle in security, f"SECURITY.md must mention {needle!r}"
    conduct = _read("CODE_OF_CONDUCT.md")
    assert "Contributor Covenant" in conduct and "version 2.1" in conduct
    assert "[INSERT CONTACT METHOD]" not in conduct, "the contact line of the Code of Conduct must be filled in"


def test_issue_and_pull_request_templates_exist():
    templates = ROOT_DIR / ".github" / "ISSUE_TEMPLATE"
    for name in ("bug_report.yml", "feature_request.yml", "question.yml", "config.yml"):
        assert (templates / name).is_file(), f".github/ISSUE_TEMPLATE/{name} is missing"
    yaml = pytest.importorskip("yaml")
    for name in ("bug_report.yml", "feature_request.yml", "question.yml"):
        form = yaml.safe_load((templates / name).read_text(encoding="utf-8"))
        assert form["name"] and form["description"] and isinstance(form["body"], list) and form["body"], name
        assert all("type" in block for block in form["body"]), name
    config = yaml.safe_load((templates / "config.yml").read_text(encoding="utf-8"))
    assert config["blank_issues_enabled"] is False
    assert any("discussions" in link["url"] for link in config["contact_links"]), "config.yml links Discussions"
    bug = (templates / "bug_report.yml").read_text(encoding="utf-8")
    assert "/about" in bug, "the bug form asks for the version from /about"
    pr = _read(".github/PULL_REQUEST_TEMPLATE.md")
    for needle in ("pytest", "locales/ru", "SPDX", "CHANGELOG.md", "git commit -s"):
        assert needle in pr, f"PULL_REQUEST_TEMPLATE.md must mention {needle!r}"
