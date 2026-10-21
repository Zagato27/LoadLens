# SPDX-License-Identifier: Apache-2.0
"""The About endpoint and section (plans/open-source-plan.md, sections 3 and 8).

``GET /about`` is public, answers without the storage database and carries no configuration value; the settings
page shows the "About" section in ru and en; the sidebar footer shows the version; ``tools/release.py --check``
and CHANGELOG.md agree with ``loadlens_app/version.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from loadlens_app import core  # noqa: E402
from loadlens_app import version  # noqa: E402
from loadlens_app.auth import InMemoryRepository, Role, policy, web  # noqa: E402
from loadlens_app.blueprints import about as about_module  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text  # noqa: E402
from tests.test_auth import Browser, add_user  # noqa: E402

EXPECTED_KEYS = {"name", "version", "build", "license", "source_url", "docs_url"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Authentication off (the conftest default): a plain test client."""
    import update_page
    from app import create_app

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def auth_app(monkeypatch):
    """Authentication on, as in tests/test_auth.py."""
    from app import create_app

    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    monkeypatch.setenv("LOADLENS_SECRET_KEY", "k" * 32)
    repo = InMemoryRepository()
    flask_app = create_app(auth_repository=repo)
    flask_app.config["TESTING"] = True
    return flask_app


def _leaf_strings(node, path: str, out: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(key path, value) of every string leaf of a configuration tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            _leaf_strings(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _leaf_strings(value, f"{path}[{index}]", out)
    elif isinstance(node, str):
        out.append((path, node))
    return out


# ---- GET /about ------------------------------------------------------------------------------------------------


def test_about_has_the_documented_shape_and_matches_version_py(client):
    resp = client.get("/about")
    assert resp.status_code == 200 and resp.mimetype == "application/json"
    data = resp.get_json()
    assert set(data) == EXPECTED_KEYS
    assert data["name"] == "LoadLens"
    assert data["version"] == version.__version__
    assert data["build"] == version.BUILD
    assert data["license"] == "Apache-2.0"
    assert data["source_url"] == version.SOURCE_URL and data["source_url"].startswith("https://github.com/")
    assert data["docs_url"].startswith(version.SOURCE_URL)
    assert all(isinstance(value, str) for value in data.values())


def test_version_is_semver_and_version_info_has_no_settings(monkeypatch):
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version.__version__)
    info = version.version_info()
    assert set(info) == {"version", "build", "python", "source_url"}
    assert info["version"] == version.__version__ and info["source_url"] == version.SOURCE_URL
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", info["python"])
    source = (ROOT_DIR / "loadlens_app" / "version.py").read_text(encoding="utf-8")
    imports = re.findall(r"^(?:from|import)\s+([\w.]+)", source, re.M)
    assert not [name for name in imports if name.startswith(("settings", "loadlens_app", "AI", "flask"))], imports
    assert re.search(r'^__version__ = "[^"]+"$', source, re.M), "tools/release.py rewrites this one line"


def test_build_comes_from_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("LOADLENS_BUILD", "2026-10-19-0320474")
    module = importlib.reload(version)
    try:
        assert module.BUILD == "2026-10-19-0320474" and module.version_info()["build"] == "2026-10-19-0320474"
    finally:
        monkeypatch.delenv("LOADLENS_BUILD")
        importlib.reload(version)


def test_about_carries_no_configuration_value(client):
    """Every string of CONFIG (hosts, tokens, names) is absent from the response; only the constants are public."""
    text = client.get("/about").get_data(as_text=True)
    expected = about_module.about_payload()
    # The constants themselves and their words ("github", "main", "README"): a configuration value equal to one of
    # them (a branch name, a provider id) is not a leak. Shorter strings ("kind", "name") are generic tokens.
    public = set(expected) | {str(value) for value in expected.values()}
    public |= {word for value in expected.values() for word in re.split(r"[^A-Za-z0-9]+", str(value)) if word}
    leaves = _leaf_strings(json.loads(json.dumps(core.CONFIG, default=str)), "", [])
    leaked = sorted({path for path, value in leaves if len(value) >= 6 and value not in public and value in text})
    # The message names the keys only, never a configuration value.
    assert len(leaked) == 0, f"configuration values of {leaked} appear in /about"


def test_about_is_public_with_and_without_a_session(auth_app):
    service = auth_app.extensions[web.EXTENSION_KEY]
    anonymous = Browser(auth_app)
    resp = anonymous.get("/about")
    assert resp.status_code == 200 and resp.get_json()["version"] == version.__version__
    assert "Cache-Control" not in resp.headers  # public, like /healthz
    add_user(service, "viewer-about", Role.VIEWER)
    viewer = Browser(auth_app)
    assert viewer.login("viewer-about").status_code == 200
    assert viewer.get("/about").get_json()["version"] == version.__version__
    assert anonymous.get("/about/third-party").status_code in (200, 404)


def test_about_rules_are_public_and_recovery_safe():
    for endpoint in ("about.about", "about.third_party"):
        rule = policy.rule_for(endpoint)
        assert rule.role is None and rule.recovery_ok and not rule.page, endpoint


def test_about_answers_while_the_storage_database_is_down(auth_app, monkeypatch):
    """The user store cannot be reached: data endpoints answer 503, the constants of /about still answer."""
    from AI.storage_state import StorageUnavailableError

    def store_is_down(*_args, **_kwargs):
        raise StorageUnavailableError("connection refused")

    monkeypatch.setattr(web, "_resolve_auth", store_is_down)
    anonymous = Browser(auth_app)
    assert anonymous.get("/runs").status_code == 503
    assert anonymous.get("/about").status_code == 200
    assert anonymous.get("/about").get_json()["version"] == version.__version__


# ---- /about/third-party ----------------------------------------------------------------------------------------


def test_third_party_notices_are_served_as_text(client, monkeypatch, tmp_path):
    notices = tmp_path / "THIRD_PARTY_NOTICES.md"
    notices.write_text("# Third-party notices\n\n| Package | License |\n| --- | --- |\n| Flask | BSD-3-Clause |\n", encoding="utf-8")
    monkeypatch.setattr(about_module, "THIRD_PARTY_NOTICES_PATH", notices)
    resp = client.get("/about/third-party")
    assert resp.status_code == 200 and resp.mimetype == "text/plain"
    assert resp.get_data(as_text=True).startswith("# Third-party notices")


def test_third_party_notices_missing_is_a_localized_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(about_module, "THIRD_PARTY_NOTICES_PATH", tmp_path / "missing.md")
    resp = client.get("/about/third-party")
    assert resp.status_code == 404 and resp.get_json() == {"error": "Файл THIRD_PARTY_NOTICES.md не найден в этой установке."}
    use_locale(client, "en")
    assert client.get("/about/third-party").get_json()["error"] == "THIRD_PARTY_NOTICES.md is not part of this installation."


def test_repository_third_party_notices_file_is_served():
    if not about_module.THIRD_PARTY_NOTICES_PATH.is_file():
        pytest.skip("THIRD_PARTY_NOTICES.md is not written yet")
    text = about_module.THIRD_PARTY_NOTICES_PATH.read_text(encoding="utf-8")
    assert "Flask" in text, "the notices name the web framework"


# ---- settings page: the About section --------------------------------------------------------------------------


def _about_section(html: str) -> str:
    start = html.index('<section id="sec-about"')
    return html[start:html.index("</section>", start)]


def test_settings_page_has_the_about_section_in_ru(client):
    html = client.get("/settings").get_data(as_text=True)
    assert '<button type="button" data-section="about">О программе</button>' in html
    assert html.index('data-section="appearance"') < html.index('data-section="about"') < html.index('data-section="users"')
    section = _about_section(html)
    assert '<h3 id="h-about">О программе</h3>' in section
    assert f'<strong id="aboutVersion">LoadLens {version.__version__}</strong>' in section
    assert "сборка не указана" in section or version.BUILD in section
    assert f'href="{version.SOURCE_URL}/blob/main/LICENSE"' in section and "Apache License 2.0" in section
    assert f'href="{version.SOURCE_URL}/blob/main/NOTICE"' in section
    assert f'href="{version.SOURCE_URL}/issues"' in section and f'href="{version.SOURCE_URL}/discussions"' in section
    assert f'href="{version.SOURCE_URL}"' in section
    assert 'id="aboutThirdPartyBody" data-url="/about/third-party"' in section
    assert "Сторонние компоненты" in section
    # The vendored markdown renderer of the report page is loaded for the notices.
    assert "vendor/purify.min.js" in html and "vendor/marked.min.js" in html


def test_settings_page_about_section_in_en_has_no_russian_text(client):
    use_locale(client, "en")
    html = client.get("/settings").get_data(as_text=True)
    assert '<button type="button" data-section="about">About</button>' in html
    section = _about_section(html)
    assert_no_cyrillic(visible_text(section))
    for phrase in ("<h3 id=\"h-about\">About</h3>", "Version", "License", "Source code", "Third-party notices",
                   "Report a bug:", "GitHub Issues", "GitHub Discussions"):
        assert phrase in section, phrase


def test_settings_page_sends_the_about_dictionary(client):
    html = client.get("/settings").get_data(as_text=True)
    body = re.search(r"window\.LoadLens\.i18n=(\{.*?\});</script>", html, re.S).group(1)
    messages = json.loads(body)["messages"]
    assert messages["about.third_party.missing"].startswith("Файл THIRD_PARTY_NOTICES.md")
    assert "about.third_party.failed" in messages


def test_about_dictionaries_are_complete_and_english_is_english():
    ru = json.loads((ROOT_DIR / "locales" / "ru" / "about.json").read_text(encoding="utf-8"))
    en = json.loads((ROOT_DIR / "locales" / "en" / "about.json").read_text(encoding="utf-8"))
    assert set(ru) == set(en) and all(key.startswith("about.") for key in ru)
    assert not [key for key, value in en.items() if CYRILLIC.search(value)]


# ---- version in the top bar ------------------------------------------------------------------------------------


def test_sidebar_footer_shows_the_version_for_every_role(auth_app):
    service = auth_app.extensions[web.EXTENSION_KEY]
    add_user(service, "admin-about", Role.ADMIN)
    add_user(service, "viewer-about", Role.VIEWER)
    admin, viewer = Browser(auth_app), Browser(auth_app)
    assert admin.login("admin-about").status_code == 200 and viewer.login("viewer-about").status_code == 200
    admin_html = admin.get("/").get_data(as_text=True)
    # Top bar (plans/instrument-ui-plan.md 2.3): the wordmark, then the bare version in mono.
    assert f'<a class="app-version" id="appVersion" href="/settings#about" title="О программе: версия, лицензия, сторонние компоненты">{version.__version__}</a>' in admin_html
    viewer_html = viewer.get("/").get_data(as_text=True)
    assert f'<span class="app-version" id="appVersion" title="О программе: версия, лицензия, сторонние компоненты">{version.__version__}</span>' in viewer_html
    assert "/settings#about" not in viewer_html  # the settings page is admin-only
    use_locale(viewer.client, "en")
    assert 'title="About LoadLens: version, license, third-party notices"' in viewer.get("/").get_data(as_text=True)


def test_sidebar_footer_without_authentication(client):
    html = client.get("/").get_data(as_text=True)
    assert f'href="/settings#about" title="О программе: версия, лицензия, сторонние компоненты">{version.__version__}</a>' in html


# ---- release tooling -------------------------------------------------------------------------------------------


def test_changelog_has_the_current_version():
    text = (ROOT_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert re.search(rf"^## \[{re.escape(version.__version__)}\]", text, re.M), "add the version to CHANGELOG.md"
    assert "keepachangelog.com" in text


def test_release_check_passes():
    result = subprocess.run([sys.executable, "-I", str(ROOT_DIR / "tools" / "release.py"), "--check"],
                            cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert version.__version__ in result.stdout


def test_release_check_rejects_a_mismatched_tag():
    result = subprocess.run([sys.executable, "-I", str(ROOT_DIR / "tools" / "release.py"), "--check", "--tag", "v0.0.1-nope"],
                            cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert result.returncode == 1 and "does not match" in result.stdout


def test_release_notes_print_the_section_of_a_version():
    result = subprocess.run([sys.executable, "-I", str(ROOT_DIR / "tools" / "release.py"), "--notes", version.__version__],
                            cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert result.returncode == 0 and "### Added" in result.stdout


def test_release_version_bump_moves_unreleased_entries(tmp_path, monkeypatch):
    """A dry run on copies: version.py gets the new version, the changelog a dated section with the entries."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_tool_under_test", ROOT_DIR / "tools" / "release.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "1.0.0"\nSOURCE_URL = "https://example.com/owner/loadlens"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- A thing.\n\n## [1.0.0] - 2026-10-20\n\n### Added\n\n- First.\n\n"
        "[Unreleased]: https://example.com/owner/loadlens/compare/v1.0.0...HEAD\n"
        "[1.0.0]: https://example.com/owner/loadlens/releases/tag/v1.0.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tool, "VERSION_FILE", version_file)
    monkeypatch.setattr(tool, "CHANGELOG", changelog)
    monkeypatch.setattr(tool, "NOTICES", tmp_path / "THIRD_PARTY_NOTICES.md")
    monkeypatch.setattr(tool, "HEADERS_TOOL", tmp_path / "license_headers.py")
    assert tool.main(["--version", "1.1.0", "--date", "2026-11-01"]) == 0
    assert '__version__ = "1.1.0"' in version_file.read_text(encoding="utf-8")
    text = changelog.read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [1.1.0] - 2026-11-01\n\n### Fixed\n\n- A thing.\n\n## [1.0.0] - 2026-10-20" in text
    assert "[Unreleased]: https://example.com/owner/loadlens/compare/v1.1.0...HEAD" in text
    assert "[1.1.0]: https://example.com/owner/loadlens/compare/v1.0.0...v1.1.0" in text
    assert tool.check() == []
    assert tool.changelog_section("1.1.0", text) == "### Fixed\n\n- A thing."


def test_release_workflow_builds_and_publishes_on_tags():
    text = (ROOT_DIR / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert 'tags: ["v*"]' in text and "packages: write" in text and "contents: write" in text
    assert "LOADLENS_REQUIRE_NODE" in text and "LOADLENS_BUILD=" in text and "ghcr.io/" in text
    assert "tools/release.py --notes" in text and "gh release create" in text
    audit = (ROOT_DIR / ".github" / "workflows" / "dependency-audit.yml").read_text(encoding="utf-8")
    assert "pip-licenses" in audit and "THIRD_PARTY_NOTICES.md" in audit
    fail_on = re.search(r'--fail-on\s+"([^"]+)"', audit).group(1).split(";")
    assert "GNU General Public License v3 (GPLv3)" in fail_on and "AGPL-3.0-only" in fail_on
    assert not any("LGPL" in name or "Lesser" in name for name in fail_on), "LGPL (psycopg2, paramiko) is accepted"


def test_dockerfile_carries_the_version_and_license_labels():
    text = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG LOADLENS_BUILD" in text and "ENV LOADLENS_BUILD=$LOADLENS_BUILD" in text
    assert 'org.opencontainers.image.licenses="Apache-2.0"' in text
    assert f'org.opencontainers.image.source="{version.SOURCE_URL}"' in text
    assert "org.opencontainers.image.version=" in text and "org.opencontainers.image.title=" in text
    ignored = (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert not {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"} & set(line.strip() for line in ignored)
