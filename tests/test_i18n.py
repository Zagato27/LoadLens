# SPDX-License-Identifier: Apache-2.0
"""Localization infrastructure (plans/i18n-plan.md, stage 1): dictionaries, the engine (AI/i18n.py) and the Flask layer
(loadlens_app/i18n.py).

Dictionary rules: locales/<lang>/<ns>.json, canonical JSON, the same base keys in ru and en, complete plural families,
matching {placeholders}, no empty values, no markup, no Cyrillic in en outside locales/allowlist.json. Every literal key
the code passes to t/tn/LL.t/LL.tn/LL.tHtml/msg/plural exists in ru; dynamic keys use a prefix listed below.
"""

from __future__ import annotations

import ast
import contextvars
import json
import pickle
import re
import subprocess
import sys
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import i18n  # noqa: E402
from loadlens_app import i18n as web_i18n  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text  # noqa: E402

LOCALES = ROOT_DIR / "locales"
LANGS = ("ru", "en")
PLURAL_FORMS = {"ru": {"one", "few", "many"}, "en": {"one", "other"}}
PLURAL_SUFFIXES = {"zero", "one", "two", "few", "many", "other"}
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
KEY = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
# Keys built at run time (t(`common.verdict.${code}`), t('common.lang.' ~ code)): the prefix must be listed here.
DYNAMIC_PREFIXES = ("common.verdict.", "common.lang.", "common.role.", "common.domain.", "common.test_type.", "glossary.",
                    "report.metric.", "report.sla_check.", "packs.role.", "packs.unit.", "packs.category.",
                    "report.focus.", "report.peak_label.", "report.hold_note.", "report.spike_note.", "report.failure_mode.",
                    "report.saturation_note.", "sla.check_title.", "sla.rationale.", "sla.summary.",
                    "capacity.verdict.", "capacity.confidence.", "capacity.ceiling_source.", "capacity.scaling.",
                    "report.step_label.")
# Definitions and docs of the helpers themselves are not usages.
ENGINE_FILES = {"AI/i18n.py", "loadlens_app/i18n.py", "static/js/i18n.js"}
TRANSLATE_CALLS = {"t", "tn", "tHtml", "msg", "plural", "translate", "LocalizedError"}
NBSP = chr(0xA0)


def _canonical(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _namespaces(lang: str) -> dict[str, dict[str, str]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted((LOCALES / lang).glob("*.json"))}


def _messages(lang: str) -> dict[str, str]:
    return {key: value for names in _namespaces(lang).values() for key, value in names.items()}


def _split_plural(key: str) -> tuple[str, str | None]:
    head, _, last = key.rpartition(".")
    return (head, last) if last in PLURAL_SUFFIXES else (key, None)


def _families(messages: dict[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for key in messages:
        base, suffix = _split_plural(key)
        if suffix:
            out.setdefault(base, set()).add(suffix)
    return out


def _base_keys(messages: dict[str, str]) -> set[str]:
    return {_split_plural(key)[0] for key in messages}


def _allowlist() -> dict[str, set[str]]:
    data = json.loads((LOCALES / "allowlist.json").read_text(encoding="utf-8"))
    return {path: {entry["literal"] for entry in entries} for path, entries in data.items()}


# ---- dictionaries ----------------------------------------------------------------------------------------------


def test_dictionary_files_are_canonical_utf8_with_lf():
    files = [path for lang in LANGS for path in sorted((LOCALES / lang).glob("*.json"))]
    assert files, "no dictionaries under locales/"
    for path in files:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{path}: BOM"
        assert b"\r" not in raw, f"{path}: CRLF"
        obj = json.loads(raw.decode("utf-8"))
        assert raw == _canonical(obj), f"{path}: not canonical, dump it with json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + newline"


def test_both_languages_have_the_same_namespace_files():
    assert sorted(_namespaces("ru")) == sorted(_namespaces("en"))
    assert {"common"} <= set(_namespaces("ru"))


def test_keys_are_snake_case_and_inside_their_namespace():
    for lang in LANGS:
        for namespace, messages in _namespaces(lang).items():
            for key, value in messages.items():
                assert KEY.match(key), f"{lang}/{namespace}.json: bad key {key!r}"
                assert key.startswith(namespace + "."), f"{lang}/{namespace}.json: {key!r} is outside the namespace"
                assert isinstance(value, str), f"{lang}: {key} is not a string"


def test_ru_and_en_have_the_same_base_keys():
    ru, en = _messages("ru"), _messages("en")
    only_ru = sorted(_base_keys(ru) - _base_keys(en))
    only_en = sorted(_base_keys(en) - _base_keys(ru))
    assert not only_ru and not only_en, f"only in ru: {only_ru}; only in en: {only_en}"


def test_plural_families_are_complete_per_language():
    for lang in LANGS:
        messages = _messages(lang)
        for base, forms in _families(messages).items():
            assert forms == PLURAL_FORMS[lang], f"{lang}: {base} has forms {sorted(forms)}, needs {sorted(PLURAL_FORMS[lang])}"
            assert base not in messages, f"{lang}: {base} is both a message and a plural family"


def test_placeholders_match_between_languages():
    def placeholders(messages: dict[str, str]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for key, value in messages.items():
            out.setdefault(_split_plural(key)[0], set()).update(PLACEHOLDER.findall(value))
        return out

    ru, en = placeholders(_messages("ru")), placeholders(_messages("en"))
    for base in sorted(set(ru) & set(en)):
        assert ru[base] == en[base], f"{base}: ru {sorted(ru[base])} vs en {sorted(en[base])}"


def test_values_are_not_empty_and_carry_no_markup():
    for lang in LANGS:
        for key, value in _messages(lang).items():
            assert value.strip(), f"{lang}: {key} is empty"
            assert "<" not in value and ">" not in value, f"{lang}: {key} holds markup; pass it as a parameter"


def test_en_values_have_no_cyrillic_outside_the_allowlist():
    allow = _allowlist()
    for namespace, messages in _namespaces("en").items():
        allowed = allow.get(f"locales/en/{namespace}.json", set()) | allow.get("*", set())
        for key, value in messages.items():
            if value in allowed:
                continue
            assert not CYRILLIC.search(value), f"en: {key} = {value!r}"


def test_language_names_are_not_translated():
    ru, en = _messages("ru"), _messages("en")
    names = {key: value for key, value in ru.items() if key.startswith("common.lang.")}
    assert names == {"common.lang.ru": "Русский", "common.lang.en": "English"}
    assert {key: en[key] for key in names} == names


def test_verdict_labels_match_ai_verdicts():
    verdicts = pytest.importorskip("AI.verdicts")
    labels = getattr(verdicts, "LABELS", None)
    if not labels:
        pytest.skip("AI/verdicts.py has no LABELS yet")
    for lang in LANGS:
        messages = _messages(lang)
        assert {code: messages[f"common.verdict.{code}"] for code in labels[lang]} == labels[lang]


# ---- keys used by the code ---------------------------------------------------------------------------------------


def _extractor():
    import importlib.util

    spec = importlib.util.spec_from_file_location("i18n_extract_for_tests", ROOT_DIR / "tools" / "i18n_extract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_JS_OR_JINJA_CALL = re.compile(r"(?<![\w.$])(?:LL\.)?(t|tn|tHtml|msg|plural)\(\s*(['\"`])")


def _quoted_usages(text: str) -> list[tuple[str, str, bool]]:
    """(function, literal, dynamic) for calls in JS or Jinja text whose first argument is a quoted literal."""
    out = []
    for match in _JS_OR_JINJA_CALL.finditer(text):
        quote = match.group(2)
        end = text.find(quote, match.end())
        if end < 0:
            continue
        literal = text[match.end():end]
        tail = text[end + 1:end + 40].lstrip()
        dynamic = "${" in literal or literal.endswith(".") or tail.startswith(("+", "~"))
        if "${" in literal:
            literal = literal.split("${", 1)[0]
        out.append((match.group(1), literal, dynamic))
    return out


def _python_usages(source: str) -> list[tuple[str, str, bool]]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name not in TRANSLATE_CALLS:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            out.append((name, first.value, first.value.endswith(".")))
        elif isinstance(first, ast.JoinedStr) and first.values and isinstance(first.values[0], ast.Constant):
            out.append((name, str(first.values[0].value), True))
        elif isinstance(first, ast.BinOp) and isinstance(first.left, ast.Constant) and isinstance(first.left.value, str):
            out.append((name, first.left.value, True))
    return out


def _all_usages() -> list[tuple[str, str, str, bool]]:
    extract = _extractor()
    usages = []
    for path in extract.scanned_files():
        if path in ENGINE_FILES:
            continue
        source = (ROOT_DIR / path).read_text(encoding="utf-8")
        if path.endswith(".py"):
            found = _python_usages(source)
        elif path.endswith(".js"):
            found = _quoted_usages(extract.strip_js_comments(source))
        else:
            found = _quoted_usages(re.sub(r"\{#.*?#\}", "", source, flags=re.S))
        usages += [(path, name, literal, dynamic) for name, literal, dynamic in found]
    return usages


def test_every_literal_key_used_by_the_code_exists_in_ru():
    ru = _messages("ru")
    families = _families(ru)
    problems = []
    for path, name, literal, dynamic in _all_usages():
        if dynamic:
            if not literal.startswith(DYNAMIC_PREFIXES):
                problems.append(f"{path}: {name}({literal!r}...) builds a key outside DYNAMIC_PREFIXES")
            continue
        if not KEY.match(literal):
            problems.append(f"{path}: {name}({literal!r}) is not a dictionary key")
        elif name in ("tn", "plural"):
            if literal not in families:
                problems.append(f"{path}: plural family {literal!r} is missing in ru")
        elif literal not in ru:
            problems.append(f"{path}: {literal!r} is missing in ru")
    assert not problems, "\n".join(problems)


def test_dynamic_prefixes_have_keys():
    ru = _messages("ru")
    for prefix in DYNAMIC_PREFIXES:
        assert any(key.startswith(prefix) for key in ru), f"{prefix} has no keys: drop it from DYNAMIC_PREFIXES"


def test_usage_scanner_sees_the_call_forms():
    js = "LL.t('a.b'); LL.tn('a.c', n); LL.tHtml(`a.d`); t(`common.verdict.${code}`); LL.t('common.lang.' + x); obj.t('x.y');"
    assert _quoted_usages(js) == [
        ("t", "a.b", False), ("tn", "a.c", False), ("tHtml", "a.d", False),
        ("t", "common.verdict.", True), ("t", "common.lang.", True),
    ]
    assert _quoted_usages("{{ t('nav.title') }} {{ tn('x.pods', 3) }} {{ t('common.lang.' ~ code) }}") == [
        ("t", "nav.title", False), ("tn", "x.pods", False), ("t", "common.lang.", True),
    ]
    py = "msg('job.done', n=1)\nplural('x.pods', 3)\ni18n.t(f'common.verdict.{code}')\nraise LocalizedError('api.bad')\nprint('t(x)')\n"
    assert _python_usages(py) == [
        ("msg", "job.done", False), ("plural", "x.pods", False), ("t", "common.verdict.", True), ("LocalizedError", "api.bad", False),
    ]


def test_every_declared_i18n_namespace_has_a_dictionary():
    declared = re.compile(r"\{%-?\s*set\s+i18n_ns\s*=\s*\[(.*?)\]\s*-?%\}", re.S)
    for page in sorted((ROOT_DIR / "templates").glob("*.html")):
        for match in declared.finditer(page.read_text(encoding="utf-8")):
            for name in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
                for lang in LANGS:
                    assert (LOCALES / lang / f"{name}.json").is_file(), f"{page.name}: i18n_ns {name!r} has no locales/{lang}/{name}.json"


# ---- engine ----------------------------------------------------------------------------------------------------


@pytest.fixture
def catalogs(tmp_path, monkeypatch):
    """Temporary dictionaries: ``write(lang, namespace, {key: value})``; the engine reads only these."""
    root = tmp_path / "locales"
    for lang in LANGS:
        (root / lang).mkdir(parents=True)

    def write(lang, namespace, values, raw=None):
        path = root / lang / f"{namespace}.json"
        path.write_bytes(raw if raw is not None else _canonical(values))
        i18n.reload_catalogs()
        return path

    monkeypatch.setattr(i18n, "LOCALES_DIR", root)
    write("ru", "test", {
        "test.greeting": "Привет, {name}!",
        "test.only_ru": "Только по-русски",
        "test.wrapper": "Итог: {inner}",
        "test.pods.one": "{n} под",
        "test.pods.few": "{n} пода",
        "test.pods.many": "{n} подов",
    })
    write("en", "test", {
        "test.greeting": "Hello, {name}!",
        "test.wrapper": "Result: {inner}",
        "test.pods.one": "{n} pod",
        "test.pods.other": "{n} pods",
    })
    return write


@pytest.mark.parametrize(
    "n, ru, en",
    [
        (0, "many", "other"), (1, "one", "one"), (2, "few", "other"), (4, "few", "other"), (5, "many", "other"),
        (11, "many", "other"), (12, "many", "other"), (14, "many", "other"), (21, "one", "other"), (22, "few", "other"),
        (25, "many", "other"), (101, "one", "other"), (111, "many", "other"), (112, "many", "other"), (1.5, "few", "other"),
        (1.0, "one", "one"), (-1, "one", "one"), (-22, "few", "other"),
    ],
)
def test_plural_category(n, ru, en):
    assert i18n.plural_category(n, "ru") == ru
    assert i18n.plural_category(n, "en") == en


def test_plural_picks_the_form_and_passes_n(catalogs):
    assert [i18n.plural("test.pods", n) for n in (1, 2, 5, 21, 1.5)] == ["1 под", "2 пода", "5 подов", "21 под", "1,5 пода"]
    assert [i18n.plural("test.pods", n, locale="en") for n in (0, 1, 2, 1.5)] == ["0 pods", "1 pod", "2 pods", "1.5 pods"]
    assert i18n.plural("test.pods", 3, n="три") == "три пода"


def test_t_interpolates_and_keeps_unknown_placeholders(catalogs):
    assert i18n.t("test.greeting", name="Анна") == "Привет, Анна!"
    assert i18n.t("test.greeting", locale="en", name="Ann") == "Hello, Ann!"
    assert i18n.t("test.greeting") == "Привет, {name}!"  # a missing parameter does not crash
    assert i18n.t("test.greeting", name=None) == "Привет, !"
    assert i18n.interpolate("{0} {a.b} {x} {}", {"x": 1}) == "{0} {a.b} 1 {}"  # only {identifier} fields


def test_missing_key_falls_back_to_ru_then_to_the_key_and_warns_once(catalogs, monkeypatch, caplog):
    monkeypatch.delenv(i18n.STRICT_ENV, raising=False)
    with caplog.at_level("WARNING", logger="AI.i18n"):
        assert i18n.t("test.only_ru", locale="en") == "Только по-русски"
        assert i18n.t("test.only_ru", locale="en") == "Только по-русски"
        assert i18n.t("test.nowhere") == "test.nowhere"
        assert i18n.plural("test.nothing", 2, locale="en") == "test.nothing"
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 3, warnings
    assert any("test.only_ru" in text and "'en'" in text for text in warnings)


def test_strict_mode_raises_on_a_missing_key(catalogs):
    # tests/conftest.py switches strict mode on for the whole suite.
    with pytest.raises(KeyError, match="test.only_ru"):
        i18n.t("test.only_ru", locale="en")
    with pytest.raises(KeyError):
        i18n.t("test.nowhere")
    with pytest.raises(KeyError):
        i18n.plural("test.nothing", 1)


def test_locale_order_explicit_then_scope_then_resolver_then_default(catalogs, monkeypatch):
    monkeypatch.setattr(i18n, "_resolver", None)
    assert i18n.get_locale() == "ru"
    i18n.set_locale_resolver(lambda: "en")
    assert i18n.get_locale() == "en"
    assert i18n.t("test.greeting", name="A") == "Hello, A!"
    with i18n.with_locale("ru"):
        assert i18n.get_locale() == "ru"
        assert i18n.t("test.greeting", locale="en", name="A") == "Hello, A!"
        with i18n.with_locale("EN-gb"):
            assert i18n.get_locale() == "en"
        assert i18n.get_locale() == "ru"
    with i18n.with_locale(None):  # a report without report_language is ru
        assert i18n.get_locale() == "ru"
    i18n.set_locale_resolver(lambda: 1 / 0)
    assert i18n.get_locale() == "ru"  # a broken resolver never breaks rendering
    i18n.set_locale_resolver(lambda: "de")
    assert i18n.get_locale() == "ru"


def test_with_locale_reaches_threads_through_copy_context(catalogs):
    seen = {}

    def work(name):
        seen[name] = i18n.get_locale()

    with i18n.with_locale("en"):
        plain = threading.Thread(target=work, args=("plain",))
        copied = threading.Thread(target=contextvars.copy_context().run, args=(work, "copied"))
        plain.start(), copied.start()
        plain.join(), copied.join()
    assert seen["copied"] == "en"
    assert seen["plain"] == i18n.get_locale()  # a bare thread starts without the scope


def test_normalize_lang():
    assert [i18n.normalize_lang(v) for v in ("ru", "EN", " en-GB ", "en_US", "de", "", None, 5)] == ["ru", "en", "en", "en", None, None, None, None]


def test_msg_ref_and_render_ref_round_trip(catalogs):
    message = i18n.msg("test.greeting", name="Анна")
    assert message == "Привет, Анна!" and isinstance(message, str)
    assert message.ref() == {"key": "test.greeting", "params": {"name": "Анна"}}
    assert json.dumps({"m": message}, ensure_ascii=False) == '{"m": "Привет, Анна!"}'
    assert i18n.render_ref(message.ref(), locale="en") == "Hello, Анна!"

    nested = i18n.msg("test.wrapper", inner=i18n.msg("test.greeting", name="Ann"))
    assert nested == "Итог: Привет, Ann!"
    ref = nested.ref()
    assert ref == {"key": "test.wrapper", "params": {"inner": {"key": "test.greeting", "params": {"name": "Ann"}}}}
    assert i18n.render_ref(json.loads(json.dumps(ref)), locale="en") == "Result: Hello, Ann!"
    with i18n.with_locale("en"):
        assert i18n.render_ref(ref) == "Result: Hello, Ann!"
        assert i18n.t("test.wrapper", inner=nested.params["inner"]) == "Result: Hello, Ann!"

    restored = pickle.loads(pickle.dumps(nested))
    assert restored == nested and restored.key == "test.wrapper" and restored.ref() == ref


def test_render_ref_falls_back_for_broken_or_unknown_refs(catalogs, monkeypatch):
    monkeypatch.delenv(i18n.STRICT_ENV, raising=False)
    assert i18n.render_ref(None, "текст") == "текст"
    assert i18n.render_ref({"key": ""}, "текст") == "текст"
    assert i18n.render_ref({"key": "test.greeting", "extra": 1}, "текст") == "текст"
    assert i18n.render_ref({"key": "test.renamed", "params": {}}, "старый текст") == "старый текст"
    assert i18n.render_ref({"key": "test.pods"}, "x") == "x"  # a plural family needs n
    assert i18n.render_ref({"key": "test.pods", "params": {"n": 22}}) == "22 пода"
    assert i18n.render_ref({"key": "test.pods", "params": {"n": 1.5}}, locale="en") == "1.5 pods"
    assert i18n.render_ref({"key": "test.greeting"}) == "Привет, {name}!"


def test_localized_error_renders_in_the_current_locale(catalogs):
    error = i18n.LocalizedError("test.greeting", name="Ann")
    assert str(error) == "Привет, Ann!"
    with i18n.with_locale("en"):
        assert str(error) == "Hello, Ann!"
    assert error.key == "test.greeting" and error.params == {"name": "Ann"}
    assert error.ref() == {"key": "test.greeting", "params": {"name": "Ann"}}
    assert i18n.t("test.wrapper", locale="en", inner=error) == "Result: Hello, Ann!"

    class PackError(i18n.LocalizedError, ValueError):
        pass

    with pytest.raises(ValueError, match="Привет, Bob!"):
        raise PackError("test.greeting", name="Bob")
    restored = pickle.loads(pickle.dumps(i18n.LocalizedError("test.greeting", name="Eve")))
    assert str(restored) == "Привет, Eve!" and restored.params == {"name": "Eve"}


def test_fmt_number_and_fmt_date():
    assert i18n.fmt_number(1234.567) == "1234,6"
    assert i18n.fmt_number(1234.567, 2, locale="en") == "1234.57"
    assert i18n.fmt_number(3, 0) == "3"
    assert i18n.fmt_number(None) == "—" and i18n.fmt_number(float("nan")) == "—"
    assert i18n.fmt_number("n/a") == "n/a"
    moment = datetime(2026, 10, 18, 9, 5, tzinfo=timezone.utc)
    assert i18n.fmt_date(moment) == "18.10.2026"
    assert i18n.fmt_date(moment, locale="en") == "2026-10-18"
    assert i18n.fmt_date(date(2026, 1, 2)) == "02.01.2026"
    assert i18n.fmt_date("2026-10-18T09:05:00Z", locale="en") == "2026-10-18"
    assert i18n.fmt_date("вчера") == "вчера" and i18n.fmt_date(None) == ""
    with i18n.with_locale("en"):
        assert i18n.fmt_number(0.5) == "0.5"


@pytest.mark.parametrize(
    "raw, problem",
    [
        (b"{not json", "invalid JSON"),
        (b"\xef\xbb\xbf{}", "BOM"),
        (b'{"other.key": "x"}', "outside the namespace"),
        (b'{"test.a": "x", "test.a": "y"}', "duplicate key"),
        (b'{"test.a": 1}', "must be a string"),
        (b'["test.a"]', "one JSON object"),
        (b'{"test.a": "\xff"}', "not valid UTF-8"),
    ],
)
def test_a_broken_dictionary_fails_loudly(catalogs, raw, problem):
    catalogs("ru", "test", None, raw=raw)
    with pytest.raises(i18n.CatalogError, match=problem):
        i18n.t("test.greeting")


def test_reload_catalogs_picks_up_changes(catalogs):
    assert i18n.t("test.greeting", name="A") == "Привет, A!"
    catalogs("ru", "test", {"test.greeting": "Здравствуйте, {name}!"})
    assert i18n.t("test.greeting", name="A") == "Здравствуйте, A!"


def test_client_messages_fall_back_to_ru_per_key(catalogs):
    messages = i18n.client_messages("en", ["test", "missing_ns"])
    assert messages["test.greeting"] == "Hello, {name}!"
    assert messages["test.only_ru"] == "Только по-русски"
    assert i18n.client_messages("ru", None) == i18n.catalog("ru").messages


def test_ai_modules_import_neither_flask_nor_loadlens_app():
    offenders = []
    for path in sorted((ROOT_DIR / "AI").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            offenders += [f"{path.relative_to(ROOT_DIR).as_posix()}: {name}" for name in names if name.split(".")[0] in ("flask", "loadlens_app")]
    assert not offenders, offenders
    probe = "import sys; import AI.i18n; print(sorted(m for m in sys.modules if m.split('.')[0] in ('flask', 'loadlens_app')))"
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ---- Flask layer -----------------------------------------------------------------------------------------------


def _isolated_runtime(monkeypatch, tmp_path):
    import update_page

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")


@pytest.fixture
def app(monkeypatch, tmp_path):
    from app import create_app

    _isolated_runtime(monkeypatch, tmp_path)
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_app(monkeypatch, tmp_path):
    from app import create_app
    from loadlens_app.auth import InMemoryRepository

    _isolated_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    flask_app = create_app(auth_repository=InMemoryRepository())
    flask_app.config["TESTING"] = True
    return flask_app


def _html_lang(html: str) -> str:
    return re.search(r'<html lang="([^"]*)"', html).group(1)


def test_html_lang_follows_cookie_then_setting_then_ru(client, monkeypatch):
    assert _html_lang(client.get("/").get_data(as_text=True)) == "ru"
    use_locale(client, "en")
    assert _html_lang(client.get("/").get_data(as_text=True)) == "en"
    use_locale(client, "de")  # not in the white list
    assert _html_lang(client.get("/").get_data(as_text=True)) == "ru"
    use_locale(client, None)
    assert _html_lang(client.get("/", headers={"Accept-Language": "en-US,en;q=0.9"}).get_data(as_text=True)) == "ru"

    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    assert _html_lang(client.get("/").get_data(as_text=True)) == "en"
    use_locale(client, "ru")
    assert _html_lang(client.get("/").get_data(as_text=True)) == "ru"


def test_login_page_lang_follows_the_cookie(auth_app):
    client = auth_app.test_client()
    page = client.get("/login")
    assert page.status_code == 200 and _html_lang(page.get_data(as_text=True)) == "ru"
    use_locale(client, "en")
    assert _html_lang(client.get("/login").get_data(as_text=True)) == "en"
    assert "Cookie" in client.get("/login").headers.get("Vary", "")


def test_root_templates_render_the_locale():
    for page in ("base.html", "login.html", "recovery.html", "storage_unavailable.html"):
        source = (ROOT_DIR / "templates" / page).read_text(encoding="utf-8")
        assert '<html lang="{{ locale }}"' in source, page
    for page in ("base.html", "recovery.html"):
        source = (ROOT_DIR / "templates" / page).read_text(encoding="utf-8")
        script = source.index("{{ i18n_script(i18n_ns|default([])) }}")
        assert script < source.index("js/i18n.js") < source.index("js/common.js" if page == "base.html" else "js/recovery.js")


def test_inline_dictionary_follows_the_theme_script_and_carries_the_nonce(client):
    page = client.get("/")
    html = page.get_data(as_text=True)
    nonce = re.search(r"'nonce-([^']+)'", page.headers["Content-Security-Policy"]).group(1)
    scripts = re.findall(r'<script nonce="([^"]+)">(.*?)</script>', html, re.S)
    assert len(scripts) == 2 and {value for value, _ in scripts} == {nonce}
    assert "loadlens.theme" in scripts[0][1]
    body = scripts[1][1]
    prefix = "window.LoadLens=window.LoadLens||{};window.LoadLens.i18n="
    assert body.startswith(prefix) and body.endswith(";")
    data = json.loads(body[len(prefix):-1])
    assert data["locale"] == "ru" and data["intl"] == "ru-RU"
    assert data["messages"]["common.verdict.pass"] == "Успешно"
    assert "Успешно" in body  # Cyrillic as is, not \u escapes
    assert html.index("window.LoadLens.i18n") < html.index("js/i18n.js") < html.index("js/common.js")

    use_locale(client, "en")
    html = client.get("/").get_data(as_text=True)
    data = json.loads(re.search(r"window\.LoadLens\.i18n=(\{.*?\});</script>", html, re.S).group(1))
    assert data["locale"] == "en" and data["intl"] == "en-GB"
    assert data["messages"]["common.verdict.pass"] == "Passed"


def test_i18n_script_namespaces_fallback_and_escaping(app, catalogs):
    catalogs("ru", "common", {"common.tricky": "a<b>&'c" + chr(0x2028) + "d"})
    catalogs("ru", "page", {"page.title": "Страница", "page.only_ru": "Только ru"})
    catalogs("en", "page", {"page.title": "Page"})
    with app.test_request_context("/", headers={"Cookie": "loadlens.lang=en"}):
        markup = str(web_i18n.i18n_script(["page", "no_such_ns", "../etc", "page"]))
        payload = json.loads(re.search(r"i18n=(\{.*\});</script>$", markup).group(1))
        assert payload["messages"] == {
            "common.tricky": "a<b>&'c" + chr(0x2028) + "d",
            "page.only_ru": "Только ru",
            "page.title": "Page",
            # ru-only test keys of the catalogs fixture are not in the requested namespaces
        }
        for raw, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027"), (chr(0x2028), "\\u2028")):
            assert raw not in markup.split(">", 1)[1].rsplit("<", 1)[0], raw
            assert escaped in markup
        assert str(web_i18n.i18n_script(["page"])).split(">", 1)[1] == markup.split(">", 1)[1]


def test_vary_cookie_only_where_the_locale_was_read(client):
    assert "Cookie" in client.get("/").headers.get("Vary", "")
    static = client.get("/static/js/common.js")
    assert static.status_code == 200 and "Cookie" not in static.headers.get("Vary", "")
    logo = client.get("/assets/logo.png")
    assert logo.status_code == 200 and "Cookie" not in logo.headers.get("Vary", "")


def test_jinja_t_escapes_the_message_and_params(app, catalogs):
    from flask import render_template_string
    from markupsafe import Markup

    catalogs("ru", "page", {"page.hello": "Привет, {name} & {link}!", "page.items.one": "{n} пункт", "page.items.few": "{n} пункта", "page.items.many": "{n} пунктов"})
    catalogs("en", "page", {"page.hello": "Hello, {name} & {link}!", "page.items.one": "{n} item", "page.items.other": "{n} items"})
    with app.test_request_context("/"):
        html = render_template_string(
            "{{ t('page.hello', name=name, link=link) }}|{{ tn('page.items', 3) }}|{{ t('page.hello', name=m) }}",
            name="<b>Анна</b>",
            link=Markup('<a href="/x">x</a>'),
            m=i18n.msg("page.items.one"),
        )
    assert html == 'Привет, &lt;b&gt;Анна&lt;/b&gt; &amp; <a href="/x">x</a>!|3 пункта|Привет, {n} пункт &amp; {link}!'
    with app.test_request_context("/", headers={"Cookie": "loadlens.lang=en"}):
        assert render_template_string("{{ tn('page.items', 1) }} {{ t('page.hello', name='A') }}") == "1 item Hello, A &amp; {link}!"


def test_report_language_reads_the_appearance_setting(monkeypatch):
    assert web_i18n.report_language() == "ru"
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    assert web_i18n.report_language() == "en"
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "de")
    assert web_i18n.report_language() == "ru"
    monkeypatch.setitem(core.CONFIG, "appearance", None)
    assert web_i18n.report_language() == "ru"


def test_current_locale_outside_a_request_follows_the_engine(app, monkeypatch):
    assert web_i18n.current_locale() == "ru"
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    assert web_i18n.current_locale() == "en"  # the resolver of install(): the installation setting
    with i18n.with_locale("ru"):
        assert web_i18n.current_locale() == "ru"


def test_helpers_find_cyrillic_in_visible_text_and_attributes():
    html = '<html><head><title>Отчёт</title><script>var x = "скрипт";</script></head><body><p title="Подсказка">Text</p><img alt="Logo"></body></html>'
    text = visible_text(html)
    assert "Отчёт" in text and "Подсказка" in text and "скрипт" not in text
    with pytest.raises(AssertionError, match="Подсказка"):
        assert_no_cyrillic(text, allow=("Отчёт",))
    assert_no_cyrillic(text, allow=("Отчёт", "Подсказка"))
