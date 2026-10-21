# SPDX-License-Identifier: Apache-2.0
"""Verdict codes (AI/verdicts.py): the mapping of stored labels and free text, labels, tones, ranks, the SQL CASE,
the JS copy in static/js/common.js and a guard that keeps the Russian labels out of the rest of the code."""

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AI import verdicts  # noqa: E402
from AI.verdicts import (  # noqa: E402
    CODES,
    LABELS,
    LEGACY,
    RANK,
    TONES,
    code_sql,
    exact_code,
    label,
    normalize_report,
    optional_code,
    rank,
    rank_sql,
    to_code,
    tone,
)
from tests.js_harness import run_js  # noqa: E402

RU = {"pass": "Успешно", "risk": "Есть риски", "fail": "Провал", "no_data": "Недостаточно данных"}


def test_tables():
    assert CODES == ("pass", "risk", "fail", "no_data")
    assert LABELS["ru"] == RU
    assert LABELS["en"] == {"pass": "Passed", "risk": "Risks found", "fail": "Failed", "no_data": "Insufficient data"}
    assert LEGACY == {text: code for code, text in RU.items()}
    assert TONES == {"pass": "ok", "risk": "warn", "fail": "fail", "no_data": "na"}
    assert RANK == {"pass": 0, "risk": 1, "fail": 2, "no_data": 3}
    assert set(LABELS["ru"]) == set(LABELS["en"]) == set(TONES) == set(RANK) == set(CODES)


@pytest.mark.parametrize(
    ("value", "code"),
    [
        # codes and the labels stored before them; case, ё and spaces do not matter
        ("pass", "pass"), ("RISK", "risk"), (" fail ", "fail"), ("no_data", "no_data"),
        ("Успешно", "pass"), ("Есть риски", "risk"), ("Провал", "fail"), ("Недостаточно данных", "no_data"),
        ("успешно", "pass"), ("  есть   РИСКИ ", "risk"), ("ПРОВАЛ", "fail"), ("недостаточно  данных", "no_data"),
        ("Есть риски\n", "risk"),
        # the English labels and synonyms of the old normalizers
        ("Passed", "pass"), ("Risks found", "risk"), ("Failed", "fail"), ("Insufficient data", "no_data"),
        ("OK", "pass"), ("ok.", "pass"), ("okay", "pass"), ("green", "pass"), ("success", "pass"), ("Successful", "pass"),
        ("warning", "risk"), ("degraded", "risk"), ("at risk", "risk"), ("unstable", "risk"),
        ("critical", "fail"), ("FAILURE", "fail"), ("error", "fail"), ("red", "fail"), ("severe", "fail"),
        ("unknown", "no_data"), ("N/A", "no_data"), ("no data", "no_data"),
        # Russian free text
        ("Успех", "pass"), ("норма", "pass"), ("стабильно", "pass"), ("Есть риск", "risk"), ("Деградация", "risk"),
        ("есть замечания", "risk"), ("под вопросом", "risk"), ("Предупреждение", "risk"),
        ("Критично", "fail"), ("неудовлетворительно", "fail"), ("ошибка", "fail"),
        ("нет данных", "no_data"), ("Недостаточно метрик", "no_data"),
        # the settled disagreements of the old normalizers
        ("нестабильно", "risk"), ("risk of failure", "risk"), ("Риск провала", "risk"),
        # short English tokens match whole words only
        ("covered", "no_data"), ("token", "no_data"), ("bypass", "no_data"), ("predicted", "no_data"),
        # an order: no_data, then fail, then risk, then pass
        ("unknown, ok", "no_data"), ("fail, warning", "fail"), ("risk, but ok", "risk"),
        # «критерии» is not «критично»
        ("критерии выполнены", "no_data"),
        # nothing
        (None, "no_data"), ("", "no_data"), ("   ", "no_data"), ("что-то новое", "no_data"), (42, "no_data"),
    ],
)
def test_to_code_maps_labels_codes_and_free_text(value, code):
    assert to_code(value) == code
    assert to_code(to_code(value)) == to_code(value)  # idempotent


def test_to_code_round_trips_labels_of_every_language():
    for lang, labels in LABELS.items():
        for code in CODES:
            assert to_code(labels[code]) == code, (lang, code)
            assert to_code(label(code, lang)) == code
            assert to_code(code) == code


@pytest.mark.parametrize(
    ("value", "code"),
    [("pass", "pass"), ("Провал", "fail"), (" есть  риски ", "risk"), ("НЕДОСТАТОЧНО ДАННЫХ", "no_data"),
     ("passed", None), ("OK", None), ("Failed", None), ("нестабильно", None), ("", None), (None, None)],
)
def test_exact_code_takes_only_codes_and_old_labels(value, code):
    assert exact_code(value) == code


def test_optional_code_keeps_missing_values():
    assert optional_code(None) is None and optional_code("") is None and optional_code("  ") is None
    assert optional_code("Провал") == "fail" and optional_code("fail") == "fail"


def test_label_tone_rank():
    assert [label(code) for code in CODES] == list(RU.values())
    assert [label(code, "en") for code in CODES] == ["Passed", "Risks found", "Failed", "Insufficient data"]
    assert label("Провал", "en") == "Failed" and label(None) == "Недостаточно данных"
    assert label("risk", "de") == "Есть риски" and label("risk", "EN-gb") == "Risks found"  # unknown -> ru
    assert [tone(code) for code in CODES] == ["ok", "warn", "fail", "na"]
    assert tone("Есть риски") == "warn" and tone(None) == "na" and tone("что-то") == "na"
    assert [rank(code) for code in CODES] == [0, 1, 2, 3]
    assert sorted(["no_data", "Провал", "pass", "Есть риски"], key=rank) == ["pass", "Есть риски", "Провал", "no_data"]


def _sqlite_eval(expression_sql, value):
    with sqlite3.connect(":memory:") as conn:
        return conn.execute(f"SELECT {expression_sql} FROM (SELECT ? AS v)", (value,)).fetchone()[0]


def test_code_sql_and_rank_sql_cover_codes_and_legacy_labels():
    case = code_sql("v")
    ranks = rank_sql("v")
    for text in (*CODES, *LEGACY):
        assert f"WHEN '{text}' THEN" in case and f"WHEN '{text}' THEN" in ranks
    # Evaluated by a real SQL engine: every stored value maps like to_code; NULL and others are no_data.
    for value in (*CODES, *LEGACY, None, "", "что-то новое"):
        expected = exact_code(value) or "no_data"
        assert _sqlite_eval(case, value) == expected, value
        assert _sqlite_eval(ranks, value) == RANK[expected], value
    # Only constants are inlined; the expression is evaluated once (a simple CASE).
    assert case.startswith("(CASE v WHEN ") and case.endswith(" ELSE 'no_data' END)") and "%" not in case
    assert code_sql("COALESCE(mv.verdict, f.sla_verdict, f.verdict)").count("COALESCE") == 1


def test_normalize_report_maps_verdicts_and_keeps_texts():
    parsed = {"verdict": "Есть риски", "verdict_rationale": "Статус «Провал»: текст модели",
              "deterministic_sla": {"verdict": "Провал", "summary": "SLA verdict: Провал. Нарушено: p95_latency"}, "findings": []}
    out = normalize_report(parsed)
    assert out["verdict"] == "risk" and out["deterministic_sla"]["verdict"] == "fail"
    assert out["verdict_rationale"] == parsed["verdict_rationale"]
    assert out["deterministic_sla"]["summary"] == parsed["deterministic_sla"]["summary"]
    assert parsed["verdict"] == "Есть риски" and parsed["deterministic_sla"]["verdict"] == "Провал"  # a copy
    assert normalize_report({"findings": []}) == {"findings": []}  # no verdict is not added
    assert normalize_report(None) is None and normalize_report("{}") == "{}"
    assert normalize_report(normalize_report(parsed)) == out


# ---- the JS copy (static/js/common.js) ------------------------------------------------------------------

JS_INPUTS = [
    *CODES, *LEGACY, "PASS", " Fail ", "  есть   РИСКИ ", "недостаточно данных", "ПРОВАЛ", "Есть риски\n",
    "passed", "OK", "Failed", "нестабильно", "covered", "", "что-то", None,
]


def test_js_verdict_helpers_match_python():
    result = run_js(["static/js/common.js"], f"""(() => {{
      const inputs = {json.dumps(JS_INPUTS, ensure_ascii=False)};
      return {{
        codes: LL.VERDICTS,
        labels: LL.VERDICT_LABELS,
        code: inputs.map((v) => LL.verdictCode(v)),
        exact: inputs.map((v) => LL.verdictCode(v, null)),
        label: inputs.map((v) => LL.verdictLabel(v)),
        tone: inputs.map((v) => LL.verdictClass(v)),
        report: LL.VERDICTS.map((v) => LL.verdictReportClass(v)),
        term: LL.VERDICTS.map((v) => LL.verdictTermKey(v)),
        undef: [LL.verdictCode(undefined), LL.verdictLabel(undefined), LL.verdictClass(undefined)],
      }};
    }})()""")
    assert result["codes"] == list(CODES)
    assert result["labels"] == LABELS["ru"]
    expected = [exact_code(v) or "no_data" for v in JS_INPUTS]
    assert result["code"] == expected
    assert result["exact"] == [exact_code(v) for v in JS_INPUTS]
    assert result["label"] == [label(code) for code in expected]
    assert result["tone"] == [tone(code) for code in expected]
    assert result["report"] == ["report-verdict-success", "report-verdict-risk", "report-verdict-fail", "report-verdict-na"]
    assert result["term"] == ["verdict_success", "verdict_risk", "verdict_fail", "verdict_na"]
    assert result["undef"] == ["no_data", "Недостаточно данных", "na"]


def test_js_glossary_has_the_verdict_terms():
    terms = run_js(["static/js/common.js", "static/js/ui.js"], "LL.VERDICTS.map((v) => !!LL.GLOSSARY[LL.verdictTermKey(v)])")
    assert terms == [True, True, True, True]


def test_js_texts_of_stored_codes_match_python():
    """Readers build test-type names, focus, notes and failure modes from codes; JS and Python tables are equal."""
    from AI.saturation import FAILURE_MODE_TEXTS, SATURATION_NOTE_TEXTS
    from AI.spike import SPIKE_NOTE_TEXTS
    from AI.stability import HOLD_NOTE_TEXTS
    from AI.test_profile import TEST_TYPE_TEXTS

    files = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
    js = run_js(files, "({ labels: LL.TEST_TYPE_LABELS, stored: LL.report._internals.STORED_TEXTS })")
    assert js["labels"] == {key: texts["label"] for key, texts in TEST_TYPE_TEXTS["ru"].items()}
    stored = js["stored"]
    assert stored["typeFocus"] == {key: texts["focus"] for key, texts in TEST_TYPE_TEXTS["ru"].items()}
    assert stored["holdNotes"] == HOLD_NOTE_TEXTS["ru"]
    assert stored["spikeNotes"] == SPIKE_NOTE_TEXTS["ru"]
    assert stored["saturationNotes"] == SATURATION_NOTE_TEXTS["ru"]
    assert stored["failureModes"] == FAILURE_MODE_TEXTS["ru"]
    for table in (TEST_TYPE_TEXTS, HOLD_NOTE_TEXTS, SPIKE_NOTE_TEXTS, SATURATION_NOTE_TEXTS, FAILURE_MODE_TEXTS):
        assert set(table) == {"ru", "en"} and set(table["ru"]) == set(table["en"])


def test_dictionary_labels_equal_the_verdict_and_test_type_tables_when_present():
    """The ``common.verdict.<code>`` and ``common.test_type.<key>`` keys of the dictionaries hold the labels of
    AI/verdicts.py and AI/test_profile.py."""
    from AI.test_profile import TEST_TYPE_TEXTS

    for lang in ("ru", "en"):
        path = ROOT / "locales" / lang / "common.json"
        if not path.is_file():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        present = {code: entries[f"common.verdict.{code}"] for code in CODES if f"common.verdict.{code}" in entries}
        assert present == {code: LABELS[lang][code] for code in present}, lang
        types = {key: entries[f"common.test_type.{key}"] for key in TEST_TYPE_TEXTS[lang] if f"common.test_type.{key}" in entries}
        assert types == {key: TEST_TYPE_TEXTS[lang][key]["label"] for key in types}, lang


# ---- guard: the labels live in one place ------------------------------------------------------------------

# Where a Russian verdict label may stand as a whole quoted literal ('Провал', "Успешно", `Есть риски`).
LABEL_HOMES = {
    "AI/verdicts.py",  # the table
    "static/js/common.js",  # its JS copy
    "tools/stubs/llm_stub.py",  # an LLM stub answers like a model that follows the current prompts
}
# Literals that equal a label but are not a verdict: path -> {label: count}.
NOT_VERDICTS = {
    "confluence_manager/update_confluence_template.py": {"Успешно": 2},  # status of a page update
}
# The code of the service: Python packages, page scripts and templates, tools and the root modules.
# Not scanned: tests (fixtures of old reports), docs, prompts (they still ask the model for labels), vendor code.
CODE_ROOTS = ("AI", "loadlens_app", "confluence_manager", "data_collectors", "static/js", "templates", "tools")
SKIP_DIRS = {"__pycache__", "prompts", "vendor", "node_modules"}
LABEL_LITERAL = re.compile(r"""(['"`])(%s)\1""" % "|".join(re.escape(text) for text in RU.values()))


def _code_files():
    paths = [path for path in ROOT.glob("*.py") if path.is_file()]
    for root in CODE_ROOTS:
        paths.extend(path for path in (ROOT / root).rglob("*") if path.suffix in {".py", ".js", ".html"} and path.is_file())
    for path in paths:
        rel = path.relative_to(ROOT)
        if set(rel.parts[:-1]) & SKIP_DIRS:
            continue
        yield rel.as_posix(), path


def test_verdict_labels_live_only_in_the_verdict_tables():
    found = {}
    for rel, path in _code_files():
        if rel in LABEL_HOMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        counts = {}
        for match in LABEL_LITERAL.finditer(text):
            counts[match.group(2)] = counts.get(match.group(2), 0) + 1
        if counts and counts != NOT_VERDICTS.get(rel):
            found[rel] = counts
    assert not found, (
        "Russian verdict labels as literals outside AI/verdicts.py and static/js/common.js; "
        f"compare codes (AI.verdicts / LL.verdictCode) and show labels (label() / LL.verdictLabel): {found}"
    )


def test_guard_sees_a_label_literal():
    assert LABEL_LITERAL.search("if verdict == 'Провал':") and LABEL_LITERAL.search('x = "Есть риски"')
    assert not LABEL_LITERAL.search("Статус «Провал»") and not LABEL_LITERAL.search("'Провалы'")
    assert verdicts.__file__.endswith("verdicts.py")
