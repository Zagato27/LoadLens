# SPDX-License-Identifier: Apache-2.0
"""Verdict codes of a report: the one table of codes, labels, tones, sort ranks and SQL mapping.

A verdict is stored and sent by the API as a code: ``pass | risk | fail | no_data``. Rows saved before
the codes keep the Russian labels («Успешно», «Есть риски», «Провал», «Недостаточно данных»); there is no
data migration, readers translate them when they read (``to_code`` in Python, ``code_sql`` in SQL).

Free text (a model that ignored the allowed values) goes through one heuristic, checked in this order:
no_data, fail, risk, pass. Short English tokens (``ok``, ``red``...) match whole words only, so
"covered" is not a failure; "risk of failure" is a risk and "нестабильно" is a risk, not a pass.

The JS copy lives in static/js/common.js (``LL.VERDICTS``, ``LL.verdictCode``, ``LL.verdictLabel``...);
it maps codes and the old labels exactly and has no heuristic. tests/test_verdicts.py keeps both in step.
No Flask or psycopg2 imports: the module is shared by the pipeline, the web app and the tools.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

PASS = "pass"
RISK = "risk"
FAIL = "fail"
NO_DATA = "no_data"
CODES: Tuple[str, ...] = (PASS, RISK, FAIL, NO_DATA)

DEFAULT_LANG = "ru"
LABELS: Dict[str, Dict[str, str]] = {
    "ru": {PASS: "Успешно", RISK: "Есть риски", FAIL: "Провал", NO_DATA: "Недостаточно данных"},
    "en": {PASS: "Passed", RISK: "Risks found", FAIL: "Failed", NO_DATA: "Insufficient data"},
}
# Values stored before the codes (llm_reports.verdict/sla_verdict, report_verdicts.*, parsed.verdict).
LEGACY: Dict[str, str] = {
    "Успешно": PASS,
    "Есть риски": RISK,
    "Провал": FAIL,
    "Недостаточно данных": NO_DATA,
}
# Status tones of the existing CSS (pill ok/warn/fail/na); the class names are not renamed.
TONES: Dict[str, str] = {PASS: "ok", RISK: "warn", FAIL: "fail", NO_DATA: "na"}
# Severity order of the archive sort and of the dashboard chart.
RANK: Dict[str, int] = {PASS: 0, RISK: 1, FAIL: 2, NO_DATA: 3}


def _normalized(value: Any) -> str:
    """Lowercase text with ё folded into е and runs of whitespace collapsed to one space."""
    text = str(value).replace("ё", "е").replace("Ё", "Е").lower()
    return " ".join(text.split())


# Exact matches: the codes and the legacy labels, case, ё and whitespace insensitive.
_EXACT: Dict[str, str] = {
    **{_normalized(code): code for code in CODES},
    **{_normalized(text): code for text, code in LEGACY.items()},
}

# Heuristic for free text: the union of the token sets of the six normalizers that existed before the
# codes. Narrowed where a stem matched common words: "крит" (критерии) is "критич", "рис" (рисунок) is "риск".
_NO_DATA_TOKENS = ("insufficient", "недостаточно", "нет данных", "no data", "unknown", "n/a")
# A failure named as the object of a risk is a risk ("risk of failure", "риск провала").
_RISK_OF_FAILURE = ("risk of fail", "risks of fail", "риск провала", "риски провала", "риск отказа", "риски отказа")
_FAIL_TOKENS = ("fail", "провал", "критич", "critical", "неудовлет", "severe", "error", "ошибка")
_RISK_TOKENS = ("warn", "risk", "риск", "замеч", "degrad", "деград", "под вопрос", "нестабил", "unstable", "предуп")
_PASS_TOKENS = ("success", "passed", "усп", "норма", "стаб")


def _words(*words: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![0-9a-zа-я_])(?:" + "|".join(re.escape(word) for word in words) + r")(?![0-9a-zа-я_])")


_FAIL_WORDS = _words("red")
_PASS_WORDS = _words("ok", "okay", "green", "pass")


def exact_code(value: Any) -> Optional[str]:
    """The code of a code or a legacy label (case, ё and whitespace insensitive), else None.

    For input that must name a verdict: the API body of a manual verdict and the ``?verdict=`` filter.
    """
    if value is None:
        return None
    return _EXACT.get(_normalized(value))


def to_code(value: Any) -> str:
    """The code of any stored or model-written verdict; ``no_data`` when nothing matches.

    Codes and legacy labels map exactly; other text goes through the heuristic. Idempotent:
    ``to_code(to_code(x)) == to_code(x)`` and ``to_code(label(code, lang)) == code``.
    """
    if value is None:
        return NO_DATA
    exact = exact_code(value)
    if exact is not None:
        return exact
    text = _normalized(value)
    if not text:
        return NO_DATA
    if any(token in text for token in _NO_DATA_TOKENS):
        return NO_DATA
    if any(phrase in text for phrase in _RISK_OF_FAILURE):
        return RISK
    if any(token in text for token in _FAIL_TOKENS) or _FAIL_WORDS.search(text):
        return FAIL
    if any(token in text for token in _RISK_TOKENS):
        return RISK
    if any(token in text for token in _PASS_TOKENS) or _PASS_WORDS.search(text):
        return PASS
    return NO_DATA


def optional_code(value: Any) -> Optional[str]:
    """``to_code`` for a nullable column: None and blank text stay None (no SLA verdict, no model verdict)."""
    if value is None or not str(value).strip():
        return None
    return to_code(value)


def _lang(lang: Any) -> str:
    key = str(lang or "").strip().lower()[:2]
    return key if key in LABELS else DEFAULT_LANG


def label(value: Any, lang: str = DEFAULT_LANG) -> str:
    """Visible label of a verdict (code, legacy label or free text) in ``lang`` (unknown languages: ru)."""
    return LABELS[_lang(lang)][to_code(value)]


def tone(value: Any) -> str:
    """Status tone of a verdict: ok / warn / fail / na."""
    return TONES[to_code(value)]


def rank(value: Any) -> int:
    """Sort rank by severity: pass 0, risk 1, fail 2, no_data 3."""
    return RANK[to_code(value)]


def _sql_literal(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _sql_cases() -> Tuple[Tuple[str, str], ...]:
    """(stored value, code) pairs of the SQL mapping: the codes, then the legacy labels as stored."""
    return tuple((code, code) for code in CODES) + tuple(LEGACY.items())


def code_sql(expr: str) -> str:
    """SQL expression mapping ``expr`` (a verdict column or COALESCE) to its code; NULL and others -> no_data.

    Only constants of this module are inlined, never request values. Stored values are written in their
    canonical form, so the CASE compares them as they are (no LOWER: it depends on the database locale).
    """
    whens = " ".join(f"WHEN {_sql_literal(value)} THEN {_sql_literal(code)}" for value, code in _sql_cases())
    return f"(CASE {expr} {whens} ELSE {_sql_literal(NO_DATA)} END)"


def rank_sql(expr: str) -> str:
    """SQL expression with the ``RANK`` of a verdict column (code or legacy label); others rank as no_data."""
    whens = " ".join(f"WHEN {_sql_literal(value)} THEN {RANK[code]}" for value, code in _sql_cases())
    return f"(CASE {expr} {whens} ELSE {RANK[NO_DATA]} END)"


def normalize_report(parsed: Any) -> Any:
    """Copy of a stored analysis with codes in ``verdict`` and ``deterministic_sla.verdict``.

    Texts (rationale, summaries) are not touched; values that are not a dict are returned as they are.
    """
    if not isinstance(parsed, Mapping):
        return parsed
    out = dict(parsed)
    if "verdict" in out:
        out["verdict"] = to_code(out["verdict"])
    sla = out.get("deterministic_sla")
    if isinstance(sla, Mapping) and "verdict" in sla:
        out["deterministic_sla"] = {**sla, "verdict": to_code(sla["verdict"])}
    return out


__all__ = [
    "CODES",
    "DEFAULT_LANG",
    "FAIL",
    "LABELS",
    "LEGACY",
    "NO_DATA",
    "PASS",
    "RANK",
    "RISK",
    "TONES",
    "code_sql",
    "exact_code",
    "label",
    "normalize_report",
    "optional_code",
    "rank",
    "rank_sql",
    "to_code",
    "tone",
]
