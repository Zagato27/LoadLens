# SPDX-License-Identifier: Apache-2.0
"""Publish a LoadLens report to Confluence as a tabbed page with chart attachments.

The page and its PNG charts are written in the language of the report (``stored_report_language`` of the stored
rows: ``parsed.report_language``, then ``sla_details.report_language``, else ``ru``), never in the language of the
person who publishes it. Stored texts (LLM answers, ``domain_titles``, SLA messages) are shown as they were saved.
Progress messages and errors of the publication job are rendered in the locale of the job.

A report built without the model has the final row alone (``analysis: deterministic``): the page shows its verdict
and test figures under the header and has no analysis section. A run with metrics but no stored rows (built
without the model before that row existed) gets its charts and the engineer summary, in the locale of the job.
"""

from __future__ import annotations

import html
import io
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Callable, Iterable
from xml.sax.saxutils import escape as xml_escape

import requests
from psycopg2 import sql

from AI import report_markers
from AI.i18n import SUPPORTED, LocalizedError, get_locale, msg, t, with_locale
from AI.report_sections import test_type_section
from AI.test_profile import normalize_test_type, test_type_label
from AI.verdicts import NO_DATA, is_deterministic_report, normalize_report, optional_code, to_code
from AI.verdicts import label as verdict_label
from loadlens_app.core import _ts_conn
from loadlens_app.html_sanitizer import safe_href
from loadlens_app.i18n import stored_report_language
from settings import CONFIG

logger = logging.getLogger(__name__)

_RATE_LIMIT_STATUS = {429, 503}
_REQUEST_INTERVAL_SEC = 0.35
_MAX_REQUEST_ATTEMPTS = 5


def _retry_delay_seconds(headers: Any, fallback: float) -> float:
    """Seconds to wait after Confluence answers 429, from Retry-After or the fallback."""
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw:
        try:
            return max(float(str(raw).strip()), 0.5)
        except ValueError:
            return fallback
    return fallback

LLM_DOMAIN_ORDER = [
    "final",
    "jvm",
    "database",
    "kafka",
    "microservices",
    "hard_resources",
    "lt_framework",
    "lt_generators",
    "application_logs",
]
# Tab titles of the LLM domains without stored ``domain_titles``: the domain key; «final» is common.domain.final.
LLM_DOMAIN_TITLES = {domain: domain for domain in LLM_DOMAIN_ORDER}
METRIC_DOMAIN_ORDER = [
    "jvm",
    "database",
    "kafka",
    "microservices",
    "hard_resources",
    "lt_framework",
    "lt_generators",
]
ALLOWED_HTML_TAGS = {
    "p", "br", "b", "strong", "i", "em", "u", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "a", "span", "div", "pre", "code",
}
VOID_HTML_TAGS = {"br"}
ProgressCallback = Callable[[str, int | None], None]


class ConfluenceConfigError(LocalizedError, ValueError):
    """Publishing is off or its settings are incomplete; the text renders in the locale where it is shown."""


class ConfluencePublishError(LocalizedError, ValueError):
    """The run has nothing to publish."""


class ConfluenceDependencyError(LocalizedError, RuntimeError):
    """A package the publication needs is not installed."""


def _cfg_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def confluence_config() -> dict[str, Any]:
    return _cfg_dict(CONFIG.get("confluence"))


def _storage_cfg() -> dict[str, Any]:
    return _cfg_dict(_cfg_dict(CONFIG.get("storage")).get("timescale"))


def _require_str(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _peak_table_applies(domain: str, peak: dict[str, Any]) -> bool:
    """Peak RPS belongs only to the final report and the load-tool domain."""
    if domain not in {"final", "lt_framework"}:
        return False
    if peak.get("not_applicable") or peak.get("notApplicable"):
        return False
    return any(_cell_text(peak.get(key), "") for key in ("max_rps", "max_time", "drop_time"))


def _cell_text(value: object, default: str = "—") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return t("confluence.yes") if value else t("confluence.no")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or default
    text = str(value).strip()
    return text or default


def xml_text(value: object) -> str:
    return xml_escape("" if value is None else str(value), {"\"": "&quot;", "'": "&apos;"})


def _parse_jsonish(value: object) -> object:
    if isinstance(value, (dict, list)) or value is None:
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return value
    return value


def _safe_filename(value: str, fallback: str = "chart") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or fallback)[:80]


def _ms_to_text(value: object) -> str:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return "—"
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def wrap_macro(name: str, body: str, params: dict[str, str] | None = None) -> str:
    param_xml = "".join(
        f'<ac:parameter ac:name="{xml_text(key)}">{xml_text(val)}</ac:parameter>'
        for key, val in (params or {}).items()
    )
    inner = f"{param_xml}<ac:rich-text-body>{body}</ac:rich-text-body>" if body else param_xml
    return f'<ac:structured-macro ac:name="{xml_text(name)}">{inner}</ac:structured-macro>'


def wrap_expand(title: str, body: str) -> str:
    return wrap_macro("expand", body, {"title": title})


def wrap_ui_tabs(panels: Iterable[tuple[str, str]]) -> str:
    tabs = "".join(
        wrap_macro("ui-tab", body, {"title": title})
        for title, body in panels
        if title and body is not None
    )
    if not tabs:
        return p(t("confluence.no_data"))
    return wrap_macro("ui-tabs", tabs)


def attachment_image(filename: str) -> str:
    return (
        '<ac:image ac:width="1000">'
        f'<ri:attachment ri:filename="{xml_text(filename)}"/>'
        "</ac:image>"
    )


class _HtmlToStorage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name not in ALLOWED_HTML_TAGS:
            return
        if name == "br":
            self.parts.append("<br />")
            return
        attr_map = {k.lower(): v for k, v in attrs if v}
        rendered: list[str] = [name]
        href = safe_href(attr_map.get("href") or "") if name == "a" else ""
        if href:  # http(s), mailto or relative only: no javascript: links on the wiki page
            rendered.append(f'href="{xml_text(href)}"')
        self.parts.append("<" + " ".join(rendered) + ">")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in ALLOWED_HTML_TAGS and name not in VOID_HTML_TAGS:
            self.parts.append(f"</{name}>")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(xml_text(data))


def html_to_storage(raw_html: str) -> str:
    text = (raw_html or "").strip()
    if not text:
        return p(t("confluence.no_engineer_comment"))
    parser = _HtmlToStorage()
    parser.feed(text)
    parser.close()
    converted = "".join(parser.parts).strip()
    return converted or f"<p>{xml_text(html.unescape(re.sub(r'<[^>]+>', '', text)))}</p>"


def p(text: str) -> str:
    return f"<p>{xml_text(text)}</p>"


def h2(text: str) -> str:
    return f"<h2>{xml_text(text)}</h2>"


def h3(text: str) -> str:
    return f"<h3>{xml_text(text)}</h3>"


def li_list(items: Iterable[str]) -> str:
    rows = "".join(f"<li>{xml_text(item)}</li>" for item in items if str(item).strip())
    return f"<ul>{rows}</ul>" if rows else "<p>—</p>"


def _iso_to_text(value: object) -> str:
    """``2026-10-17 15:20 UTC`` of an ISO timestamp, '' when it does not parse."""
    try:
        dt = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th>{xml_text(label)}</th><td>{xml_text(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table><tbody>{body}</tbody></table>"


def _hhmm(value: object) -> str:
    match = re.search(r"T(\d{2}:\d{2})", str(value or ""))
    return match.group(1) if match else ""


def _step_interval(step: dict[str, Any]) -> str:
    start = _hhmm(step.get("plateau_start_iso") or step.get("start_iso"))
    end = _hhmm(step.get("plateau_end_iso") or step.get("end_iso"))
    if start and end:
        return f"{start}–{end}"
    return start or end or "—"


def _step_flag(step: dict[str, Any]) -> str:
    """Why a step is not a clean load level; same precedence as the report page."""
    if step.get("after_drop"):
        return t("report.steps.flag_after_drop")
    if step.get("dip"):
        return t("report.steps.flag_dip")
    return t("report.steps.flag_unstable") if step.get("stable") is False else ""


def _step_tag(step: dict[str, Any]) -> str:
    """Tag of a step as on the report page: "SLA" (or "target" / "margin" on a step test decided on its target
    step) on the step the verdict was decided on, "max" on the ceiling of such a run."""
    if step.get("selected"):
        role = str(step.get("sla_role") or "")
        if role == "margin":
            return t("report.steps.tag_margin")
        return t("report.steps.tag_target") if role == "target" else "SLA"
    return t("report.steps.tag_ceiling") if step.get("ceiling") else ""


def _step_name(step: dict[str, Any]) -> str:
    flag = _step_flag(step)
    tag = _step_tag(step)
    name = str(step.get("index") or "") + (f" {tag}" if tag else "")
    return f"{name}, {flag}" if flag else name


def load_step_expand(table: dict[str, Any] | None) -> str:
    """Collapsible Confluence block with RPS and response time per load step."""
    if not isinstance(table, dict):
        return ""
    steps = [step for step in (table.get("steps") or []) if isinstance(step, dict)]
    if not steps:
        return ""
    body = "".join(
        "<tr>"
        f"<td>{xml_text(_step_name(step))}</td>"
        f"<td>{xml_text(_step_interval(step))}</td>"
        f"<td>{xml_text(_cell_text(step.get('rps')))}</td>"
        f"<td>{xml_text(_cell_text(step.get('latency_p95')))}</td>"
        "</tr>"
        for step in steps
    )
    head = "<tr>" + "".join(
        f"<th>{xml_text(title)}</th>"
        for title in (t("report.steps.col_step"), t("report.steps.col_interval"), "RPS", t("report.steps.col_latency"))
    ) + "</tr>"
    query = str(table.get("latency_query") or "").strip()
    series = str(table.get("latency_series") or "").strip()
    source_parts = [part for part in (
        t("report.steps.source_query", query=query) if query else "",
        t("report.steps.source_series", series=series) if series else "",
    ) if part]
    source = p(t("confluence.steps.source", source=", ".join(source_parts))) if source_parts else ""
    note = p(t("confluence.steps.note"))
    return wrap_expand(t("report.steps.title"), f"<table><thead>{head}</thead><tbody>{body}</tbody></table>{source}{note}")


# ---- SLA criteria (same content as the block at the top of the report page) ----

# Checks with a title in report.sla_check.<name> (the same titles as on the report page).
SLA_CHECK_NAMES = (
    "target_rps", "target_margin", "error_rate", "p95_latency", "p99_latency", "cpu_usage", "memory_usage", "recovery",
)
# Confluence status lozenge per check result: passed, failed, no data (colour, title key).
_SLA_STATUS = {
    True: ("Green", "confluence.sla.status_pass"),
    False: ("Red", "confluence.sla.status_fail"),
    None: ("Grey", "confluence.sla.status_na"),
}
# Offset the report page assumes when the step times carry none (the page does the same).
_DEFAULT_OFFSET_HOURS = 3.0


def status_macro(title: str, colour: str) -> str:
    return wrap_macro("status", "", {"colour": colour, "title": title})


def sla_check_label(name: str) -> str:
    """Title of an SLA check in the current locale; an unknown check keeps its name."""
    return t(f"report.sla_check.{name}") if name in SLA_CHECK_NAMES else name


def _number_text(value: object) -> str:
    """Number as the report page shows it: integers as is, otherwise up to two decimals."""
    if value is None or value == "" or isinstance(value, bool):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def sla_checks_of(final_row: dict[str, Any]) -> list[dict[str, Any]]:
    sla = _as_dict(final_row.get("sla_details"))
    checks = sla.get("checks")
    if not (isinstance(checks, list) and checks):
        checks = _as_dict(_as_dict(final_row.get("parsed")).get("deterministic_sla")).get("checks")
    return [check for check in checks if isinstance(check, dict)] if isinstance(checks, list) else []


def sla_window_of(final_row: dict[str, Any]) -> dict[str, Any]:
    window = _as_dict(final_row.get("sla_details")).get("stable_window")
    if not isinstance(window, dict):
        window = _as_dict(_as_dict(final_row.get("parsed")).get("deterministic_sla")).get("stable_window")
    return window if isinstance(window, dict) else {}


def sla_target_window_of(final_row: dict[str, Any]) -> dict[str, Any]:
    """Target (or margin) step a step test was decided on; {} without one (AI/report_markers.sla_target_window)."""
    return report_markers.sla_target_window(final_row)


def _moment(value: object) -> datetime | None:
    text = str(value or "").strip().replace(" ", "T")
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _offset_hours(steps: list[dict[str, Any]]) -> float:
    sample = next((str(step.get("end_iso") or step.get("start_iso")) for step in steps if step.get("end_iso") or step.get("start_iso")), "")
    match = re.search(r"([+-])(\d{2}):(\d{2})$", sample)
    if not match:
        return _DEFAULT_OFFSET_HOURS
    sign = -1 if match.group(1) == "-" else 1
    return sign * (int(match.group(2)) + int(match.group(3)) / 60)


def _clock(moment: datetime | None, offset_hours: float) -> str:
    if moment is None:
        return ""
    return (moment.astimezone(timezone.utc) + timedelta(hours=offset_hours)).strftime("%H:%M")


def _step_covering(steps: list[dict[str, Any]], moment: datetime | None) -> dict[str, Any] | None:
    if moment is None:
        return None
    for step in steps:
        start = _moment(step.get("plateau_start_iso") or step.get("start_iso"))
        end = _moment(step.get("plateau_end_iso") or step.get("end_iso"))
        if start and end and start <= moment <= end:
            return step
    return None


def _target_window_note(target: dict[str, Any], window: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    """Note of a step test decided on its target step: 'The target is confirmed on step 1 (…); the latency and
    error thresholds hold up to step 3 (≈300 RPS).'"""
    chosen = next((step for step in steps if step.get("selected")), None) or _step_covering(steps, _moment(target.get("start") or target.get("start_iso")))
    if chosen is None:
        return ""
    key = "report.steps.sla_margin_step" if str(target.get("role") or "") == "margin" else "report.steps.sla_target_step"
    note = t(key, index=chosen.get("index") or "", interval=_step_interval(chosen))
    same = (target.get("start"), target.get("end")) == (window.get("start"), window.get("end"))
    ceiling = None if same else (
        next((step for step in steps if step.get("ceiling")), None) or _step_covering(steps, _moment(window.get("start") or window.get("start_iso")))
    )
    if ceiling is not None and ceiling is not chosen:
        level = _number_text(window.get("level"))
        note += "; " + t("report.steps.sla_ceiling", index=ceiling.get("index") or "", level=_number_text(ceiling.get("rps")) if level == "—" else level)
    return note + "."


def sla_window_note(window: dict[str, Any], step_table: dict[str, Any] | None, target: dict[str, Any] | None = None) -> str:
    """Where the thresholds were checked: the load step (the target step and the ceiling of a step test decided on
    its target step), the soak hold window or the spikes."""
    if not window:
        return ""
    steps = [step for step in ((step_table or {}).get("steps") or []) if isinstance(step, dict)]
    if target and str(window.get("kind") or "step") == "step":
        note = _target_window_note(target, window, steps)
        if note:
            return note
    offset = _offset_hours(steps)
    start, end = _moment(window.get("start") or window.get("start_iso")), _moment(window.get("end") or window.get("end_iso"))
    span = f"{_clock(start, offset)}–{_clock(end, offset)}" if start and end else ""
    level = _number_text(window.get("level"))
    level = "" if level == "—" else level
    kind = str(window.get("kind") or "step")
    if kind == "spike":
        return t("report.steps.sla_spike") + "."
    if kind == "hold":
        key = "report.steps.sla_hold" + ("_span" if span else "") + ("_level" if level else "")
        return t(key, span=span, level=level) + "."
    chosen = next((step for step in steps if step.get("selected")), None) or _step_covering(steps, start)
    if chosen is None:
        if not (span or level):
            return ""
        key = "confluence.sla.stable_step" + ("_span" if span else "") + ("_level" if level else "")
        return t(key, window=span, level=level) + "."
    keys = {
        (False, False): "report.steps.sla_step",
        (True, False): "report.steps.sla_step_window",
        (True, True): "report.steps.sla_step_window_level",
        (False, True): "confluence.sla.step_level",
    }
    note = t(keys[(bool(span), bool(level))], index=chosen.get("index") or "", interval=_step_interval(chosen), window=span, level=level) + "."
    try:
        differs = level and abs(float(chosen.get("rps")) - float(window.get("level"))) > 1
    except (TypeError, ValueError):
        differs = False
    if differs:
        note += " " + t("confluence.sla.mean_note")
    return note


def sla_criteria_block(final_row: dict[str, Any], step_table: dict[str, Any] | None = None) -> str:
    """SLA criteria with status, threshold and actual value, the SLA summary and the checked window."""
    checks = sla_checks_of(final_row)
    if not checks:
        return p(t("confluence.sla.empty", verdict=verdict_label(NO_DATA, get_locale())))
    rows = []
    for check in checks:
        name = str(check.get("name") or "")
        passed = check.get("passed") if isinstance(check.get("passed"), bool) else None
        colour, title_key = _SLA_STATUS[passed]
        note = str(check.get("message") or "").strip()
        label = xml_text(sla_check_label(name) or "—")
        criterion = label + (f"<br />{xml_text(note)}" if note else "")
        rows.append(
            "<tr>"
            f"<td>{status_macro(t(title_key), colour)}</td>"
            f"<td>{criterion}</td>"
            f"<td>{xml_text(_number_text(check.get('threshold')))}</td>"
            f"<td>{xml_text(_number_text(check.get('actual')))}</td>"
            "</tr>"
        )
    head = "<tr>" + "".join(
        f"<th>{xml_text(title)}</th>"
        for title in (t("confluence.sla.col_status"), t("report.sla.col_criterion"), t("report.sla.col_threshold"), t("report.sla.col_actual"))
    ) + "</tr>"
    table = f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"
    summary = str(_as_dict(final_row.get("sla_details")).get("summary") or "").strip()
    note = sla_window_note(sla_window_of(final_row), step_table, sla_target_window_of(final_row))
    return table + (p(summary) if summary else "") + (p(note) if note else "")


def two_col_table(left_title: str, right_title: str, rows: list[tuple[str, str]]) -> str:
    head = (
        "<thead><tr>"
        f"<th>{xml_text(left_title)}</th>"
        f"<th>{xml_text(right_title)}</th>"
        "</tr></thead>"
    )
    body = "".join(
        f"<tr><td>{left}</td><td>{right}</td></tr>"
        for left, right in rows
    )
    return f"<table><colgroup><col /><col /></colgroup>{head}<tbody>{body}</tbody></table>"


def _verdict_text(raw: object) -> str:
    """Label of a verdict (a code, an old label or the model's free text) on the page, in the current locale."""
    return verdict_label(to_code(raw), get_locale())


def _as_dict(value: object) -> dict[str, Any]:
    parsed = _parse_jsonish(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _as_list(value: object) -> list[Any]:
    parsed = _parse_jsonish(value)
    return list(parsed) if isinstance(parsed, list) else []


def _html_paragraphs(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]
    if not blocks:
        return ""
    parts: list[str] = []
    for block in blocks:
        lines = [xml_text(line.strip()) for line in block.splitlines() if line.strip()]
        if lines:
            parts.append(f"<p>{'<br />'.join(lines)}</p>")
    return "".join(parts)


def _empty_cell(text: str) -> str:
    return f"<p><em>{xml_text(text)}</em></p>"


def _link_id(value: object, fallback: str = "") -> str:
    raw = re.sub(r"[^a-z0-9_-]+", "_", _cell_text(value, "").lower())
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or fallback


def _link_ids(value: object) -> list[str]:
    items = value if isinstance(value, list) else ([value] if _cell_text(value, "") else [])
    ids: list[str] = []
    for item in items:
        normalized = _link_id(item)
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def _component_names(data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    single = _cell_text(data.get("component"), "").lower()
    if single:
        names.append(single)
    for item in _as_list(data.get("affected_components")):
        name = _cell_text(item, "").lower()
        if name and name not in names:
            names.append(name)
    return names


def _evidence_items(value: object) -> list[str]:
    items = value if isinstance(value, list) else ([value] if isinstance(value, dict) else [])
    lines: list[str] = []
    for item in items:
        data = _as_dict(item)
        if not data:
            text = _cell_text(item, "")
            if text:
                lines.append(text)
            continue
        bits: list[str] = []
        metric = _cell_text(data.get("metric") or data.get("name") or data.get("label"), "")
        observed = _cell_text(data.get("observed_value") or data.get("value") or data.get("actual"), "")
        threshold = _cell_text(data.get("threshold") or data.get("limit") or data.get("baseline"), "")
        note = _cell_text(data.get("note") or data.get("details") or data.get("evidence"), "")
        if metric:
            bits.append(metric)
        if observed:
            bits.append(t("report.evidence.value", value=observed))
        if threshold:
            bits.append(t("report.evidence.threshold", value=threshold))
        if note:
            bits.append(note)
        if bits:
            lines.append(" | ".join(bits))
    return lines


def unverified_suffix() -> str:
    """Mark of a finding the data did not confirm, in the current locale (the verify prompt writes the same)."""
    return t("confluence.unverified_suffix")


def _has_unverified_mark(summary: str) -> bool:
    """Whether the model already marked the summary, in any language of the prompts."""
    return any(t("confluence.unverified_suffix", locale=lang) in summary for lang in SUPPORTED)


def _finding_html(item: object) -> str:
    data = _as_dict(item) if not isinstance(item, str) else {"summary": item}
    summary = _cell_text(data.get("summary") or data.get("title") or data.get("text"), "")
    verification = data.get("verification") if isinstance(data.get("verification"), dict) else {}
    if summary and str(verification.get("status") or "").lower() == "unverified" and not _has_unverified_mark(summary):
        summary = f"{summary} {unverified_suffix()}"
    parts: list[str] = []
    if summary:
        parts.append(f"<p><strong>{xml_text(summary)}</strong></p>")
    evidence = _cell_text(data.get("evidence_summary") or data.get("evidence"), "")
    if evidence:
        parts.append(_html_paragraphs(evidence))
    evidence_lines = _evidence_items(data.get("evidence_items") or data.get("evidence_list") or data.get("evidence_rows"))
    if evidence_lines:
        parts.append("<ul>" + "".join(f"<li>{xml_text(line)}</li>" for line in evidence_lines) + "</ul>")
    meta: list[str] = []
    severity = _cell_text(data.get("severity"), "")
    component = _cell_text(data.get("component"), "")
    if severity:
        meta.append(t("confluence.line.severity", value=severity))
    if component:
        meta.append(t("confluence.line.component", value=component))
    if meta:
        parts.append(f"<p>{xml_text(' · '.join(meta))}</p>")
    return "".join(parts) or _empty_cell("—")


def _action_html(item: object) -> str:
    data = _as_dict(item) if not isinstance(item, str) else {"summary": item}
    summary = _cell_text(data.get("summary") or data.get("action") or data.get("text"), "")
    details = _cell_text(
        data.get("details") or data.get("description") or data.get("implementation_details"),
        "",
    )
    parts: list[str] = []
    if summary:
        parts.append(f"<p><strong>{xml_text(summary)}</strong></p>")
    if details:
        parts.append(_html_paragraphs(details))
    return "".join(parts) or _empty_cell("—")


def _action_stack(items: list[object], empty_text: str) -> str:
    if not items:
        return _empty_cell(empty_text)
    return "".join(_action_html(item) for item in items)


def _normalize_finding_entry(item: object, idx: int) -> dict[str, Any] | None:
    data = _as_dict(item) if not isinstance(item, str) else {"summary": item}
    summary = _cell_text(data.get("summary") or data.get("title") or data.get("text"), "")
    if not summary:
        return None
    return {
        "idx": idx,
        "id": _link_id(data.get("id") or data.get("finding_id") or data.get("key"), f"finding_{idx + 1}"),
        "components": _component_names(data),
        "item": item if not isinstance(item, str) else data,
    }


def _normalize_action_entry(item: object, idx: int) -> dict[str, Any] | None:
    data = _as_dict(item) if not isinstance(item, str) else {"summary": item}
    summary = _cell_text(data.get("summary") or data.get("action") or data.get("text"), "")
    if not summary:
        return None
    return {
        "idx": idx,
        "components": _component_names(data),
        "for_finding_ids": _link_ids(
            data.get("for_finding_ids")
            or data.get("for_findings")
            or data.get("finding_ids")
            or data.get("related_findings")
            or data.get("for_finding_id")
            or data.get("finding_id")
        ),
        "item": item if not isinstance(item, str) else data,
    }


def _match_legacy_action(
    finding: dict[str, Any],
    action_entries: list[dict[str, Any]],
    unused_ids: set[int],
) -> dict[str, Any] | None:
    legacy = [item for item in action_entries if not item["for_finding_ids"] and item["idx"] in unused_ids]
    matched: dict[str, Any] | None = None
    if finding["components"]:
        matched = next(
            (
                action for action in legacy
                if any(component in finding["components"] for component in action["components"])
            ),
            None,
        )
    if matched is None and finding["idx"] in unused_ids:
        matched = next((action for action in legacy if action["idx"] == finding["idx"]), None)
    if matched is None:
        matched = legacy[0] if legacy else None
    if matched is not None:
        unused_ids.discard(matched["idx"])
    return matched


def pair_findings_with_actions(findings: list[Any], actions: list[Any]) -> list[tuple[str, str]]:
    finding_entries = [
        entry for idx, item in enumerate(findings)
        if (entry := _normalize_finding_entry(item, idx))
    ]
    action_entries = [
        entry for idx, item in enumerate(actions)
        if (entry := _normalize_action_entry(item, idx))
    ]
    linked_ids = {entry["id"] for entry in finding_entries}
    unused_legacy = {entry["idx"] for entry in action_entries if not entry["for_finding_ids"]}
    rows: list[tuple[str, str]] = []
    for finding in finding_entries:
        explicit = [
            action["item"]
            for action in action_entries
            if finding["id"] in action["for_finding_ids"]
        ]
        legacy = None if explicit else _match_legacy_action(finding, action_entries, unused_legacy)
        recommendations = explicit or ([legacy["item"]] if legacy else [])
        rows.append((
            _finding_html(finding["item"]),
            _action_stack(recommendations, t("report.findings.no_recommendation")),
        ))
    for action in action_entries:
        dangling = (
            (not action["for_finding_ids"] and action["idx"] in unused_legacy)
            or (action["for_finding_ids"] and not any(item_id in linked_ids for item_id in action["for_finding_ids"]))
        )
        if dangling:
            rows.append((
                _empty_cell(t("report.findings.extra_recommendation")),
                _action_stack([action["item"]], t("report.findings.no_recommendation")),
            ))
    if not rows:
        rows.append((
            _empty_cell(t("report.findings.no_problems")),
            _action_stack([entry["item"] for entry in action_entries], t("report.findings.no_recommendations")),
        ))
    return rows


def _verdict_rows(row: dict[str, Any], parsed: dict[str, Any]) -> list[tuple[str, str]]:
    """Verdict rows of a report: the verdict; a manual one names who, when and why and keeps the automatic one."""
    verdict = _verdict_text(row.get("sla_verdict") or parsed.get("verdict") or row.get("verdict"))
    manual = row.get("manual_verdict") if row.get("domain") == "final" else None
    if isinstance(manual, dict) and manual.get("verdict"):
        who = ", ".join(part for part in (str(manual.get("author") or ""), _iso_to_text(manual.get("changed_at"))) if part)
        reason = str(manual.get("reason") or "").strip()
        rows = [(t("confluence.verdict.title"), _verdict_text(manual["verdict"]))]
        changed = t("confluence.verdict.manual_by", who=who, reason=reason) if who else t("report.verdict.reason_line", reason=reason)
        rows.append((t("confluence.verdict.manual"), changed))
        rows.append((t("confluence.verdict.auto"), verdict))
        return rows
    rows = [(t("confluence.verdict.title"), verdict)]
    llm = _verdict_text(row.get("llm_verdict") or parsed.get("verdict"))
    # Rows saved before the SLA verdict replaced the model's one keep a different model verdict.
    if row.get("domain") == "final" and row.get("llm_verdict") and row.get("sla_verdict") and llm != verdict:
        rows.append((t("confluence.verdict.llm"), llm))
        rows.append(("SLA", _verdict_text(row.get("sla_verdict"))))
    return rows


def _test_figures(row: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    """Test-type section of the final report (saturation, spikes, stability), else the peak RPS table."""
    peak = _as_dict(parsed.get("peak_performance") or parsed.get("peak_perfomance"))
    section = test_type_section(parsed) if row.get("domain") == "final" else None
    if section is not None:
        title, section_rows = section
        return [h3(title), kv_table(section_rows)]
    if _peak_table_applies(str(row.get("domain") or ""), peak):
        return [h3(t("report.structured.peak_title")), kv_table([
            (t("report.structured.max_rps"), _cell_text(peak.get("max_rps"))),
            (t("report.section.peak_time"), _cell_text(peak.get("max_time"))),
            (t("report.section.drop_time"), _cell_text(peak.get("drop_time"))),
        ])]
    return []


def render_deterministic_summary(row: dict[str, Any]) -> str:
    """Final row of a report built without the model: the verdict, the test figures and a note on the analysis."""
    parsed = _as_dict(row.get("parsed"))
    return "".join([kv_table(_verdict_rows(row, parsed)), *_test_figures(row, parsed), p(t("confluence.no_llm_analysis"))])


def render_llm_domain_body(row: dict[str, Any]) -> str:
    parsed = _as_dict(row.get("parsed"))
    analysis_error = _as_dict(parsed.get("analysis_error"))
    if analysis_error:
        message = _cell_text(analysis_error.get("message"), t("report.analysis_error.default_message"))
        details = [
            _cell_text(analysis_error.get("provider"), ""),
            t("report.analysis_error.limit", value=analysis_error.get("max_tokens")) if analysis_error.get("max_tokens") else "",
            t("report.analysis_error.output", value=analysis_error.get("output_tokens")) if analysis_error.get("output_tokens") is not None else "",
        ]
        details = [item for item in details if item]
        body = h3(t("report.analysis_error.title")) + p(message)
        if details:
            body += p(" · ".join(details))
        excerpt = _require_str(analysis_error.get("excerpt"))
        if excerpt:
            body += f"<pre>{xml_text(excerpt[:2000])}</pre>"
        return body
    if not parsed and isinstance(row.get("text"), str) and "{" in row["text"]:
        raw = str(row["text"])
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            parsed = _as_dict(raw[start:end + 1])
    if not parsed:
        text = _require_str(row.get("text"))
        return f"<pre>{xml_text(text[:12000])}</pre>" if text else p(t("confluence.no_data"))

    parts = [kv_table(_verdict_rows(row, parsed))]

    rationale = _cell_text(
        parsed.get("verdict_rationale") or parsed.get("verdict_reason") or parsed.get("rationale"),
        "",
    )
    if rationale:
        parts.append(h3(t("report.rationale.title")))
        parts.append(_html_paragraphs(rationale))

    parts.extend(_test_figures(row, parsed))

    parts.append(h3(t("report.findings.title")))
    parts.append(two_col_table(
        t("report.findings.problems"),
        t("report.findings.recommendations"),
        pair_findings_with_actions(
            _as_list(parsed.get("findings")),
            _as_list(parsed.get("recommended_actions") or parsed.get("actions")),
        ),
    ))

    affected = [_cell_text(item, "") for item in _as_list(parsed.get("affected_components"))]
    affected = [item for item in affected if item]
    if affected:
        parts.append(h3(t("confluence.affected_components")))
        parts.append(p(", ".join(affected)))
    return "".join(parts)


def _system_context_items(ctx: dict[str, Any]) -> list[str]:
    system = _as_dict(ctx.get("system"))
    architecture = _as_dict(ctx.get("architecture"))
    load_model = _as_dict(ctx.get("load_model"))
    operational = _as_dict(ctx.get("operational_context"))
    items: list[str] = []
    if _require_str(system.get("name")):
        items.append(t("confluence.line.system", value=_require_str(system.get("name"))))
    if _require_str(system.get("domain")):
        items.append(t("confluence.line.domain", value=_require_str(system.get("domain"))))
    if _require_str(system.get("description")):
        items.append(_require_str(system.get("description")))
    if _require_str(system.get("test_goal")):
        items.append(t("confluence.line.test_goal", value=_require_str(system.get("test_goal"))))
    if _require_str(architecture.get("style")):
        items.append(t("confluence.line.architecture", value=_require_str(architecture.get("style"))))
    for component in _as_list(architecture.get("components"))[:20]:
        data = _as_dict(component)
        name = _require_str(data.get("name") or data.get("id"))
        if name:
            items.append(t("confluence.line.component", value=name))
    for flow in _as_list(load_model.get("critical_user_flows"))[:10]:
        data = _as_dict(flow)
        name = _require_str(data.get("name") or data.get("id"))
        if name:
            items.append(t("confluence.line.critical_flow", value=name))
    for risk in _as_list(operational.get("known_risks"))[:10]:
        if _require_str(risk):
            items.append(t("confluence.line.risk", value=_require_str(risk)))
    return items


PNG_MAX_SERIES = 15
_PNG_INK = "#15171c"  # light-theme --text


def series_plan(points: list[dict[str, Any]]) -> tuple[list[str], list[tuple[str, str, dict[str, float]]]]:
    """Sorted bucket labels and the drawn series as (name, color, values by label).

    Series are ordered like the web page orders its datasets (AI/report_markers.order_series) and
    colored by slot from the shared palette, so a series has the same color on the page and in the PNG.
    """
    labels: list[str] = []
    seen: set[str] = set()
    series_map: dict[str, dict[str, float]] = {}
    for point in points:
        ts = str(point.get("t") or "")
        series = str(point.get("series") or "series")
        try:
            value = float(point.get("value"))
        except (TypeError, ValueError):
            continue
        if ts not in seen:
            seen.add(ts)
            labels.append(ts)
        series_map.setdefault(series, {})[ts] = value
    labels.sort(key=lambda ts: (report_markers.moment_ms(ts) is None, report_markers.moment_ms(ts) or 0, ts))
    ordered = report_markers.order_series({name: [values[ts] for ts in labels if ts in values] for name, values in series_map.items()})
    return labels, [(name, report_markers.series_color(i), series_map[name]) for i, name in enumerate(ordered[:PNG_MAX_SERIES])]


def marker_geometry(markers: Iterable[dict[str, Any]] | None, times: list[float]) -> list[dict[str, Any]]:
    """Markers placed on the category axis like LL.charts.markers places them on the web chart.

    Returns ``{"kind", "x" | ("x0", "x1"), "label", "tone"}`` in axis units (bucket positions);
    markers outside the data range (beyond half a bucket) are dropped and bands are clipped.
    """
    if not times:
        return []
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    half = 0.0
    if gaps:
        mid = len(gaps) // 2
        half = max(0.0, (gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2) / 2)
    first, last = times[0], times[-1]
    out: list[dict[str, Any]] = []
    for item in markers or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "line")
        tone = str(item.get("tone") or ("muted" if kind == "tick" else "accent"))
        label = str(item.get("label") or "")
        start = report_markers._finite(item.get("t"))
        if start is None:
            continue
        if kind == "band":
            end = report_markers._finite(item.get("to"))
            if end is None:
                continue
            a, b = min(start, end), max(start, end)
            if b < first - half or a > last + half:
                continue
            x0 = report_markers.fractional_index(times, max(a, first))
            x1 = report_markers.fractional_index(times, min(b, last))
            out.append({"kind": "band", "x0": x0, "x1": max(x1, x0), "label": label, "tone": tone})
            continue
        if start < first - half or start > last + half:
            continue
        out.append({"kind": "tick" if kind == "tick" else "line", "x": report_markers.fractional_index(times, start), "label": label, "tone": tone})
    return out


def _draw_thresholds(ax: Any, thresholds: Iterable[dict[str, Any]] | None) -> None:
    items = [item for item in thresholds or [] if isinstance(item, dict) and report_markers._finite(item.get("value")) is not None]
    if not items:
        return
    values = [float(item["value"]) for item in items]
    lo, hi = ax.get_ylim()
    span = (max(hi, *values) - min(lo, *values)) or abs(max(values)) or 1.0
    if max(values) + span * 0.05 > hi:
        hi = max(values) + span * 0.05
    if min(values) < lo:
        lo = min(values) - span * 0.05
    ax.set_ylim(lo, hi)
    for item in items:
        color = report_markers.TONE_COLORS.get(str(item.get("tone") or "fail"), report_markers.TONE_COLORS["fail"])
        ax.axhline(float(item["value"]), color=color, linestyle=(0, (6, 4)), linewidth=1.5, zorder=3)
        if item.get("label"):
            ax.annotate(
                str(item["label"]), xy=(1, float(item["value"])), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=_PNG_INK, zorder=5,
                bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": color, "lw": 0.8, "alpha": 0.92},
            )


def _draw_markers(fig: Any, ax: Any, placed: list[dict[str, Any]]) -> None:
    """Bands under the lines, dashed lines and ticks over them, labels stacked along the top (max 3 rows)."""
    labels: list[tuple[float, str, str]] = []
    for item in placed:
        color = report_markers.TONE_COLORS.get(item["tone"], report_markers.TONE_COLORS["accent"])
        if item["kind"] == "band":
            ax.axvspan(item["x0"], item["x1"], color=color, alpha=0.1 if item["tone"] == "muted" else 0.14, linewidth=0, zorder=0.5)
            if item["label"]:
                labels.append((item["x0"], item["label"], color))
        elif item["kind"] == "tick":
            ax.axvline(item["x"], ymin=0, ymax=0.04, color=color, linewidth=1.5, zorder=3)
        else:
            ax.axvline(item["x"], color=color, linestyle=(0, (5, 4)), linewidth=1.5, zorder=3)
            if item["label"]:
                labels.append((item["x"], item["label"], color))
    if not labels:
        return
    renderer = fig.canvas.get_renderer()
    area = ax.get_window_extent(renderer)
    rows_right = [float("-inf")] * 3
    for x, text, color in sorted(labels, key=lambda entry: entry[0]):
        note = ax.annotate(
            text, xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
            ha="left", va="top", fontsize=8, color=_PNG_INK, zorder=6, annotation_clip=False,
            bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": color, "lw": 0.8, "alpha": 0.92},
        )
        box = note.get_window_extent(renderer)
        if box.x1 > area.x1 - 2:
            note.set_horizontalalignment("right")
            note.xyann = (-4, -4)
            box = note.get_window_extent(renderer)
        row = next((i for i, right in enumerate(rows_right) if right + 4 <= box.x0), None)
        if row is None:
            row = rows_right.index(min(rows_right))
        rows_right[row] = max(rows_right[row], box.x1)
        note.xyann = (note.xyann[0], -4 - row * 16)


def render_series_png(
    points: list[dict[str, Any]],
    title: str,
    *,
    markers: Iterable[dict[str, Any]] | None = None,
    thresholds: Iterable[dict[str, Any]] | None = None,
    clock_offset: float | None = None,
) -> bytes:
    """Chart PNG for Confluence: the web chart's series colors, markers and SLA thresholds.

    ``markers`` and ``thresholds`` come from AI/report_markers (``markers_for``, ``thresholds_for``
    filtered by the chart query). ``clock_offset`` (hours) formats the time ticks in the report
    clock, as the marker labels and the step table; without it the bucket text is shown as is.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, ranked = series_plan(points)
    times = [report_markers.moment_ms(ts) for ts in labels]

    fig, ax = plt.subplots(figsize=(12.0, 4.6), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")
    ax.set_title(title)
    ax.set_xlabel(t("confluence.chart.time"))
    ax.grid(True, color="#e6e9ef")
    placed: list[dict[str, Any]] = []
    if not labels or not ranked:
        ax.text(0.5, 0.5, t("report.page.no_data"), ha="center", va="center", transform=ax.transAxes)
    else:
        for name, color, values in ranked:
            xs = [i for i, ts in enumerate(labels) if ts in values]
            ax.plot(xs, [values[labels[i]] for i in xs], label=name[:60], color=color, linewidth=1.6, zorder=2)
        if len(labels) > 1:
            ax.set_xlim(0, len(labels) - 1)
        tick_positions = list(range(0, len(labels), max(1, len(labels) // 8)))
        ax.set_xticks(tick_positions)

        def tick_text(ts: str, ms: int | None) -> str:
            if clock_offset is not None and ms is not None:
                return report_markers.clock(ms, clock_offset)
            return ts.replace("T", " ")[11:16] if "T" in ts else ts[-5:]

        ax.set_xticklabels([tick_text(labels[i], times[i]) for i in tick_positions], rotation=0)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
        _draw_thresholds(ax, thresholds)
        if all(t is not None for t in times):
            placed = marker_geometry(markers, [float(t) for t in times])
    fig.tight_layout()
    _draw_markers(fig, ax, placed)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _progress(callback: ProgressCallback | None, message: str, pct: int | None = None) -> None:
    """Reports a stage of the job; ``message`` is a :func:`AI.i18n.msg` so the job list can show it in any locale."""
    if callback:
        callback(message, pct)


def ensure_publications_table(conn, storage_cfg: dict[str, Any] | None = None) -> None:
    cfg = storage_cfg or _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("confluence_table", "confluence_publications")
    prev = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.{} (
                        id             BIGSERIAL PRIMARY KEY,
                        created_at     TIMESTAMPTZ DEFAULT now(),
                        updated_at     TIMESTAMPTZ DEFAULT now(),
                        run_name       TEXT NOT NULL UNIQUE,
                        service        TEXT,
                        page_id        TEXT NOT NULL,
                        page_url       TEXT,
                        space_key      TEXT,
                        parent_page_id TEXT
                    );
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (run_name);").format(
                    sql.Identifier(f"idx_{table}_run"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
    finally:
        try:
            conn.autocommit = prev
        except Exception:
            pass


def load_publication(run_name: str) -> dict[str, Any] | None:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("confluence_table", "confluence_publications")
    conn = _ts_conn()
    try:
        ensure_publications_table(conn, cfg)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT page_id, page_url, space_key, parent_page_id, service, updated_at "
                    "FROM {}.{} WHERE run_name = %s"
                ).format(sql.Identifier(schema), sql.Identifier(table)),
                (run_name,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "page_id": row[0],
            "page_url": row[1],
            "space_key": row[2],
            "parent_page_id": row[3],
            "service": row[4],
            "updated_at": row[5].isoformat() if row[5] else None,
        }
    finally:
        conn.close()


def save_publication(run_name: str, service: str, page_id: str, page_url: str, space_key: str, parent_page_id: str) -> None:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("confluence_table", "confluence_publications")
    conn = _ts_conn()
    try:
        ensure_publications_table(conn, cfg)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.{} (run_name, service, page_id, page_url, space_key, parent_page_id, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (run_name) DO UPDATE SET
                        service = EXCLUDED.service,
                        page_id = EXCLUDED.page_id,
                        page_url = EXCLUDED.page_url,
                        space_key = EXCLUDED.space_key,
                        parent_page_id = EXCLUDED.parent_page_id,
                        updated_at = now()
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table)),
                (run_name, service, page_id, page_url, space_key, parent_page_id),
            )
        conn.commit()
    finally:
        conn.close()


def report_domain_titles(llm_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Titles the report stored for its domains (query packs title «jvm» as «Go runtime» and the like)."""
    final = next((row for row in llm_rows if str(row.get("domain")) == "final"), {})
    parsed = final.get("parsed")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            parsed = None
    titles = parsed.get("domain_titles") if isinstance(parsed, dict) else None
    if not isinstance(titles, dict):
        return {}
    return {str(key): str(value) for key, value in titles.items() if isinstance(value, str) and value.strip()}


def load_llm_rows(run_name: str) -> list[dict[str, Any]]:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("llm_table", "llm_reports")
    conn = _ts_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                    SELECT DISTINCT ON (domain)
                    run_name, service, start_ms, end_ms, domain, verdict, text, parsed,
                    scores, sla_verdict, sla_details, system_context, created_at, test_type
                FROM {schema}.{table}
                WHERE run_name = %s AND domain <> 'engineer'
                ORDER BY domain, created_at DESC
                """,
                (run_name,),
            )
            rows = cur.fetchall()
        manual = None
        if any(row[4] == "final" for row in rows):
            from AI.db_store import load_verdict_state

            try:
                state = load_verdict_state(conn, schema, table, run_name)
                manual = state["manual"] if state else None
            except Exception as exc:  # the page is published with the automatic verdict
                logger.warning("confluence: manual verdict of %s not read: %s", run_name, exc)
    finally:
        conn.close()
    data: list[dict[str, Any]] = []
    for row in rows:
        data.append({
            "manual_verdict": manual if row[4] == "final" else None,
            "run_name": row[0],
            "service": row[1],
            "start_ms": int(row[2]) if row[2] is not None else None,
            "end_ms": int(row[3]) if row[3] is not None else None,
            "domain": row[4],
            "verdict": optional_code(row[5]),
            "llm_verdict": optional_code(row[5]),
            "text": row[6],
            "parsed": normalize_report(row[7]),
            "scores": row[8],
            "sla_verdict": optional_code(row[9]),
            "sla_details": row[10],
            "system_context": row[11],
            "test_type": row[13] if len(row) > 13 else None,
            "created_at": row[12],
        })
    return data


def _embedded_step_table(row: dict[str, Any]) -> dict[str, Any] | None:
    for source_name in ("sla_details", "parsed"):
        source = _as_dict(row.get(source_name))
        table = source.get("load_step_table")
        if isinstance(table, dict) and table.get("steps"):
            return table
    return None


def _load_domain_context(run_name: str, domain: str) -> dict[str, Any]:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("llm_table", "llm_reports")
    conn = _ts_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT context FROM {schema}.{table}
                WHERE run_name = %s AND domain = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (run_name, domain),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    parsed = _parse_jsonish(row[0]) if row else None
    return parsed if isinstance(parsed, dict) else {}


def resolve_load_step_table(run_name: str, llm_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Step table saved with the report, or rebuilt from the stored model context."""
    final = next((row for row in llm_rows if str(row.get("domain")) == "final"), {})
    if not isinstance(final, dict):
        final = {}
    embedded = _embedded_step_table(final)
    if embedded:
        return embedded
    try:
        from AI.context_pack import load_step_report_from_context

        pack = _load_domain_context(run_name, "lt_framework")
        if not pack.get("load_steps"):
            pack = _load_domain_context(run_name, "final")
        checks = _as_dict(final.get("sla_details")).get("checks")
        return load_step_report_from_context(pack, checks=checks if isinstance(checks, list) else None)
    except Exception:
        logger.exception("Не удалось собрать таблицу ступеней для Confluence, run=%s", run_name)
        return None


def load_engineer_html(run_name: str) -> str:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("engineer_table", "engineer_reports")
    conn = _ts_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT content_html FROM {schema}.{table}
                WHERE run_name = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (run_name,),
            )
            row = cur.fetchone()
        return str(row[0] or "") if row else ""
    except Exception:
        return ""
    finally:
        conn.close()


def load_domains_schema(run_name: str) -> dict[str, list[str]]:
    conn = _ts_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT domain, query_label
                FROM public.metrics
                WHERE run_name = %s
                GROUP BY domain, query_label
                ORDER BY domain, query_label
                """,
                (run_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    out: dict[str, list[str]] = {}
    for domain, query_label in rows:
        out.setdefault(str(domain), []).append(str(query_label))
    return out


def _datetime_ms(value: object) -> int | None:
    return int(value.timestamp() * 1000) if isinstance(value, datetime) else None


def load_run_meta(run_name: str) -> dict[str, Any] | None:
    """``{service, start_ms, end_ms}`` of a run without stored report rows, from ``run_index`` or the metrics.

    None when the run has no metrics either.
    """
    from AI.db_store import RUN_INDEX_TABLE

    schema = str(_storage_cfg().get("schema") or "public")
    conn = _ts_conn()
    try:
        row = None
        with conn.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL("SELECT COALESCE(service, ''), start_time, end_time FROM {} WHERE run_name = %s").format(
                        sql.Identifier(schema, RUN_INDEX_TABLE)
                    ),
                    (run_name,),
                )
                row = cur.fetchone()
            except Exception as exc:  # a database without the index answers from the metrics
                logger.warning("confluence: run_index of %s not read: %s", run_name, exc)
                conn.rollback()
        if row is None:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT COALESCE(MAX(service), ''), MIN("time"), MAX("time") FROM public.metrics WHERE run_name = %s""",
                    (run_name,),
                )
                row = cur.fetchone()
    finally:
        conn.close()
    if not row or row[1] is None:
        return None
    return {"service": str(row[0] or ""), "start_ms": _datetime_ms(row[1]), "end_ms": _datetime_ms(row[2])}


def load_run_series(run_name: str, domain: str, query_label: str) -> list[dict[str, Any]]:
    conn = _ts_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH base AS (
                  SELECT m."time", m.series AS series_name, m.value
                  FROM public.metrics m
                  WHERE m.run_name = %s AND m.domain = %s AND m.query_label = %s
                )
                SELECT time_bucket('1 minute'::interval, base."time") AS t,
                       base.series_name, avg(base.value) AS v
                FROM base GROUP BY 1, 2 ORDER BY 1, 2
                """,
                (run_name, domain, query_label),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"t": row[0].isoformat(), "series": row[1], "value": float(row[2])} for row in rows]


class ConfluenceClient:
    def __init__(self, cfg: dict[str, Any]):
        self.base_url = _require_str(cfg.get("base_url")).rstrip("/")
        self.username = _require_str(cfg.get("username"))
        self.password = _require_str(cfg.get("password"))
        self.space_key = _require_str(cfg.get("space_key"))
        self.parent_page_id = _require_str(cfg.get("parent_page_id"))
        self.verify_ssl = bool(cfg.get("verify_ssl", False))
        self.timeout = int(cfg.get("request_timeout_sec") or 60)
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self.session.headers.update({"Accept": "application/json"})
        self._next_request_at = 0.0

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _pace(self) -> None:
        wait = self._next_request_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next_request_at = time.monotonic() + _REQUEST_INTERVAL_SEC

    def _request(self, method: str, path: str, action: str, **kwargs: Any) -> requests.Response:
        """One Confluence call, spaced from the previous one and retried on 429/503."""
        kwargs.setdefault("verify", self.verify_ssl)
        kwargs.setdefault("timeout", self.timeout)
        fallback = 1.0
        response: requests.Response | None = None
        for attempt in range(_MAX_REQUEST_ATTEMPTS):
            self._pace()
            response = self.session.request(method, self._url(path), **kwargs)
            if response.status_code not in _RATE_LIMIT_STATUS or attempt == _MAX_REQUEST_ATTEMPTS - 1:
                return response
            delay = _retry_delay_seconds(response.headers, fallback)
            logger.warning(
                "Confluence %s rate limited: status=%s, retry %s/%s in %.1fs",
                action, response.status_code, attempt + 1, _MAX_REQUEST_ATTEMPTS, delay,
            )
            time.sleep(delay)
            fallback = min(fallback * 2, 30.0)
        raise RuntimeError(f"Confluence {action} failed without a response")

    def _raise_for_status(self, response: requests.Response, action: str) -> None:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Confluence {action} failed: status={response.status_code}, body={response.text[:1000]}"
            )

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        response = self._request(
            "GET",
            f"/rest/api/content/{page_id}",
            "get page",
            params={"expand": "version,space,ancestors"},
        )
        if response.status_code == 404:
            return None
        self._raise_for_status(response, "get page")
        payload = response.json()
        return payload if isinstance(payload, dict) else None

    def find_page_by_title(self, title: str) -> dict[str, Any] | None:
        response = self._request(
            "GET",
            "/rest/api/content",
            "search page",
            params={"spaceKey": self.space_key, "title": title, "expand": "version,space"},
        )
        self._raise_for_status(response, "search page")
        payload = response.json() if response.content else {}
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0]
        return None

    def create_page(self, title: str, body: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "page",
            "title": title,
            "space": {"key": self.space_key},
            "body": {"storage": {"value": body or "<p></p>", "representation": "storage"}},
        }
        if self.parent_page_id:
            payload["ancestors"] = [{"id": self.parent_page_id}]
        response = self._request("POST", "/rest/api/content", "create page", json=payload)
        self._raise_for_status(response, "create page")
        created = response.json()
        if not isinstance(created, dict) or not created.get("id"):
            raise RuntimeError("Confluence create page returned no page id")
        return created

    def update_page(self, page_id: str, title: str, body: str, version: int) -> dict[str, Any]:
        payload = {
            "id": str(page_id),
            "type": "page",
            "title": title,
            "space": {"key": self.space_key},
            "version": {"number": int(version) + 1},
            "body": {"storage": {"value": body, "representation": "storage"}},
        }
        if self.parent_page_id:
            payload["ancestors"] = [{"id": self.parent_page_id}]
        response = self._request("PUT", f"/rest/api/content/{page_id}", "update page", json=payload)
        self._raise_for_status(response, "update page")
        updated = response.json()
        if not isinstance(updated, dict):
            raise RuntimeError("Confluence update page returned invalid payload")
        return updated

    def list_attachments(self, page_id: str) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            f"/rest/api/content/{page_id}/child/attachment",
            "list attachments",
            params={"limit": 200},
        )
        self._raise_for_status(response, "list attachments")
        payload = response.json() if response.content else {}
        results = payload.get("results") if isinstance(payload, dict) else []
        return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []

    def delete_attachment(self, attachment_id: str) -> None:
        response = self._request(
            "DELETE",
            f"/rest/api/content/{attachment_id}",
            "delete attachment",
            headers={"X-Atlassian-Token": "no-check"},
        )
        if response.status_code not in {204, 404}:
            self._raise_for_status(response, "delete attachment")

    def upload_attachment(self, page_id: str, filename: str, content: bytes, existing_id: str = "") -> None:
        """Uploads a chart, replacing the attachment of the same name when it already exists."""
        headers = {"X-Atlassian-Token": "no-check"}
        files = {"file": (filename, content, "image/png")}
        timeout = max(self.timeout, 120)
        action = f"upload attachment {filename}"
        if existing_id:
            response = self._request(
                "POST",
                f"/rest/api/content/{page_id}/child/attachment/{existing_id}/data",
                action,
                headers=headers,
                files=files,
                timeout=timeout,
            )
            if response.status_code not in {404, 405}:
                self._raise_for_status(response, action)
                return
        response = self._request(
            "POST",
            f"/rest/api/content/{page_id}/child/attachment",
            action,
            headers=headers,
            files={"file": (filename, content, "image/png")},
            timeout=timeout,
        )
        self._raise_for_status(response, action)

    def page_url(self, page: dict[str, Any]) -> str:
        links = page.get("_links") if isinstance(page.get("_links"), dict) else {}
        webui = _require_str(links.get("webui"))
        base = _require_str(links.get("base")) or self.base_url
        if webui:
            return f"{base.rstrip('/')}{webui}"
        return f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}"


_UNRESOLVED: Any = object()


def chart_annotations(llm_rows: list[dict[str, Any]], step_table: dict[str, Any] | None) -> dict[str, Any]:
    """Markers (for every chart), SLA thresholds (per query) and the report clock offset.

    The web report draws the same items (static/js/reports.js); see AI/report_markers.
    """
    final_row = next((row for row in llm_rows if str(row.get("domain")) == "final"), {})
    final_row = final_row if isinstance(final_row, dict) else {}
    steps = [step for step in ((step_table or {}).get("steps") or []) if isinstance(step, dict)]
    try:
        markers = report_markers.markers_for(final_row, step_table)
        thresholds = report_markers.thresholds_for(final_row, step_table)
    except Exception:
        logger.exception("Не удалось собрать маркеры графиков для Confluence")
        markers, thresholds = [], []
    return {"markers": markers, "thresholds": thresholds, "clock_offset": report_markers.offset_hours(steps)}


def build_page_body(
    *,
    run_name: str,
    service: str,
    source_url: str,
    llm_rows: list[dict[str, Any]],
    engineer_html: str,
    charts: list[dict[str, str]],
    step_table: dict[str, Any] | None = _UNRESOLVED,
    run_meta: dict[str, Any] | None = None,
) -> str:
    """Storage-format page of the report, in the language the report was written in (``ru`` for older reports).

    ``run_meta`` (``service``, ``start_ms``, ``end_ms``) heads the page of a run without stored report rows,
    which is written in the current locale.
    """
    with with_locale(_page_language(llm_rows)):
        return _page_body(run_name, service, source_url, llm_rows, engineer_html, charts, step_table, run_meta)


def _page_language(llm_rows: list[dict[str, Any]]) -> str:
    """The stored language of the report; a run without report rows has none and takes the current locale."""
    return stored_report_language(llm_rows) if llm_rows else get_locale()


def _llm_domain_title(domain: str, stored_titles: dict[str, str]) -> str:
    if stored_titles.get(domain):
        return stored_titles[domain]
    return t("common.domain.final") if domain == "final" else LLM_DOMAIN_TITLES.get(domain, domain)


def _page_body(
    run_name: str,
    service: str,
    source_url: str,
    llm_rows: list[dict[str, Any]],
    engineer_html: str,
    charts: list[dict[str, str]],
    step_table: Any,
    run_meta: dict[str, Any] | None = None,
) -> str:
    first = llm_rows[0] if llm_rows else (run_meta or {})
    test_type = next((str(row.get("test_type")) for row in llm_rows if row.get("test_type")), "")
    type_label = (test_type_label(normalize_test_type(test_type), get_locale()) or test_type) if test_type else ""
    start_text = _ms_to_text(first.get("start_ms"))
    end_text = _ms_to_text(first.get("end_ms"))
    system_context = {}
    for row in llm_rows:
        ctx = _as_dict(row.get("system_context"))
        if ctx:
            system_context = ctx
            break

    parts = [
        h2(f"LoadLens: {run_name}"),
        kv_table([
            (t("report.hero.service"), service or "—"),
            (t("confluence.header.run"), run_name),
            (t("confluence.header.test_time"), f"{start_text} — {end_text}"),
            *([(t("report.section.test_type"), type_label)] if type_label else []),
        ]),
    ]
    if source_url:
        parts.append(f'<p><a href="{xml_text(source_url)}">{xml_text(t("confluence.open_report"))}</a></p>')
    final_row = next((row for row in llm_rows if str(row.get("domain")) == "final"), {})
    final_row = final_row if isinstance(final_row, dict) else {}
    if is_deterministic_report(final_row.get("parsed")):
        # Built without the model: the verdict and the test figures head the page; there is no analysis section.
        parts.append(h2(t("report.hero.verdict_title")))
        parts.append(render_deterministic_summary(final_row))
    if llm_rows:
        if step_table is _UNRESOLVED:
            step_table = resolve_load_step_table(run_name, llm_rows)
        parts.append(h2(t("report.sla.title")))
        parts.append(sla_criteria_block(final_row, step_table))
    else:
        # A run with metrics and no stored report: charts and the engineer summary only.
        step_table = None
        parts.append(p(t("confluence.no_llm_analysis")))
    parts.append(h2(t("report.engineer.title")))
    parts.append(html_to_storage(engineer_html))
    context_items = _system_context_items(system_context)
    if context_items:
        parts.append(wrap_expand(t("report.context.title"), li_list(context_items)))
    step_block = load_step_expand(step_table)
    if step_block:
        parts.append(step_block)

    by_domain = {str(row.get("domain")): row for row in llm_rows if not is_deterministic_report(row.get("parsed"))}
    stored_titles = report_domain_titles(llm_rows)
    llm_panels: list[tuple[str, str]] = []
    for domain in LLM_DOMAIN_ORDER:
        if domain in by_domain:
            llm_panels.append((_llm_domain_title(domain, stored_titles), render_llm_domain_body(by_domain[domain])))
    for domain, row in by_domain.items():
        if domain not in LLM_DOMAIN_TITLES:
            llm_panels.append((stored_titles.get(domain) or domain, render_llm_domain_body(row)))
    if llm_panels:
        parts.append(h2(t("confluence.llm_analysis")))
        parts.append(wrap_ui_tabs(llm_panels))

    charts_by_domain: dict[str, list[dict[str, str]]] = {}
    for chart in charts:
        charts_by_domain.setdefault(chart["domain"], []).append(chart)
    metric_panels: list[tuple[str, str]] = []
    ordered_domains = [d for d in METRIC_DOMAIN_ORDER if d in charts_by_domain]
    ordered_domains.extend([d for d in charts_by_domain if d not in ordered_domains])
    for domain in ordered_domains:
        body = ""
        for chart in charts_by_domain[domain]:
            body += h3(chart["title"])
            body += attachment_image(chart["filename"])
        metric_panels.append((stored_titles.get(domain) or domain, body or p(t("confluence.no_charts"))))
    parts.append(h2(t("report.charts.section_title")))
    parts.append(wrap_ui_tabs(metric_panels or [("metrics", p(t("confluence.no_run_charts")))]))
    return "".join(parts)


def validate_confluence_config(cfg: dict[str, Any]) -> None:
    """Raises :class:`ConfluenceConfigError` (a ValueError) when publishing is off or a setting is empty."""
    if not bool(cfg.get("enabled")):
        raise ConfluenceConfigError("confluence.error.disabled")
    missing = [key for key in ("base_url", "username", "password", "space_key", "parent_page_id") if not _require_str(cfg.get(key))]
    if missing:
        raise ConfluenceConfigError("confluence.error.missing_settings", fields=", ".join(missing))


def publish_report(
    run_name: str,
    service: str = "",
    source_url: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """Publishes the stored report of ``run_name`` in its own language (see the module docstring)."""
    cfg = confluence_config()
    validate_confluence_config(cfg)
    _progress(progress_callback, msg("confluence.progress.loading_report"), 5)
    llm_rows = load_llm_rows(run_name)
    run_meta: dict[str, Any] | None = None
    if not llm_rows:
        # A run built without the model before its final row was stored: charts and the engineer summary.
        run_meta = load_run_meta(run_name)
        if run_meta is None:
            raise ConfluencePublishError("confluence.error.no_report_data", run=run_name)
    service_name = service or _require_str((llm_rows[0] if llm_rows else run_meta or {}).get("service")) or "unknown"
    engineer_html = load_engineer_html(run_name)
    schema = load_domains_schema(run_name)

    _progress(progress_callback, msg("confluence.progress.preparing_charts"), 20)
    if schema:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ConfluenceDependencyError("confluence.error.matplotlib") from exc
    # Page text, chart captions, markers and thresholds in the language of the report; the job's own
    # messages (progress, errors) stay in the job's locale.
    with with_locale(_page_language(llm_rows)):
        title, body, attachments = _report_page(run_name, service_name, source_url, llm_rows, engineer_html, schema, run_meta)

    client = ConfluenceClient(cfg)
    existing = load_publication(run_name)
    page_id, page_url = upsert_page(client, str((existing or {}).get("page_id") or ""), title, body, attachments, progress_callback)
    save_publication(
        run_name=run_name,
        service=service_name,
        page_id=page_id,
        page_url=page_url,
        space_key=client.space_key,
        parent_page_id=client.parent_page_id,
    )
    _progress(progress_callback, msg("confluence.progress.done"), 100)
    return {"page_id": page_id, "page_url": page_url, "title": title}


def _report_page(
    run_name: str,
    service_name: str,
    source_url: str,
    llm_rows: list[dict[str, Any]],
    engineer_html: str,
    schema: dict[str, list[str]],
    run_meta: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, bytes]]:
    """Title, storage body and PNG attachments of the report page, in the current locale."""
    step_table = resolve_load_step_table(run_name, llm_rows) if llm_rows else None
    annotations = chart_annotations(llm_rows, step_table)
    all_labels = [label for labels in schema.values() for label in labels]
    charts: list[dict[str, str]] = []
    attachments: dict[str, bytes] = {}
    for domain, query_labels in schema.items():
        for query_label in query_labels:
            filename = _safe_filename(f"{domain}-{query_label}") + ".png"
            try:
                png = render_series_png(
                    load_run_series(run_name, domain, query_label),
                    query_label,
                    markers=annotations["markers"],
                    thresholds=report_markers.thresholds_for_query(annotations["thresholds"], query_label, all_labels),
                    clock_offset=annotations["clock_offset"],
                )
            except Exception as exc:
                logger.warning("Failed to render chart %s/%s: %s", domain, query_label, exc)
                continue
            attachments[filename] = png
            charts.append({"domain": domain, "title": query_label, "filename": filename})

    title = f"LoadLens: {run_name} ({service_name})"
    body = build_page_body(
        run_name=run_name,
        service=service_name,
        source_url=source_url,
        llm_rows=llm_rows,
        engineer_html=engineer_html,
        charts=charts,
        step_table=step_table,
        run_meta=run_meta,
    )
    return title, body, attachments


def upsert_page(
    client: ConfluenceClient,
    known_page_id: str,
    title: str,
    body: str,
    attachments: dict[str, bytes],
    progress_callback: ProgressCallback | None,
) -> tuple[str, str]:
    """Updates the known page, or the page with this title, or creates one; returns (page_id, page_url)."""
    page = client.get_page(known_page_id) if known_page_id else None
    if page is None:
        page = client.find_page_by_title(title)
    if page is None:
        _progress(progress_callback, msg("confluence.progress.creating_page"), 55)
        page = client.create_page(title, p(t("confluence.page_placeholder")))
    page_id = str(page.get("id"))

    _progress(progress_callback, msg("confluence.progress.uploading_charts"), 70)
    current_attachments = {str(item.get("title")): str(item.get("id")) for item in client.list_attachments(page_id)}
    for filename, content in attachments.items():
        client.upload_attachment(page_id, filename, content, existing_id=current_attachments.get(filename, ""))

    fresh = client.get_page(page_id) or page
    version = int((_as_dict(fresh.get("version")).get("number") or 1))
    _progress(progress_callback, msg("confluence.progress.updating_page"), 90)
    updated = client.update_page(page_id, title, body, version)
    return str(updated.get("id") or page_id), client.page_url(updated)
