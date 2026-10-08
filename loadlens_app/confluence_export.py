"""Publish a LoadLens report to Confluence as a tabbed page with chart attachments."""

from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable, Iterable
from xml.sax.saxutils import escape as xml_escape

import requests
from psycopg2 import sql

from loadlens_app.core import _ts_conn
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
    "application_logs",
]
LLM_DOMAIN_TITLES = {
    "final": "Итог",
    "jvm": "jvm",
    "database": "database",
    "kafka": "kafka",
    "microservices": "microservices",
    "hard_resources": "hard_resources",
    "lt_framework": "lt_framework",
    "application_logs": "application_logs",
}
METRIC_DOMAIN_ORDER = [
    "jvm",
    "database",
    "kafka",
    "microservices",
    "hard_resources",
    "lt_framework",
]
ALLOWED_HTML_TAGS = {
    "p", "br", "b", "strong", "i", "em", "u", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "a", "span", "div", "pre", "code",
}
VOID_HTML_TAGS = {"br"}
ProgressCallback = Callable[[str, int | None], None]


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
        return "да" if value else "нет"
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
        return "<p>Нет данных.</p>"
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
        if name == "a" and attr_map.get("href"):
            rendered.append(f'href="{xml_text(attr_map["href"])}"')
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
        return "<p>Нет комментария инженера.</p>"
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
        return "после просадки"
    if step.get("dip"):
        return "кратковременная просадка"
    return "нестабильная" if step.get("stable") is False else ""


def _step_name(step: dict[str, Any]) -> str:
    flag = _step_flag(step)
    name = str(step.get("index") or "") + (" SLA" if step.get("selected") else "")
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
    head = "<tr><th>Ступень</th><th>Интервал</th><th>RPS</th><th>Время отклика, p95</th></tr>"
    query = str(table.get("latency_query") or "").strip()
    series = str(table.get("latency_series") or "").strip()
    source_parts = [part for part in (
        f"запрос «{query}»" if query else "",
        f"серия «{series}»" if series else "",
    ) if part]
    source = p("Источник: " + ", ".join(source_parts)) if source_parts else ""
    note = p("RPS — уровень ступени. Время отклика — p95 на плато, без хвоста после просадки.")
    return wrap_expand("Ступени теста", f"<table><thead>{head}</thead><tbody>{body}</tbody></table>{source}{note}")


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


def _standardize_verdict(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return "Недостаточно данных"
    if any(token in value for token in ("успеш", "success", "passed", "ok", "green")):
        return "Успешно"
    if any(token in value for token in ("риск", "warn", "degrad")):
        return "Есть риски"
    if any(token in value for token in ("провал", "fail", "critical", "error", "red")):
        return "Провал"
    if any(token in value for token in ("недостаточно", "no data", "n/a", "unknown")):
        return "Недостаточно данных"
    if raw in {"Успешно", "Есть риски", "Провал", "Недостаточно данных"}:
        return str(raw)
    return str(raw) if str(raw).strip() else "Недостаточно данных"


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
            bits.append(f"значение: {observed}")
        if threshold:
            bits.append(f"порог: {threshold}")
        if note:
            bits.append(note)
        if bits:
            lines.append(" | ".join(bits))
    return lines


UNVERIFIED_SUFFIX = "(не подтверждено данными)"


def _finding_html(item: object) -> str:
    data = _as_dict(item) if not isinstance(item, str) else {"summary": item}
    summary = _cell_text(data.get("summary") or data.get("title") or data.get("text"), "")
    verification = data.get("verification") if isinstance(data.get("verification"), dict) else {}
    if summary and str(verification.get("status") or "").lower() == "unverified" and UNVERIFIED_SUFFIX not in summary:
        summary = f"{summary} {UNVERIFIED_SUFFIX}"
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
        meta.append(f"Критичность: {severity}")
    if component:
        meta.append(f"Компонент: {component}")
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
            _action_stack(recommendations, "Нет рекомендации."),
        ))
    for action in action_entries:
        dangling = (
            (not action["for_finding_ids"] and action["idx"] in unused_legacy)
            or (action["for_finding_ids"] and not any(item_id in linked_ids for item_id in action["for_finding_ids"]))
        )
        if dangling:
            rows.append((
                _empty_cell("Дополнительная рекомендация"),
                _action_stack([action["item"]], "Нет рекомендации."),
            ))
    if not rows:
        rows.append((
            _empty_cell("Нет существенных проблем."),
            _action_stack([entry["item"] for entry in action_entries], "Нет рекомендаций."),
        ))
    return rows


def render_llm_domain_body(row: dict[str, Any]) -> str:
    parsed = _as_dict(row.get("parsed"))
    analysis_error = _as_dict(parsed.get("analysis_error"))
    if analysis_error:
        message = _cell_text(analysis_error.get("message"), "Анализ ИИ не выполнен.")
        details = [
            _cell_text(analysis_error.get("provider"), ""),
            f"лимит {analysis_error.get('max_tokens')}" if analysis_error.get("max_tokens") else "",
            f"выдано {analysis_error.get('output_tokens')}" if analysis_error.get("output_tokens") is not None else "",
        ]
        details = [item for item in details if item]
        body = h3("Анализ ИИ не выполнен") + p(message)
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
        return f"<pre>{xml_text(text[:12000])}</pre>" if text else p("Нет данных.")

    verdict = _standardize_verdict(row.get("sla_verdict") or parsed.get("verdict") or row.get("verdict"))
    rows = [("Вердикт", verdict)]
    if row.get("domain") == "final" and row.get("llm_verdict") and row.get("sla_verdict"):
        rows.append(("Вердикт LLM", _standardize_verdict(row.get("llm_verdict") or parsed.get("verdict"))))
        rows.append(("SLA", _standardize_verdict(row.get("sla_verdict"))))
    parts = [kv_table(rows)]

    rationale = _cell_text(
        parsed.get("verdict_rationale") or parsed.get("verdict_reason") or parsed.get("rationale"),
        "",
    )
    if rationale:
        parts.append(h3("Обоснование вердикта"))
        parts.append(_html_paragraphs(rationale))

    peak = _as_dict(parsed.get("peak_performance") or parsed.get("peak_perfomance"))
    if _peak_table_applies(str(row.get("domain") or ""), peak):
        parts.append(h3("Пиковая производительность"))
        parts.append(kv_table([
            ("Максимальный RPS", _cell_text(peak.get("max_rps"))),
            ("Время пиковой производительности", _cell_text(peak.get("max_time"))),
            ("Время деградации", _cell_text(peak.get("drop_time"))),
        ]))

    parts.append(h3("Проблемы и рекомендации"))
    parts.append(two_col_table(
        "Проблемы",
        "Рекомендации по устранению",
        pair_findings_with_actions(
            _as_list(parsed.get("findings")),
            _as_list(parsed.get("recommended_actions") or parsed.get("actions")),
        ),
    ))

    affected = [_cell_text(item, "") for item in _as_list(parsed.get("affected_components"))]
    affected = [item for item in affected if item]
    if affected:
        parts.append(h3("Затронутые компоненты"))
        parts.append(p(", ".join(affected)))
    return "".join(parts)


def _system_context_items(ctx: dict[str, Any]) -> list[str]:
    system = _as_dict(ctx.get("system"))
    architecture = _as_dict(ctx.get("architecture"))
    load_model = _as_dict(ctx.get("load_model"))
    operational = _as_dict(ctx.get("operational_context"))
    items: list[str] = []
    if _require_str(system.get("name")):
        items.append(f"Система: {_require_str(system.get('name'))}")
    if _require_str(system.get("domain")):
        items.append(f"Домен: {_require_str(system.get('domain'))}")
    if _require_str(system.get("description")):
        items.append(_require_str(system.get("description")))
    if _require_str(system.get("test_goal")):
        items.append(f"Цель теста: {_require_str(system.get('test_goal'))}")
    if _require_str(architecture.get("style")):
        items.append(f"Архитектура: {_require_str(architecture.get('style'))}")
    for component in _as_list(architecture.get("components"))[:20]:
        data = _as_dict(component)
        name = _require_str(data.get("name") or data.get("id"))
        if name:
            items.append(f"Компонент: {name}")
    for flow in _as_list(load_model.get("critical_user_flows"))[:10]:
        data = _as_dict(flow)
        name = _require_str(data.get("name") or data.get("id"))
        if name:
            items.append(f"Критичный поток: {name}")
    for risk in _as_list(operational.get("known_risks"))[:10]:
        if _require_str(risk):
            items.append(f"Риск: {_require_str(risk)}")
    return items


def _hash_color(label: str) -> str:
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def render_series_png(points: list[dict[str, Any]], title: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: list[str] = []
    series_map: dict[str, dict[str, float]] = {}
    for point in points:
        ts = str(point.get("t") or "")
        series = str(point.get("series") or "series")
        try:
            value = float(point.get("value"))
        except (TypeError, ValueError):
            continue
        if ts not in labels:
            labels.append(ts)
        series_map.setdefault(series, {})[ts] = value
    labels.sort()
    ranked = sorted(
        series_map.items(),
        key=lambda item: abs(sum(item[1].values()) / max(len(item[1]), 1)),
        reverse=True,
    )[:15]

    fig, ax = plt.subplots(figsize=(12.0, 4.6), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f7f7")
    ax.set_title(title)
    ax.set_xlabel("Время")
    ax.grid(True, color="#dddddd")
    if not labels or not ranked:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
    else:
        xticks = labels[:: max(1, len(labels) // 8)]
        for name, values in ranked:
            ax.plot(
                labels,
                [values.get(ts) for ts in labels],
                label=name[:60],
                color=_hash_color(name),
                linewidth=1.6,
            )
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [ts.replace("T", " ")[11:16] if "T" in ts else ts[-5:] for ts in xticks],
            rotation=0,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _progress(callback: ProgressCallback | None, message: str, pct: int | None = None) -> None:
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
                    scores, sla_verdict, sla_details, system_context, created_at
                FROM {schema}.{table}
                WHERE run_name = %s AND domain <> 'engineer'
                ORDER BY domain, created_at DESC
                """,
                (run_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    data: list[dict[str, Any]] = []
    for row in rows:
        data.append({
            "run_name": row[0],
            "service": row[1],
            "start_ms": int(row[2]) if row[2] is not None else None,
            "end_ms": int(row[3]) if row[3] is not None else None,
            "domain": row[4],
            "verdict": row[5],
            "llm_verdict": row[5],
            "text": row[6],
            "parsed": row[7],
            "scores": row[8],
            "sla_verdict": row[9],
            "sla_details": row[10],
            "system_context": row[11],
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


def build_page_body(
    *,
    run_name: str,
    service: str,
    source_url: str,
    llm_rows: list[dict[str, Any]],
    engineer_html: str,
    charts: list[dict[str, str]],
) -> str:
    first = llm_rows[0] if llm_rows else {}
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
            ("Сервис", service or "—"),
            ("Запуск", run_name),
            ("Время теста", f"{start_text} — {end_text}"),
        ]),
    ]
    if source_url:
        parts.append(f'<p><a href="{xml_text(source_url)}">Открыть отчёт в LoadLens</a></p>')
    parts.append(h2("Итоги от инженера"))
    parts.append(html_to_storage(engineer_html))
    context_items = _system_context_items(system_context)
    if context_items:
        parts.append(wrap_expand("Контекст тестируемой системы", li_list(context_items)))
    step_block = load_step_expand(resolve_load_step_table(run_name, llm_rows))
    if step_block:
        parts.append(step_block)

    by_domain = {str(row.get("domain")): row for row in llm_rows}
    llm_panels: list[tuple[str, str]] = []
    for domain in LLM_DOMAIN_ORDER:
        if domain in by_domain:
            llm_panels.append((LLM_DOMAIN_TITLES.get(domain, domain), render_llm_domain_body(by_domain[domain])))
    for domain, row in by_domain.items():
        if domain not in LLM_DOMAIN_TITLES:
            llm_panels.append((domain, render_llm_domain_body(row)))
    parts.append(h2("Анализ LLM"))
    parts.append(wrap_ui_tabs(llm_panels or [("Итог", p("Нет LLM-отчётов."))]))

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
        metric_panels.append((domain, body or p("Нет графиков.")))
    parts.append(h2("Графики метрик"))
    parts.append(wrap_ui_tabs(metric_panels or [("metrics", p("Нет графиков для этого запуска."))]))
    return "".join(parts)


def validate_confluence_config(cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("enabled")):
        raise ValueError("Публикация в Confluence выключена: установите confluence.enabled=true")
    missing = [key for key in ("base_url", "username", "password", "space_key", "parent_page_id") if not _require_str(cfg.get(key))]
    if missing:
        raise ValueError(f"Не заполнены настройки Confluence: {', '.join(missing)}")


def publish_report(
    run_name: str,
    service: str = "",
    source_url: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    cfg = confluence_config()
    validate_confluence_config(cfg)
    _progress(progress_callback, "Загрузка отчёта…", 5)
    llm_rows = load_llm_rows(run_name)
    if not llm_rows:
        raise ValueError(f"Для запуска '{run_name}' нет LLM-отчётов")
    service_name = service or _require_str(llm_rows[0].get("service")) or "unknown"
    engineer_html = load_engineer_html(run_name)
    schema = load_domains_schema(run_name)

    _progress(progress_callback, "Подготовка графиков…", 20)
    if schema:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Для публикации графиков в Confluence нужен пакет matplotlib. Установите: pip install matplotlib"
            ) from exc
    charts: list[dict[str, str]] = []
    attachments: dict[str, bytes] = {}
    for domain, query_labels in schema.items():
        for query_label in query_labels:
            filename = _safe_filename(f"{domain}-{query_label}") + ".png"
            try:
                png = render_series_png(load_run_series(run_name, domain, query_label), query_label)
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
    )

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
    _progress(progress_callback, "Готово", 100)
    return {"page_id": page_id, "page_url": page_url, "title": title}


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
        _progress(progress_callback, "Создание страницы Confluence…", 55)
        page = client.create_page(title, "<p>Публикация отчёта LoadLens…</p>")
    page_id = str(page.get("id"))

    _progress(progress_callback, "Загрузка графиков…", 70)
    current_attachments = {str(item.get("title")): str(item.get("id")) for item in client.list_attachments(page_id)}
    for filename, content in attachments.items():
        client.upload_attachment(page_id, filename, content, existing_id=current_attachments.get(filename, ""))

    fresh = client.get_page(page_id) or page
    version = int((_as_dict(fresh.get("version")).get("number") or 1))
    _progress(progress_callback, "Обновление содержимого страницы…", 90)
    updated = client.update_page(page_id, title, body, version)
    return str(updated.get("id") or page_id), client.page_url(updated)
