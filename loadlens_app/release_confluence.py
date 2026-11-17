# SPDX-License-Identifier: Apache-2.0
"""A release as its own Confluence page: status, the services with their accepted runs and the deltas to the previous
release, the services not tested yet and a trend picture per service and test type.

A release joins reports of several languages, so the page is written in the locale of the job (the installation
setting), like its progress messages. Links lead to the published Confluence pages of the reports and comparisons
when there are any, else to LoadLens. The data is the one of the release page (``release_pages.release_payload``)
for the project area the publication was started from; publishing never asks the model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from AI.i18n import LocalizedError, fmt_date, fmt_number, get_locale, msg, t
from AI.pipeline import _time_shift_hours
from AI.release_data import TREND_LIMIT, ensure_release_tables, load_publication, save_publication
from AI.test_profile import test_type_label
from AI.verdicts import FAIL, NO_DATA, PASS, RISK, to_code
from AI.verdicts import label as verdict_label
from loadlens_app.compare_confluence import load_compare_publication
from loadlens_app.confluence_export import (
    ConfluenceClient,
    ProgressCallback,
    _safe_filename,
    attachment_image,
    confluence_config,
    figure,
    h2,
    h3,
    li_list,
    load_publications,
    p,
    png,
    status_macro,
    upsert_page,
    validate_confluence_config,
    wrap_expand,
    xml_text,
)
from loadlens_app.core import _ts_conn
from settings import CONFIG

logger = logging.getLogger(__name__)

STATUS_COLOURS = {PASS: "Green", RISK: "Yellow", FAIL: "Red", NO_DATA: "Grey", "incomplete": "Blue"}
TONE_COLOURS = {"ok": "#22c55e", "warn": "#eab308", "fail": "#ef4444", "na": "#9aa0a6"}
VERDICT_TONES = {PASS: "ok", RISK: "warn", FAIL: "fail", NO_DATA: "na"}
METRICS = ("p95", "errors", "capacity")
LINE_COLOUR = "#6c5ce7"


class ReleasePublishError(LocalizedError, ValueError):
    """The release cannot be published (unknown in the project area)."""


@dataclass(frozen=True)
class ReleasePublication:
    release: str
    page_id: str
    page_url: str
    updated_at: Optional[str]


def _progress(callback: ProgressCallback | None, message: Any, pct: int | None = None) -> None:
    if callback:
        callback(message, pct)


# ---- text of the page ----------------------------------------------------------------------------------------


def release_page_title(name: str, project_area: str = "") -> str:
    title = t("confluence.release.page_title", name=name)
    return f"{title} [{project_area}]" if project_area else title


def _short(value: Any) -> str:
    """A number with at most one decimal, the decimal dropped for whole values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    rounded = round(number, 1)
    return fmt_number(rounded, 0 if float(rounded).is_integer() else 1)


def status_text(status: str) -> str:
    if status == "incomplete":
        return t("confluence.release.status_incomplete")
    return verdict_label(to_code(status), _locale())


def _locale() -> str:
    return get_locale()


def delta_text(item: Mapping[str, Any]) -> str:
    """Signed change of a metric: «+12.5 %», «−8 %», «+0.3 pp»."""
    change = item.get("change")
    try:
        number = float(change)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if number > 0 else ("−" if number < 0 else "")
    unit = t("confluence.release.unit_pp") if item.get("unit") == "pp" else "%"
    return f"{sign}{_short(abs(number))} {unit}"


def level_text(deltas: Mapping[str, Any]) -> str:
    """Where the runs were compared: «at 600 RPS», «at the ceiling, 540 RPS»…; '' without a level."""
    keys = {
        "target": "confluence.release.level_target",
        "ceiling": "confluence.release.level_ceiling",
        "hold": "confluence.release.level_hold",
        "spike": "confluence.release.level_spike",
    }
    key = keys.get(str(deltas.get("kind") or ""))
    level = deltas.get("level")
    if level is None or key is None:
        return ""
    return t(key, level=_short(level))


def _reason_text(reason: Optional[str]) -> str:
    keys = {
        "no_previous": "confluence.release.reason_no_previous",
        "no_accepted": "confluence.release.reason_no_accepted",
        "mode_unsupported": "confluence.release.reason_mode_unsupported",
    }
    return t(keys.get(str(reason or ""), "confluence.release.reason_other"))


def _link(url: str, text: str) -> str:
    return f'<a href="{xml_text(url)}">{xml_text(text)}</a>' if url else xml_text(text)


def _kv_html(rows: Sequence[tuple[str, str]]) -> str:
    """Two-column table whose values are storage markup (links, lozenges)."""
    body = "".join(f"<tr><th>{xml_text(label)}</th><td>{value}</td></tr>" for label, value in rows)
    return f"<table><tbody>{body}</tbody></table>"


def _deltas_cells(row: Mapping[str, Any]) -> list[str]:
    deltas = row.get("deltas") or {}
    items = {str(item.get("metric")): item for item in deltas.get("items") or [] if isinstance(item, Mapping)}
    if not deltas.get("available"):
        reason = xml_text(_reason_text(deltas.get("reason")))
        return [reason, "—", "—"]
    cells = []
    for metric in METRICS:
        item = items.get(metric)
        cells.append(xml_text(delta_text(item)) if item else "—")
    level = level_text(deltas)
    if level:
        cells[-1] = f"{cells[-1]}<br/><em>{xml_text(level)}</em>"
    return cells


def services_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_url: str,
    report_pages: Mapping[str, str],
    compare_pages: Mapping[tuple, str],
) -> str:
    """One row per service and test type: run (published report, else LoadLens), verdict, three deltas, previous."""
    head = (
        t("confluence.release.col_service"), t("confluence.release.col_type"), t("confluence.release.col_run"),
        t("confluence.release.col_verdict"), t("confluence.release.col_p95"), t("confluence.release.col_errors"),
        t("confluence.release.col_capacity"), t("confluence.release.col_previous"),
    )
    header = "".join(f"<th>{xml_text(cell)}</th>" for cell in head)
    body = []
    for row in rows:
        run = row.get("run") or {}
        run_name = str(run.get("run_name") or "")
        if run_name:
            url = report_pages.get(run_name) or (f"{base_url}{row.get('report_url')}" if row.get("report_url") else "")
            run_cell = _link(url, run_name)
            if row.get("baseline"):
                run_cell += f" <em>({xml_text(t('confluence.release.baseline'))})</em>"
            verdict = to_code(run.get("verdict"))
            verdict_cell = status_macro(verdict_label(verdict, _locale()), STATUS_COLOURS.get(verdict, "Grey"))
        else:
            run_cell, verdict_cell = "—", "—"
        previous = row.get("previous") or {}
        if previous.get("run_name"):
            pair = (run_name, str(previous.get("run_name")))
            url = compare_pages.get(pair) or (f"{base_url}{row.get('compare_url')}" if row.get("compare_url") else "")
            previous_cell = _link(url, f"{previous.get('release') or '—'} · {previous.get('run_name')}")
        else:
            previous_cell = "—"
        cells = [
            xml_text(row.get("service") or "—"),
            xml_text(test_type_label(row.get("test_type"), _locale()) or row.get("test_type") or "—"),
            run_cell,
            verdict_cell,
            *_deltas_cells(row),
            previous_cell,
        ]
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def trend_filename(service: str, test_type: str) -> str:
    return _safe_filename(f"trend-{service}-{test_type}", "trend") + ".png"


def _method_items() -> list[str]:
    return [
        t("confluence.release.method_accepted"),
        t("confluence.release.method_previous"),
        t("confluence.release.method_level"),
        t("confluence.release.method_status"),
    ]


def release_page_body(
    view: Mapping[str, Any],
    *,
    source_url: str,
    base_url: str,
    generated_text: str,
    report_pages: Mapping[str, str],
    compare_pages: Mapping[tuple, str],
    trend_files: Sequence[tuple[str, str]],
    project_area: str = "",
) -> str:
    """Storage-format page of a release; ``trend_files`` are (caption, attachment name) of the trend pictures."""
    name = str(view.get("release") or "")
    rows = list(view.get("rows") or [])
    status = str(view.get("status") or "")
    tested = len({row.get("service") for row in rows if row.get("run")})
    jira = view.get("jira") or {}
    ticket = str(view.get("ticket") or "")
    header = [
        (t("confluence.release.release"), xml_text(name)),
        (t("confluence.release.status"), status_macro(status_text(status), STATUS_COLOURS.get(status, "Grey"))),
        (t("confluence.release.services"), xml_text(t(
            "confluence.release.services_value", tested=tested, required=int(view.get("services_required") or 0),
        ))),
    ]
    if ticket:
        header.append((t("confluence.release.ticket"), _link(str(jira.get("ticket_url") or ""), ticket)))
    if jira.get("issues_url"):
        header.append((t("confluence.release.jira"), _link(str(jira["issues_url"]), t("confluence.release.jira_issues"))))
    if source_url:
        header.append((t("confluence.release.loadlens"), _link(source_url, t("confluence.release.open_release"))))
    header.append((t("confluence.forecast.generated"), xml_text(generated_text)))
    parts = [h2(release_page_title(name, project_area)), _kv_html(header)]
    note = str(view.get("note") or "").strip()
    if note:
        parts.append(p(note))
    parts.append(h3(t("confluence.release.section_services")))
    if rows:
        parts.append(services_table(rows, base_url=base_url, report_pages=report_pages, compare_pages=compare_pages))
    else:
        parts.append(p(t("confluence.release.no_runs")))
    untested = list(view.get("untested") or [])
    if untested:
        parts.extend([h3(t("confluence.release.section_untested")), li_list(untested)])
    if trend_files:
        parts.append(h3(t("confluence.release.section_trend")))
        for caption, filename in trend_files:
            parts.append(f"<h4>{xml_text(caption)}</h4>{attachment_image(filename)}")
    parts.append(wrap_expand(t("confluence.release.section_method"), li_list(_method_items())))
    return "".join(parts)


# ---- the trend picture ---------------------------------------------------------------------------------------


def render_trend_png(points: Sequence[Mapping[str, Any]], service: str, test_type: str) -> bytes:
    """Three small charts (p95, errors, load) over the release names; points coloured by verdict, the pinned
    baseline ringed."""
    titles = {
        "p95": t("confluence.release.trend_p95"),
        "errors": t("confluence.release.trend_errors"),
        "capacity": t("confluence.release.trend_capacity"),
    }
    plt, fig, axes = figure(12.0, 3.2, 1, 3)
    labels = [str(point.get("release") or "") for point in points]
    xs = list(range(len(points)))
    for ax, metric in zip(axes, METRICS):
        values = [point.get(metric) for point in points]
        known = [(x, float(v)) for x, v in zip(xs, values) if isinstance(v, (int, float))]
        if len(known) > 1:
            ax.plot([x for x, _ in known], [v for _, v in known], color=LINE_COLOUR, linewidth=1.5, zorder=1)
        for x, value in known:
            point = points[x]
            tone = VERDICT_TONES.get(to_code(point.get("verdict")), "na")
            ax.scatter([x], [value], s=36, color=TONE_COLOURS[tone], zorder=2,
                       edgecolors="#15171c" if point.get("baseline") else "none", linewidths=1.6)
        ax.set_title(titles[metric], fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        ax.tick_params(axis="y", labelsize=8)
        if not known:
            ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", color="#9aa0a6")
    fig.suptitle(f"{service} · {test_type_label(test_type, _locale()) or test_type}", fontsize=11)
    return png(plt, fig)


# ---- publications --------------------------------------------------------------------------------------------


def _schema() -> str:
    storage = CONFIG.get("storage") if isinstance(CONFIG.get("storage"), dict) else {}
    cfg = storage.get("timescale") if isinstance(storage.get("timescale"), dict) else {}
    return str(cfg.get("schema") or "public")


def load_release_publication(name: str, project_area: str = "") -> Optional[ReleasePublication]:
    """Confluence page of the release, if it was published."""
    schema = _schema()
    conn = _ts_conn()
    try:
        ensure_release_tables(conn, schema)
        row = load_publication(conn, schema, name, project_area)
    finally:
        conn.close()
    if row is None:
        return None
    return ReleasePublication(release=name, page_id=row["page_id"], page_url=row["page_url"], updated_at=row["updated_at"])


def save_release_publication(
    name: str, page_id: str, page_url: str, space_key: str, parent_page_id: str, project_area: str = "",
) -> None:
    schema = _schema()
    conn = _ts_conn()
    try:
        ensure_release_tables(conn, schema)
        with conn:
            save_publication(conn, schema, name, page_id, page_url, space_key, parent_page_id, project_area)
    finally:
        conn.close()


def _release_parent(cfg: Mapping[str, Any]) -> str:
    """confluence.release_parent_page_id, or the parent page of test reports when it is empty."""
    own = str(cfg.get("release_parent_page_id") or "").strip()
    return own or str(cfg.get("parent_page_id") or "").strip()


def _generated_text() -> str:
    """Local time of the publication: the date in the format of the current locale, then HH:MM."""
    local = datetime.now(timezone.utc) + timedelta(hours=_time_shift_hours())
    return f"{fmt_date(local)} {local.strftime('%H:%M')}"


def _base_url(source_url: str) -> str:
    """Address of LoadLens: the link to the release page without its path."""
    cut = source_url.rfind("/releases/")
    return source_url[:cut] if cut >= 0 else source_url.rstrip("/")


def _compare_pages(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, str]:
    pages: dict[tuple, str] = {}
    for row in rows:
        run = (row.get("run") or {}).get("run_name")
        previous = (row.get("previous") or {}).get("run_name")
        if not run or not previous:
            continue
        try:
            publication = load_compare_publication(str(run), str(previous))
        except Exception:
            logger.warning("Could not read the comparison page of %s and %s", run, previous, exc_info=True)
            continue
        if publication is not None and publication.page_url:
            pages[(str(run), str(previous))] = publication.page_url
    return pages


def _trend_pictures(rows: Sequence[Mapping[str, Any]], area: str) -> tuple[list[tuple[str, str]], dict[str, bytes]]:
    from loadlens_app.blueprints.release_pages import trend_payload

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.warning("matplotlib is not installed: the release page is published without the trend pictures")
        return [], {}
    files: list[tuple[str, str]] = []
    attachments: dict[str, bytes] = {}
    for service, test_type in dict.fromkeys((str(row.get("service")), str(row.get("test_type"))) for row in rows if row.get("run")):
        try:
            points = trend_payload(service, test_type, TREND_LIMIT, area)
            if not points:
                continue
            filename = trend_filename(service, test_type)
            attachments[filename] = render_trend_png(points, service, test_type)
        except Exception:
            logger.warning("Could not draw the trend of %s (%s)", service, test_type, exc_info=True)
            continue
        files.append((f"{service} · {test_type_label(test_type, _locale()) or test_type}", filename))
    return files, attachments


def publish_release(
    name: str,
    project_area: str = "",
    source_url: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """Publishes the release as its own Confluence page (created once, then updated)."""
    from loadlens_app.blueprints.release_pages import release_payload

    cfg = confluence_config()
    validate_confluence_config(cfg)
    _progress(progress_callback, msg("confluence.progress.release"), 10)
    view = release_payload(name, project_area)
    if view is None:
        raise ReleasePublishError("confluence.release.not_found", name=name)
    rows = list(view.get("rows") or [])
    _progress(progress_callback, msg("confluence.progress.preparing_charts"), 30)
    trend_files, attachments = _trend_pictures(rows, project_area)
    report_pages = {
        run_name: str(page.get("page_url") or "")
        for run_name, page in load_publications([(row.get("run") or {}).get("run_name") for row in rows]).items()
        if page.get("page_url")
    }
    title = release_page_title(name, project_area)
    body = release_page_body(
        view,
        source_url=source_url,
        base_url=_base_url(source_url),
        generated_text=_generated_text(),
        report_pages=report_pages,
        compare_pages=_compare_pages(rows),
        trend_files=trend_files,
        project_area=project_area,
    )
    client = ConfluenceClient({**cfg, "parent_page_id": _release_parent(cfg)})
    known = load_release_publication(name, project_area)
    page_id, page_url = upsert_page(client, known.page_id if known else "", title, body, attachments, progress_callback)
    save_release_publication(name, page_id, page_url, client.space_key, client.parent_page_id, project_area)
    _progress(progress_callback, msg("confluence.progress.done"), 100)
    return {"page_id": page_id, "page_url": page_url, "title": title}


__all__ = [
    "ReleasePublication",
    "ReleasePublishError",
    "delta_text",
    "level_text",
    "load_release_publication",
    "publish_release",
    "release_page_body",
    "release_page_title",
    "render_trend_png",
    "save_release_publication",
    "services_table",
    "status_text",
    "trend_filename",
]
