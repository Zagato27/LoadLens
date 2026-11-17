# SPDX-License-Identifier: Apache-2.0
"""Releases as a whole: attributes of a release, the release page rows, the release list and the trend by release.

Marks (the release of a run, the accepted run of a pair «release × service × test type», the pinned baseline) live
in ``release_marks`` (AI/release_marks.py). This module reads them together with the final rows of ``llm_reports``
and keeps two tables of its own:

- ``releases (name, ticket, note, author, updated_at)``: what an engineer adds to a release (the Jira ticket, a note).
  A row exists only after the first change; the release list is the names of the marks plus these rows.
- ``release_publications (release, project_area, page_id, page_url, space_key, parent_page_id, updated_at)``: the
  Confluence page of a release in one project area (loadlens_app/release_confluence.py).

The deltas of the release page are computed live with AI/regression.py (``compare_stored``) against the accepted run
of the previous release; ``sla_details.regression`` of a report keeps the baseline it had when it was built
(``stored_regression``). The trend takes the values of each accepted run at its own level (``comparison_point``).

Status of a release: ``incomplete`` while a required service (a service of the project area) has no accepted run,
else the worst verdict of its accepted runs: fail > risk > no_data > pass. No Flask here: the routes live in
loadlens_app/blueprints/release_pages.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from psycopg2 import sql

from AI.db_store import effective_verdict_sql, manual_verdicts_cte, manual_verdicts_ready
from AI.regression import compare_stored, comparison_point
from AI.release_marks import RELEASE_MARKS_TABLE, mark_test_type, previous_release_run, release_marks_ready
from AI.verdicts import FAIL, NO_DATA, PASS, RISK, optional_code, to_code

logger = logging.getLogger(__name__)

RELEASES_TABLE = "releases"
RELEASE_PUBLICATIONS_TABLE = "release_publications"
RELEASE_NOTE_MAX = 1000
TREND_LIMIT = 12
TREND_LIMIT_MAX = 50
STATUS_INCOMPLETE = "incomplete"
# Worst first: the status of a complete release is the first of these its accepted runs have.
STATUS_ORDER = (FAIL, RISK, NO_DATA, PASS)
STATUSES = (STATUS_INCOMPLETE,) + STATUS_ORDER
_ENSURED: set[str] = set()

ThresholdsOf = Callable[[str], Optional[Mapping[str, Any]]]


# ---- small helpers -------------------------------------------------------------------------------------------


def _ref(schema: str, table: str) -> sql.Identifier:
    return sql.Identifier(str(schema or "public"), table)


def iso(value: Any) -> Optional[str]:
    """ISO text of a timestamp; other values pass as text, None stays None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_time(value: Any) -> str:
    """Sort key of a timestamp: ISO text (all values of one column have the same form)."""
    return iso(value) or ""


def release_status(verdicts: Iterable[Any], untested: Sequence[str] = ()) -> str:
    """``incomplete`` while a required service has no accepted run, else the worst verdict (no runs: ``no_data``)."""
    if untested:
        return STATUS_INCOMPLETE
    codes = {to_code(value) for value in verdicts}
    for code in STATUS_ORDER:
        if code in codes:
            return code
    return NO_DATA


# ---- tables of this module -----------------------------------------------------------------------------------


def ensure_release_tables(conn: Any, schema: str) -> None:
    """Creates and upgrades release tables before any query on ``conn``.

    Legacy publication rows have an unknown area and remain in the empty (global) scope. Never reuse one for
    a particular area: publishing that area would overwrite a page that may belong to another one.
    """
    schema = str(schema or "public")
    if schema in _ENSURED:
        return
    previous = getattr(conn, "autocommit", False)
    conn.autocommit = False
    try:
        with conn, conn.cursor() as cur:
            # Serialize the DDL across web processes and workers, including first creation of the tables.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"loadlens.release_tables.{schema}",))
            cur.execute(sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    name       TEXT        PRIMARY KEY,
                    ticket     TEXT        NOT NULL DEFAULT '',
                    note       TEXT        NOT NULL DEFAULT '',
                    author     TEXT        NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(_ref(schema, RELEASES_TABLE)))
            cur.execute(sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    release        TEXT        NOT NULL,
                    project_area   TEXT        NOT NULL DEFAULT '',
                    page_id        TEXT        NOT NULL,
                    page_url       TEXT,
                    space_key      TEXT,
                    parent_page_id TEXT,
                    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (release, project_area)
                )
                """
            ).format(_ref(schema, RELEASE_PUBLICATIONS_TABLE)))
            publications = _ref(schema, RELEASE_PUBLICATIONS_TABLE)
            cur.execute(sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS project_area TEXT NOT NULL DEFAULT ''").format(
                publications,
            ))
            cur.execute(
                "SELECT c.conname, array_agg(a.attname ORDER BY k.ordinality) "
                "FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
                "JOIN pg_namespace n ON n.oid = t.relnamespace "
                "CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality) "
                "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum "
                "WHERE n.nspname = %s AND t.relname = %s AND c.contype = 'p' GROUP BY c.conname",
                (schema, RELEASE_PUBLICATIONS_TABLE),
            )
            primary_key = cur.fetchone()
            if primary_key is None:
                cur.execute(sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (release, project_area)").format(publications))
            elif list(primary_key[1]) != ["release", "project_area"]:
                cur.execute(sql.SQL(
                    "ALTER TABLE {} DROP CONSTRAINT {}, ADD PRIMARY KEY (release, project_area)"
                ).format(publications, sql.Identifier(primary_key[0])))
    finally:
        try:
            conn.autocommit = previous
        except Exception:
            pass
    _ENSURED.add(schema)


def _release_dict(row: Sequence[Any]) -> Dict[str, Any]:
    name, ticket, note, author, updated_at = row[:5]
    return {
        "release": str(name),
        "ticket": str(ticket or ""),
        "note": str(note or ""),
        "author": str(author or ""),
        "updated_at": iso(updated_at),
    }


def load_release(conn: Any, schema: str, name: str) -> Optional[Dict[str, Any]]:
    """Attributes of a release (``{release, ticket, note, author, updated_at}``), None when it has no row."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT name, ticket, note, author, updated_at FROM {} WHERE name = %s").format(_ref(schema, RELEASES_TABLE)),
            (name,),
        )
        row = cur.fetchone()
    return _release_dict(row) if row else None


def list_release_rows(conn: Any, schema: str) -> List[Dict[str, Any]]:
    """Every row of ``releases``."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT name, ticket, note, author, updated_at FROM {}").format(_ref(schema, RELEASES_TABLE)))
        rows = cur.fetchall()
    return [_release_dict(row) for row in rows]


def save_release(
    conn: Any,
    schema: str,
    name: str,
    *,
    ticket: Optional[str] = None,
    note: Optional[str] = None,
    author: str = "",
) -> Dict[str, Any]:
    """Creates or changes the row of a release; ``None`` keeps the stored value. The caller validates and commits."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {} AS rel (name, ticket, note, author, updated_at)
                VALUES (%s, COALESCE(%s, ''), COALESCE(%s, ''), %s, now())
                ON CONFLICT (name) DO UPDATE SET
                    ticket = COALESCE(%s, rel.ticket), note = COALESCE(%s, rel.note),
                    author = EXCLUDED.author, updated_at = now()
                RETURNING name, ticket, note, author, updated_at
                """
            ).format(_ref(schema, RELEASES_TABLE)),
            (name, ticket, note, str(author or ""), ticket, note),
        )
        row = cur.fetchone()
    return _release_dict(row)


def delete_release(conn: Any, schema: str, name: str) -> bool:
    """Deletes the row of a release and the record of its Confluence page (not the marks of its runs).

    True when there was something to delete. The caller commits.
    """
    deleted = 0
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {} WHERE name = %s").format(_ref(schema, RELEASES_TABLE)), (name,))
        deleted += max(cur.rowcount or 0, 0)
        cur.execute(sql.SQL("DELETE FROM {} WHERE release = %s").format(_ref(schema, RELEASE_PUBLICATIONS_TABLE)), (name,))
        deleted += max(cur.rowcount or 0, 0)
    return deleted > 0


def load_publication(conn: Any, schema: str, name: str, project_area: str = "") -> Optional[Dict[str, Any]]:
    """Confluence page of a release in the requested area only (no fallback to another area's page)."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT page_id, page_url, space_key, parent_page_id, updated_at FROM {} "
                "WHERE release = %s AND project_area = %s"
            ).format(_ref(schema, RELEASE_PUBLICATIONS_TABLE)),
            (name, project_area or ""),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "page_id": str(row[0]),
        "page_url": str(row[1] or ""),
        "space_key": str(row[2] or ""),
        "parent_page_id": str(row[3] or ""),
        "updated_at": iso(row[4]),
    }


def save_publication(
    conn: Any, schema: str, name: str, page_id: str, page_url: str, space_key: str, parent_page_id: str,
    project_area: str = "",
) -> None:
    """Records the Confluence page of a release. The caller commits."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {} (release, project_area, page_id, page_url, space_key, parent_page_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (release, project_area) DO UPDATE SET
                    page_id = EXCLUDED.page_id, page_url = EXCLUDED.page_url, space_key = EXCLUDED.space_key,
                    parent_page_id = EXCLUDED.parent_page_id, updated_at = now()
                """
            ).format(_ref(schema, RELEASE_PUBLICATIONS_TABLE)),
            (name, project_area or "", page_id, page_url, space_key, parent_page_id),
        )


# ---- marked runs ---------------------------------------------------------------------------------------------


def marked_runs(
    conn: Any,
    schema: str,
    llm_table: str,
    *,
    release: Optional[str] = None,
    services: Optional[Sequence[str]] = None,
    service: Optional[str] = None,
    test_type: Optional[str] = None,
    accepted_only: bool = False,
    limit: Optional[int] = None,
    with_details: bool = False,
) -> List[Dict[str, Any]]:
    """Runs with a release mark joined with their latest final row, newest report first.

    Each item: ``release, run_name, run_id, service, test_type, created_at, verdict, manual_verdict, max_rps,
    accepted, baseline`` (+ ``sla_details`` with ``with_details``); verdicts are codes, ``verdict`` the one the run
    shows (manual, else SLA, else the stored one). ``services`` limits the runs to the project area (None: any).
    Runs whose final report is gone are skipped. Empty without the marks table.
    """
    if not release_marks_ready(conn, schema):
        return []
    filters = [sql.SQL("m.release <> ''")]
    params: List[Any] = []
    if release is not None:
        filters.append(sql.SQL("m.release = %s"))
        params.append(release)
    if accepted_only:
        filters.append(sql.SQL("m.accepted"))
    if services is not None:
        filters.append(sql.SQL("m.service = ANY(%s)"))
        params.append([str(item) for item in services])
    if service is not None:
        filters.append(sql.SQL("m.service = %s"))
        params.append(service)
    if test_type is not None:
        filters.append(sql.SQL("m.test_type = %s"))
        params.append(mark_test_type(test_type))
    where = sql.SQL(" AND ").join(filters)
    details = sql.SQL(", f.sla_details") if with_details else sql.SQL("")
    limit_sql = sql.SQL(" LIMIT %s") if limit else sql.SQL("")
    query = sql.SQL(
        """
        WITH {manual_cte},
        marks AS (SELECT * FROM {marks} m WHERE {where}),
        final AS (
            SELECT DISTINCT ON (r.run_name) r.run_name, r.run_id, r.created_at, r.verdict, r.sla_verdict, r.max_rps,
                   r.sla_details
            FROM {llm} r JOIN marks m ON m.run_name = r.run_name
            WHERE r.domain = 'final'
            ORDER BY r.run_name, r.created_at DESC
        )
        SELECT m.release, m.run_name, COALESCE(NULLIF(f.run_id, ''), m.run_id, ''), m.service, m.test_type,
               f.created_at, {verdict}, mv.verdict, f.max_rps, m.accepted, m.baseline{details}
        FROM marks m
        JOIN final f ON f.run_name = m.run_name
        LEFT JOIN manual_verdicts mv ON mv.run_name = m.run_name
        ORDER BY f.created_at DESC, m.run_name{limit}
        """
    ).format(
        manual_cte=sql.SQL(manual_verdicts_cte(schema, manual_verdicts_ready(conn, schema))),
        marks=_ref(schema, RELEASE_MARKS_TABLE),
        where=where,
        llm=_ref(schema, llm_table),
        verdict=sql.SQL(effective_verdict_sql("f")),
        details=details,
        limit=limit_sql,
    )
    if limit:
        params.append(int(limit))
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
    out = []
    for row in rows:
        item = {
            "release": str(row[0] or ""),
            "run_name": str(row[1] or ""),
            "run_id": str(row[2] or ""),
            "service": str(row[3] or ""),
            "test_type": str(row[4] or ""),
            "created_at": row[5],
            "verdict": to_code(row[6]),
            "manual_verdict": optional_code(row[7]),
            "max_rps": _number(row[8]),
            "accepted": bool(row[9]),
            "baseline": bool(row[10]),
        }
        if with_details:
            item["sla_details"] = row[11] if isinstance(row[11], dict) else {}
        out.append(item)
    return out


def final_details(conn: Any, schema: str, llm_table: str, run_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """``{run_name: sla_details}`` of the latest final rows of the runs (``{}`` for a row without details)."""
    names = sorted({str(name) for name in run_names if name})
    if not names:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT DISTINCT ON (run_name) run_name, sla_details FROM {} "
                "WHERE domain = 'final' AND run_name = ANY(%s) ORDER BY run_name, created_at DESC"
            ).format(_ref(schema, llm_table)),
            (names,),
        )
        rows = cur.fetchall()
    return {str(row[0]): (row[1] if isinstance(row[1], dict) else {}) for row in rows}


# ---- the release list ----------------------------------------------------------------------------------------


def overview_items(
    accepted: Sequence[Mapping[str, Any]],
    attributes: Sequence[Mapping[str, Any]],
    required: Sequence[str],
) -> List[Dict[str, Any]]:
    """Items of the release list from the accepted runs and the rows of ``releases``, newest release first.

    Item: ``{release, status, services_tested, services_required, verdicts: {pass, risk, fail, no_data},
    last_created_at, ticket}``; releases without accepted runs (only a row) follow, newest row first.
    """
    by_release: Dict[str, List[Mapping[str, Any]]] = {}
    for run in accepted:
        by_release.setdefault(str(run.get("release") or ""), []).append(run)
    rows = {str(row.get("release")): row for row in attributes}
    names = [name for name in dict.fromkeys(list(by_release) + list(rows)) if name]
    required_set = [str(item) for item in dict.fromkeys(required)]
    items = []
    for name in names:
        runs = by_release.get(name, [])
        tested = {str(run.get("service") or "") for run in runs}
        untested = [service for service in required_set if service not in tested]
        counts = {code: 0 for code in (PASS, RISK, FAIL, NO_DATA)}
        for run in runs:
            counts[to_code(run.get("verdict"))] += 1
        last = max((run.get("created_at") for run in runs if run.get("created_at") is not None), default=None, key=_sort_time)
        row = rows.get(name) or {}
        items.append({
            "release": name,
            "status": release_status([run.get("verdict") for run in runs], untested),
            "services_tested": len(tested),
            "services_required": len(required_set),
            "verdicts": counts,
            "last_created_at": iso(last),
            "ticket": str(row.get("ticket") or ""),
            "_updated_at": str(row.get("updated_at") or ""),
        })
    items.sort(key=lambda item: (item["last_created_at"] is not None, item["last_created_at"] or item["_updated_at"]), reverse=True)
    for item in items:
        item.pop("_updated_at", None)
    return items


def release_overview(
    conn: Any, schema: str, llm_table: str, *, services: Optional[Sequence[str]], required: Sequence[str],
) -> List[Dict[str, Any]]:
    """The release list of the project area (see :func:`overview_items`)."""
    accepted = marked_runs(conn, schema, llm_table, services=services, accepted_only=True)
    return overview_items(accepted, list_release_rows(conn, schema), required)


# ---- the release page ----------------------------------------------------------------------------------------


def _run_view(run: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_name": run.get("run_name"),
        "run_id": run.get("run_id") or "",
        "created_at": iso(run.get("created_at")),
        "verdict": to_code(run.get("verdict")),
        "manual_verdict": run.get("manual_verdict"),
        "max_rps": run.get("max_rps"),
    }


def _previous_view(previous: Any) -> Optional[Dict[str, Any]]:
    if previous is None:
        return None
    return {
        "release": str(getattr(previous, "release", "") or ""),
        "run_name": str(getattr(previous, "run_name", "") or ""),
        "run_id": str(getattr(previous, "run_id", "") or ""),
        "created_at": iso(getattr(previous, "created_at", None)),
        "verdict": to_code(getattr(previous, "verdict", None)),
    }


def report_url(run_name: str, run_id: str = "") -> str:
    """Web report of a run, by run_id when known (the same rule as ``LL.reportUrl``)."""
    from urllib.parse import quote

    key = run_id or run_name
    return f"/reports/{quote(str(key), safe='')}" if key else "/reports"


def compare_url(run_a: str, run_b: str) -> str:
    """Comparison of A (assessed) with B (baseline)."""
    from urllib.parse import urlencode

    return "/compare?" + urlencode({"run_a": run_a, "run_b": run_b})


def _no_deltas(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason, "kind": None, "level": None, "step_label": None, "items": [], "skipped": []}


def build_rows(
    marks: Sequence[Mapping[str, Any]],
    *,
    previous_of: Callable[[Mapping[str, Any]], Any],
    details: Mapping[str, Mapping[str, Any]],
    thresholds_of: ThresholdsOf,
) -> List[Dict[str, Any]]:
    """Rows of the release page, one per pair «service × test type» with a mark in the release.

    ``marks`` are the marked runs of the release (any acceptance), ``previous_of(accepted_run)`` gives the accepted
    run of the previous release (a ``BaselineRun`` or None), ``details`` the ``sla_details`` by run name and
    ``thresholds_of(service)`` the regression thresholds of the service's effective SLA (None: no colouring).
    """
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for mark in marks:
        groups.setdefault((str(mark.get("service") or ""), str(mark.get("test_type") or "")), []).append(mark)
    rows = []
    for (service, test_type), runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda run: _sort_time(run.get("created_at")), reverse=True)
        accepted = next((run for run in runs if run.get("accepted")), None)
        previous = previous_of(accepted) if accepted is not None else None
        if accepted is None:
            deltas = _no_deltas("no_accepted")
        elif previous is None:
            deltas = _no_deltas("no_previous")
        else:
            current_details = details.get(str(accepted.get("run_name"))) or {}
            baseline_details = details.get(str(previous.run_name)) or {}
            try:
                deltas = compare_stored(current_details, baseline_details, test_type, thresholds_of(service))
            except Exception:  # a report the comparison cannot read must not break the page
                logger.warning("Release deltas of %s against %s failed", accepted.get("run_name"), previous.run_name, exc_info=True)
                deltas = _no_deltas("error")
        stored = None
        if accepted is not None:
            regression = (details.get(str(accepted.get("run_name"))) or {}).get("regression")
            stored = regression if isinstance(regression, dict) else None
        rows.append({
            "service": service,
            "test_type": test_type,
            "run": _run_view(accepted) if accepted is not None else None,
            "attempts": [
                {
                    "run_name": run.get("run_name"),
                    "run_id": run.get("run_id") or "",
                    "created_at": iso(run.get("created_at")),
                    "verdict": to_code(run.get("verdict")),
                    "accepted": bool(run.get("accepted")),
                }
                for run in runs
            ],
            "previous": _previous_view(previous),
            "deltas": deltas,
            "stored_regression": stored,
            "report_url": report_url(str(accepted.get("run_name")), str(accepted.get("run_id") or "")) if accepted else None,
            "compare_url": compare_url(str(accepted.get("run_name")), str(previous.run_name)) if accepted and previous else None,
            "baseline": bool(accepted.get("baseline")) if accepted else False,
        })
    return rows


def release_view(
    conn: Any,
    schema: str,
    llm_table: str,
    release: str,
    *,
    services: Optional[Sequence[str]],
    required: Sequence[str],
    thresholds_of: ThresholdsOf,
    project_area: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """``{release, ticket, note, author, updated_at, status, rows, untested}`` of a release; None when it has
    neither a mark in the area nor a row in ``releases``."""
    attributes = load_release(conn, schema, release)
    marks = marked_runs(conn, schema, llm_table, release=release, services=services)
    if not marks and attributes is None:
        return None

    def previous_of(run: Mapping[str, Any]) -> Any:
        return previous_release_run(
            conn, schema, llm_table,
            service=str(run.get("service") or ""),
            test_type=str(run.get("test_type") or ""),
            before=run.get("created_at"),
            exclude_run_name=str(run.get("run_name") or ""),
            exclude_release=release,
            project_area=project_area,
        )

    previous_cache: Dict[str, Any] = {}

    def cached_previous(run: Mapping[str, Any]) -> Any:
        key = str(run.get("run_name"))
        if key not in previous_cache:
            previous_cache[key] = previous_of(run)
        return previous_cache[key]

    accepted = [run for run in marks if run.get("accepted")]
    for run in accepted:
        cached_previous(run)
    names = [str(run.get("run_name")) for run in accepted]
    names += [str(previous.run_name) for previous in previous_cache.values() if previous is not None]
    details = final_details(conn, schema, llm_table, names)
    rows = build_rows(marks, previous_of=cached_previous, details=details, thresholds_of=thresholds_of)
    tested = {row["service"] for row in rows if row["run"] is not None}
    untested = [service for service in dict.fromkeys(required) if service not in tested]
    base = attributes or {"release": release, "ticket": "", "note": "", "author": "", "updated_at": None}
    return {
        **base,
        "status": release_status([row["run"]["verdict"] for row in rows if row["run"] is not None], untested),
        "rows": rows,
        "untested": untested,
    }


# ---- the trend by release ------------------------------------------------------------------------------------


def trend_point(run: Mapping[str, Any], test_type: str) -> Dict[str, Any]:
    """Point of the trend: the run's values at its own level (target step, else the ceiling).

    A step run on which no step holds the latency and error thresholds has no ceiling: its values are taken at the
    level its SLA checks were measured on (``stable_window``), as ``comparable_level`` does for a pair of runs.
    """
    details = run.get("sla_details") or {}
    try:
        point = comparison_point(details, test_type) or {}
        window = details.get("stable_window") if isinstance(details, Mapping) else None
        measured = _number(window.get("level")) if isinstance(window, Mapping) else None
        if point.get("level") is None and measured is not None and point.get("mode") not in ("stability", "spike"):
            point = comparison_point(details, test_type, measured, kind="ceiling") or point
    except Exception:
        logger.warning("Trend point of %s failed", run.get("run_name"), exc_info=True)
        point = {}
    return {
        "release": run.get("release"),
        "run_name": run.get("run_name"),
        "run_id": run.get("run_id") or "",
        "created_at": iso(run.get("created_at")),
        "verdict": to_code(run.get("verdict")),
        "kind": point.get("kind"),
        "level": _number(point.get("level")),
        "p95": _number(point.get("p95")),
        "errors": _number(point.get("errors")),
        "capacity": _number(point.get("capacity")),
        "baseline": bool(run.get("baseline")),
    }


def trend_points(
    conn: Any, schema: str, llm_table: str, service: str, test_type: str, limit: int = TREND_LIMIT,
) -> List[Dict[str, Any]]:
    """Accepted runs of a service and test type over the releases, the last ``limit`` in report order."""
    normalized = mark_test_type(test_type)
    runs = marked_runs(
        conn, schema, llm_table, service=service, test_type=normalized, accepted_only=True, limit=limit, with_details=True,
    )
    return [trend_point(run, normalized) for run in reversed(runs)]


__all__ = [
    "RELEASES_TABLE",
    "RELEASE_NOTE_MAX",
    "RELEASE_PUBLICATIONS_TABLE",
    "STATUSES",
    "STATUS_INCOMPLETE",
    "TREND_LIMIT",
    "TREND_LIMIT_MAX",
    "build_rows",
    "compare_url",
    "delete_release",
    "ensure_release_tables",
    "final_details",
    "iso",
    "list_release_rows",
    "load_publication",
    "load_release",
    "marked_runs",
    "overview_items",
    "release_overview",
    "release_status",
    "release_view",
    "report_url",
    "save_publication",
    "save_release",
    "trend_point",
    "trend_points",
]
