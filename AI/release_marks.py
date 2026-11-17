# SPDX-License-Identifier: Apache-2.0
"""Release marks of runs: the release a run tested, the accepted run of a release and the pinned baseline.

One row per marked run in ``<schema>.release_marks``, created lazily like ``report_verdicts``. A row lives while
the run has a release or is the baseline of its service. Service and test type (the normalized key, ``step`` for
runs without a known type) are copied from the final report when a mark is set: partial unique indexes keep one
accepted run per release x service x type and one baseline per service x type. The history of the changes is the
audit log.

Accepted run of a release x service x type: a newly marked run becomes the accepted one when the pair has none or
when its accepted run was accepted automatically; a run accepted by hand keeps the acceptance. Accepting by hand
takes it over. When the accepted run loses its mark, the remaining run of the pair with the newest report becomes
the accepted one (automatically).

Baseline lookup, ``find_baseline_run``: ``pinned`` takes the baseline of the service and type, else continues as
``previous_release``: the accepted run of another release with an earlier report, else ``previous_success``.
``previous_success`` and ``previous`` are ``AI.db_store.find_previous_run`` unchanged. The release modes look in the
same service only (never in the project area).

Every function runs in the caller's transaction and does not commit; the writers take a transaction advisory lock
so that two changes of marks never interleave.
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from psycopg2 import sql

from AI import db_store
from AI.db_store import (
    BASELINE_MODE_PINNED,
    BASELINE_MODE_PREVIOUS,
    BASELINE_MODE_PREVIOUS_RELEASE,
    BASELINE_MODE_PREVIOUS_SUCCESS,
    BASELINE_MODES,
    BaselineRun,
    effective_verdict_sql,
    manual_verdicts_cte,
    manual_verdicts_ready,
)
from AI.i18n import LocalizedError, t
from AI.test_profile import normalize_test_type
from AI.verdicts import to_code

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_MODE_PINNED",
    "BASELINE_MODE_PREVIOUS",
    "BASELINE_MODE_PREVIOUS_RELEASE",
    "BASELINE_MODE_PREVIOUS_SUCCESS",
    "BASELINE_MODES",
    "MARK_COLUMNS",
    "RELEASE_MARKS_TABLE",
    "RELEASE_NAME_MAX",
    "RELEASE_NOTE_MAX",
    "ReleaseMarkError",
    "clear_baseline",
    "clear_release",
    "delete_service_marks",
    "empty_mark",
    "ensure_release_marks_table",
    "find_baseline_run",
    "is_protected",
    "list_releases",
    "load_mark",
    "mark_state",
    "mark_test_type",
    "pinned_run",
    "previous_release_run",
    "protected_runs",
    "release_marks_cte",
    "release_marks_ready",
    "set_baseline",
    "set_release",
    "validate_release_name",
]

RELEASE_MARKS_TABLE = "release_marks"
RELEASE_NAME_MAX = 80
RELEASE_NOTE_MAX = 1000
MARK_COLUMNS = (
    "run_name", "run_id", "service", "test_type", "release", "accepted", "accepted_manual", "baseline",
    "note", "author", "created_at", "updated_at",
)
_COLUMN_TYPES = {
    "accepted": "boolean", "accepted_manual": "boolean", "baseline": "boolean",
    "created_at": "timestamptz", "updated_at": "timestamptz",
}
# Key of the transaction advisory lock every writer takes ("LLRM").
_LOCK_KEY = 0x4C4C524D
_READY: set[str] = set()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {} (
    run_name        TEXT        PRIMARY KEY,
    run_id          TEXT        NOT NULL DEFAULT '',
    service         TEXT        NOT NULL DEFAULT '',
    test_type       TEXT        NOT NULL DEFAULT '',
    release         TEXT        NOT NULL DEFAULT '',
    accepted        BOOLEAN     NOT NULL DEFAULT false,
    accepted_manual BOOLEAN     NOT NULL DEFAULT false,
    baseline        BOOLEAN     NOT NULL DEFAULT false,
    note            TEXT        NOT NULL DEFAULT '',
    author          TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (release <> '' OR baseline),
    CHECK (NOT accepted OR release <> '')
);
"""
# (name, unique, columns, predicate)
_INDEXES = (
    (f"idx_{RELEASE_MARKS_TABLE}_accepted", True, "(release, service, test_type)", "WHERE accepted"),
    (f"idx_{RELEASE_MARKS_TABLE}_baseline", True, "(service, test_type)", "WHERE baseline"),
    (f"idx_{RELEASE_MARKS_TABLE}_release", False, "(release)", ""),
)


class ReleaseMarkError(LocalizedError, ValueError):
    """A rejected mark: ``args[0]`` and ``.key`` are the dictionary key (``server.release.*``), ``str()`` its text
    in the current locale. A ``ValueError``, so callers answer 400 with ``str(exc)``."""


# ---- names and types --------------------------------------------------------------------------------------


def validate_release_name(raw: Any) -> str:
    """The release name without surrounding spaces: 1..80 characters, no control characters, case kept.

    Raises :class:`ReleaseMarkError` (a ``ValueError``) with ``server.release.name_required``,
    ``server.release.name_too_long`` or ``server.release.name_invalid``.
    """
    name = "" if raw is None else str(raw).strip()
    if not name:
        raise ReleaseMarkError("server.release.name_required")
    if len(name) > RELEASE_NAME_MAX:
        raise ReleaseMarkError("server.release.name_too_long", max=RELEASE_NAME_MAX)
    if any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in name):
        raise ReleaseMarkError("server.release.name_invalid")
    return name


def _clean_note(note: Any) -> Optional[str]:
    """The note without surrounding spaces; None keeps the stored one."""
    if note is None:
        return None
    text = str(note).strip()
    if len(text) > RELEASE_NOTE_MAX:
        raise ReleaseMarkError("server.release.note_too_long", max=RELEASE_NOTE_MAX)
    return text


def mark_test_type(raw: Any) -> str:
    """Test type a mark stores: the normalized key; runs without a known type are analysed as ``step``."""
    return normalize_test_type(str(raw or "")) or "step"


# ---- table --------------------------------------------------------------------------------------------------


def _schema(schema: Any) -> str:
    return str(schema or "public")


def _table(schema: Any) -> sql.Identifier:
    return sql.Identifier(_schema(schema), RELEASE_MARKS_TABLE)


def _table_text(schema: Any) -> str:
    return f"{db_store._quoted_ident(_schema(schema))}.{RELEASE_MARKS_TABLE}"


def _llm(schema: Any, llm_table: str) -> sql.Identifier:
    return sql.Identifier(_schema(schema), str(llm_table or "llm_reports"))


def ensure_release_marks_table(conn, schema) -> None:
    """Creates ``<schema>.release_marks`` and its indexes once per process (DDL in autocommit).

    Call it before the first statement of the transaction, like ``ensure_report_verdicts_table``.
    """
    schema = _schema(schema)
    key = (schema, RELEASE_MARKS_TABLE)
    if key in db_store._ENSURED_TABLES:
        return
    created = False
    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql.SQL(_CREATE_TABLE).format(_table(schema)))
                created = True
            except Exception as e:
                logger.warning("Failed to create the table %s.%s: %s", schema, RELEASE_MARKS_TABLE, e)
            for name, unique, columns, predicate in _INDEXES:
                try:
                    cur.execute(
                        sql.SQL("CREATE " + ("UNIQUE " if unique else "") + "INDEX IF NOT EXISTS {} ON {} "
                                + columns + (" " + predicate if predicate else "") + ";").format(
                            sql.Identifier(name), _table(schema)
                        )
                    )
                except Exception as e:
                    logger.debug("create index skipped: %s", e)
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    db_store._ENSURED_TABLES.add(key)
    if created:
        _READY.add(schema)


def release_marks_ready(conn, schema) -> bool:
    """True when ``<schema>.release_marks`` exists: without the table no run is marked.

    A read-only probe that is safe inside a transaction; a found table is remembered for the process.
    """
    schema = _schema(schema)
    if schema in _READY:
        return True
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (_table_text(schema),))
        row = cur.fetchone()
    ready = bool(row and row[0])
    if ready:
        _READY.add(schema)
    return ready


def release_marks_cte(schema, ready: bool) -> str:
    """SQL of ``release_marks AS (...)`` with every column of the table (``MARK_COLUMNS``); no rows when the
    table does not exist yet. Readers join it as ``LEFT JOIN release_marks rm ON rm.run_name = ...``.
    """
    if not ready:
        columns = ", ".join(f"NULL::{_COLUMN_TYPES.get(name, 'text')} AS {name}" for name in MARK_COLUMNS)
        return f"release_marks AS (SELECT {columns} WHERE FALSE)"
    return f"release_marks AS (SELECT {', '.join(MARK_COLUMNS)} FROM {_table_text(schema)})"


# ---- state --------------------------------------------------------------------------------------------------


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


def _mark_dict(row: Sequence[Any]) -> Dict[str, Any]:
    values = dict(zip(MARK_COLUMNS, row))
    out: Dict[str, Any] = {}
    for name in MARK_COLUMNS:
        value = values.get(name)
        if name in ("accepted", "accepted_manual", "baseline"):
            out[name] = bool(value)
        elif name in ("created_at", "updated_at"):
            out[name] = _iso(value)
        else:
            out[name] = str(value or "")
    return out


def empty_mark(run_name: str, run_id: str = "", service: str = "", test_type: str = "") -> Dict[str, Any]:
    """State of a run without marks (same keys as a mark)."""
    return {
        "run_name": str(run_name or ""), "run_id": str(run_id or ""), "service": str(service or ""),
        "test_type": str(test_type or ""), "release": "", "accepted": False, "accepted_manual": False,
        "baseline": False, "note": "", "author": "", "created_at": None, "updated_at": None,
    }


def _select_mark(cur, schema, run_name: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        sql.SQL("SELECT " + ", ".join(MARK_COLUMNS) + " FROM {} WHERE run_name = %s").format(_table(schema)),
        (run_name,),
    )
    row = cur.fetchone()
    return _mark_dict(row) if row else None


def load_mark(conn, schema, run_name: str) -> Optional[Dict[str, Any]]:
    """Marks of a run (``MARK_COLUMNS``, booleans and ISO times) or None when it has none."""
    if not run_name or not release_marks_ready(conn, schema):
        return None
    with conn.cursor() as cur:
        return _select_mark(cur, schema, run_name)


def _final_row(cur, schema, llm_table: str, run_name: str) -> Optional[tuple]:
    """(run_id, service, test_type, created_at) of the run's latest final report."""
    cur.execute(
        sql.SQL(
            "SELECT COALESCE(run_id, ''), COALESCE(service, ''), COALESCE(test_type, ''), created_at FROM {} "
            "WHERE run_name = %s AND domain = 'final' ORDER BY created_at DESC LIMIT 1"
        ).format(_llm(schema, llm_table)),
        (run_name,),
    )
    return cur.fetchone()


def mark_state(conn, schema, llm_table: str, run_name: str) -> Optional[Dict[str, Any]]:
    """Marks of a run with defaults for an unmarked one (``release: ""``, ``baseline: False``); None without a
    final report. Service and type of an unmarked run come from its final report."""
    with conn.cursor() as cur:
        final = _final_row(cur, schema, llm_table, run_name)
    if not final:
        return None
    mark = load_mark(conn, schema, run_name)
    if mark is not None:
        return mark
    return empty_mark(run_name, final[0], final[1], mark_test_type(final[2]))


def _not_found(run_name: str) -> LookupError:
    return LookupError(t("server.report.final_not_found", run_name=run_name))


def _lock(cur) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))


def _unaccept(cur, schema, run_name: str) -> None:
    cur.execute(
        sql.SQL("UPDATE {} SET accepted = false, accepted_manual = false, updated_at = now() WHERE run_name = %s").format(
            _table(schema)
        ),
        (run_name,),
    )


def _accepted_holder(cur, schema, release: str, service: str, test_type: str, exclude: str) -> Optional[tuple]:
    """(run_name, accepted_manual) of the accepted run of the pair other than ``exclude``."""
    cur.execute(
        sql.SQL(
            "SELECT run_name, accepted_manual FROM {} WHERE release = %s AND service = %s AND test_type = %s "
            "AND accepted AND run_name <> %s LIMIT 1"
        ).format(_table(schema)),
        (release, service, test_type, exclude),
    )
    return cur.fetchone()


def _reassign_accepted(cur, schema, llm_table: str, release: str, service: str, test_type: str) -> Optional[str]:
    """Makes the run of the pair with the newest report the accepted one (automatically); its name or None.

    Called after the accepted run of the pair lost its acceptance, so the pair has no accepted run.
    """
    cur.execute(
        sql.SQL(
            "SELECT m.run_name FROM {marks} m "
            "LEFT JOIN LATERAL (SELECT f.created_at FROM {llm} f WHERE f.run_name = m.run_name AND f.domain = 'final' "
            "ORDER BY f.created_at DESC LIMIT 1) r ON TRUE "
            "WHERE m.release = %s AND m.service = %s AND m.test_type = %s "
            "ORDER BY r.created_at DESC NULLS LAST, m.updated_at DESC LIMIT 1"
        ).format(marks=_table(schema), llm=_llm(schema, llm_table)),
        (release, service, test_type),
    )
    row = cur.fetchone()
    if not row:
        return None
    cur.execute(
        sql.SQL("UPDATE {} SET accepted = true, accepted_manual = false, updated_at = now() WHERE run_name = %s").format(
            _table(schema)
        ),
        (row[0],),
    )
    return str(row[0])


def _state_after_removal(cur, schema, before: Dict[str, Any]) -> Dict[str, Any]:
    mark = _select_mark(cur, schema, before["run_name"])
    if mark is not None:
        return mark
    return empty_mark(before["run_name"], before["run_id"], before["service"], before["test_type"])


# ---- writers ------------------------------------------------------------------------------------------------


def set_release(
    conn,
    schema,
    llm_table: str,
    run_name: str,
    release: Any,
    note: Any = "",
    accepted: Optional[bool] = None,
    author: str = "",
) -> Dict[str, Any]:
    """Marks the run as a test of ``release`` and returns its marks.

    ``accepted=True`` accepts the run by hand and takes the acceptance from the other run of the pair.
    ``accepted=False`` releases this run's manual acceptance and chooses the newest report automatically;
    it never releases another run's manual acceptance. Omitting it preserves this run's existing acceptance.
    Otherwise the accepted run rule of the module applies. ``note=None`` keeps the stored note. Moving a run to another
    release frees its acceptance in the old one. Raises :class:`ReleaseMarkError` for a bad name or note and
    LookupError without a final report.
    """
    name = validate_release_name(release)
    note_text = _clean_note(note)
    author = str(author or "")
    ensure_release_marks_table(conn, schema)
    with conn.cursor() as cur:
        _lock(cur)
        final = _final_row(cur, schema, llm_table, run_name)
        if not final:
            raise _not_found(run_name)
        run_id, service, test_type = str(final[0] or ""), str(final[1] or ""), mark_test_type(final[2])
        current = _select_mark(cur, schema, run_name)
        pair = (name, service, test_type)
        old_pair = None
        same_pair = False
        if current and current["release"] and current["accepted"]:
            current_pair = (current["release"], current["service"], current["test_type"])
            same_pair = current_pair == pair
            if not same_pair:
                old_pair = current_pair
        holder = _accepted_holder(cur, schema, name, service, test_type, run_name)
        reset_manual = accepted is False and same_pair and bool(current and current["accepted_manual"])
        if accepted is True:
            if holder:
                _unaccept(cur, schema, holder[0])
            is_accepted, is_manual = True, True
        elif reset_manual:
            is_accepted, is_manual = False, False
        elif same_pair:
            is_accepted, is_manual = True, bool(current and current["accepted_manual"])
        elif holder is None:
            is_accepted, is_manual = True, False
        elif not holder[1]:
            # The accepted run of the pair was accepted automatically: the newly marked run takes over.
            _unaccept(cur, schema, holder[0])
            is_accepted, is_manual = True, False
        else:
            is_accepted, is_manual = False, False
        cur.execute(
            sql.SQL(
                "INSERT INTO {} AS m (run_name, run_id, service, test_type, release, accepted, accepted_manual, "
                "baseline, note, author) VALUES (%s, %s, %s, %s, %s, %s, %s, false, %s, %s) "
                "ON CONFLICT (run_name) DO UPDATE SET run_id = EXCLUDED.run_id, service = EXCLUDED.service, "
                "test_type = EXCLUDED.test_type, release = EXCLUDED.release, accepted = EXCLUDED.accepted, "
                "accepted_manual = EXCLUDED.accepted_manual, "
                "note = CASE WHEN %s THEN EXCLUDED.note ELSE m.note END, author = EXCLUDED.author, updated_at = now()"
            ).format(_table(schema)),
            (run_name, run_id, service, test_type, name, is_accepted, is_manual, note_text or "", author,
             note_text is not None),
        )
        if old_pair:
            _reassign_accepted(cur, schema, llm_table, *old_pair)
        if reset_manual:
            _reassign_accepted(cur, schema, llm_table, *pair)
        return _select_mark(cur, schema, run_name) or empty_mark(run_name, run_id, service, test_type)


def clear_release(conn, schema, run_name: str, author: str = "", *, llm_table: str = "llm_reports") -> Dict[str, Any]:
    """Removes the release mark of a run (the baseline stays) and returns its marks; an accepted run hands the
    acceptance to the run of the pair with the newest report. LookupError when the run has no release mark."""
    ensure_release_marks_table(conn, schema)
    with conn.cursor() as cur:
        _lock(cur)
        current = _select_mark(cur, schema, run_name)
        if not current or not current["release"]:
            raise LookupError(t("server.release.not_marked", run_name=run_name))
        if current["baseline"]:
            cur.execute(
                sql.SQL(
                    "UPDATE {} SET release = '', accepted = false, accepted_manual = false, author = %s, "
                    "updated_at = now() WHERE run_name = %s"
                ).format(_table(schema)),
                (str(author or ""), run_name),
            )
        else:
            cur.execute(sql.SQL("DELETE FROM {} WHERE run_name = %s").format(_table(schema)), (run_name,))
        if current["accepted"]:
            _reassign_accepted(cur, schema, llm_table, current["release"], current["service"], current["test_type"])
        return _state_after_removal(cur, schema, current)


def set_baseline(conn, schema, llm_table: str, run_name: str, note: Any = "", author: str = "") -> Dict[str, Any]:
    """Pins the run as the baseline of its service and type (the previous baseline loses it) and returns its marks.

    ``note=None`` or an empty note keeps the stored note. LookupError without a final report.
    """
    note_text = _clean_note(note) or None
    author = str(author or "")
    ensure_release_marks_table(conn, schema)
    with conn.cursor() as cur:
        _lock(cur)
        final = _final_row(cur, schema, llm_table, run_name)
        if not final:
            raise _not_found(run_name)
        run_id, service, test_type = str(final[0] or ""), str(final[1] or ""), mark_test_type(final[2])
        # The previous baseline of the pair: a row without a release goes, a release mark stays.
        cur.execute(
            sql.SQL(
                "DELETE FROM {} WHERE service = %s AND test_type = %s AND baseline AND release = '' AND run_name <> %s"
            ).format(_table(schema)),
            (service, test_type, run_name),
        )
        cur.execute(
            sql.SQL(
                "UPDATE {} SET baseline = false, updated_at = now() "
                "WHERE service = %s AND test_type = %s AND baseline AND run_name <> %s"
            ).format(_table(schema)),
            (service, test_type, run_name),
        )
        cur.execute(
            sql.SQL(
                "INSERT INTO {} AS m (run_name, run_id, service, test_type, release, accepted, accepted_manual, "
                "baseline, note, author) VALUES (%s, %s, %s, %s, '', false, false, true, %s, %s) "
                "ON CONFLICT (run_name) DO UPDATE SET run_id = EXCLUDED.run_id, service = EXCLUDED.service, "
                "test_type = EXCLUDED.test_type, baseline = true, "
                "note = CASE WHEN %s THEN EXCLUDED.note ELSE m.note END, author = EXCLUDED.author, updated_at = now()"
            ).format(_table(schema)),
            (run_name, run_id, service, test_type, note_text or "", author, note_text is not None),
        )
        return _select_mark(cur, schema, run_name) or empty_mark(run_name, run_id, service, test_type)


def clear_baseline(conn, schema, run_name: str, author: str = "") -> Dict[str, Any]:
    """Unpins the baseline (the release mark stays) and returns the marks; LookupError when it is not one."""
    ensure_release_marks_table(conn, schema)
    with conn.cursor() as cur:
        _lock(cur)
        current = _select_mark(cur, schema, run_name)
        if not current or not current["baseline"]:
            raise LookupError(t("server.release.not_baseline", run_name=run_name))
        if current["release"]:
            cur.execute(
                sql.SQL("UPDATE {} SET baseline = false, author = %s, updated_at = now() WHERE run_name = %s").format(
                    _table(schema)
                ),
                (str(author or ""), run_name),
            )
        else:
            cur.execute(sql.SQL("DELETE FROM {} WHERE run_name = %s").format(_table(schema)), (run_name,))
        return _state_after_removal(cur, schema, current)


def is_protected(conn, schema, run_name: str) -> bool:
    """True when the run is the accepted run of a release or a baseline: deleting it needs ``force``."""
    return bool(run_name) and run_name in protected_runs(conn, schema, [run_name])


def protected_runs(conn, schema, run_names: Sequence[str]) -> set:
    """Names among ``run_names`` that are the accepted run of a release or a baseline (one statement)."""
    names = [str(name) for name in dict.fromkeys(run_names or []) if str(name or "")]
    if not names or not release_marks_ready(conn, schema):
        return set()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT run_name FROM {} WHERE run_name = ANY(%s) AND (accepted OR baseline)").format(_table(schema)),
            (names,),
        )
        return {str(row[0]) for row in cur.fetchall() or []}


def delete_service_marks(conn, schema, service: str) -> int:
    """Removes the marks of every run of a service (the service is deleted); the number of rows."""
    if not service or not release_marks_ready(conn, schema):
        return 0
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {} WHERE service = %s").format(_table(schema)), (service,))
        return int(getattr(cur, "rowcount", 0) or 0)


# ---- lists --------------------------------------------------------------------------------------------------


def list_releases(conn, schema, services: Optional[Sequence[str]], llm_table: Optional[str] = None) -> List[Dict[str, Any]]:
    """Releases of the services ``[{release, runs, accepted, services, last_created_at}]``, newest first.

    ``services`` None means every service. ``runs`` counts the marked runs, ``accepted`` the accepted ones;
    ``last_created_at`` is the newest report of the release when ``llm_table`` is given, else the newest mark.
    """
    if services is not None:
        services = [str(item).strip() for item in services if str(item).strip()]
        if not services:
            return []
    if not release_marks_ready(conn, schema):
        return []
    if llm_table:
        newest = sql.SQL("MAX(COALESCE(r.created_at, m.created_at))")
        join = sql.SQL(
            "LEFT JOIN LATERAL (SELECT f.created_at FROM {llm} f WHERE f.run_name = m.run_name AND f.domain = 'final' "
            "ORDER BY f.created_at DESC LIMIT 1) r ON TRUE"
        ).format(llm=_llm(schema, llm_table))
    else:
        newest = sql.SQL("MAX(m.created_at)")
        join = sql.SQL("")
    where = sql.SQL("AND m.service = ANY(%s)") if services is not None else sql.SQL("")
    query = sql.SQL(
        "SELECT m.release, COUNT(*), COUNT(*) FILTER (WHERE m.accepted), "
        "ARRAY_AGG(DISTINCT m.service ORDER BY m.service), {newest} AS last_created_at "
        "FROM {marks} m {join} WHERE m.release <> '' {where} "
        "GROUP BY m.release ORDER BY last_created_at DESC NULLS LAST, m.release DESC"
    ).format(newest=newest, marks=_table(schema), join=join, where=where)
    with conn.cursor() as cur:
        cur.execute(query, (list(services),) if services is not None else ())
        rows = cur.fetchall()
    return [
        {
            "release": str(row[0]),
            "runs": int(row[1] or 0),
            "accepted": int(row[2] or 0),
            "services": [str(item) for item in (row[3] or []) if item],
            "last_created_at": _iso(row[4]),
        }
        for row in rows
    ]


# ---- baseline lookup ----------------------------------------------------------------------------------------


def _baseline_query(conn, schema, llm_table: str, where: str, order: str) -> sql.Composed:
    """Marked runs with their latest final report and the verdict in force (manual first)."""
    manual = manual_verdicts_cte(schema, manual_verdicts_ready(conn, schema))
    return sql.SQL(
        "WITH " + manual + " "
        "SELECT m.run_name, m.service, " + effective_verdict_sql("r") + ", r.created_at, COALESCE(r.run_id, ''), m.release "
        "FROM {marks} m "
        "JOIN LATERAL (SELECT f.created_at, f.run_id, f.sla_verdict, f.verdict, f.project_area FROM {llm} f "
        "WHERE f.run_name = m.run_name AND f.domain = 'final' ORDER BY f.created_at DESC LIMIT 1) r ON TRUE "
        "LEFT JOIN manual_verdicts mv ON mv.run_name = m.run_name "
        "WHERE " + where + " " + order
    ).format(marks=_table(schema), llm=_llm(schema, llm_table))


def _baseline_from_row(row: Sequence[Any], mode: str) -> BaselineRun:
    return BaselineRun(
        run_name=str(row[0]),
        service=str(row[1] or ""),
        verdict=to_code(row[2]),
        created_at=row[3],
        mode=mode,
        run_id=str(row[4] or ""),
        release=str(row[5] or ""),
    )


def pinned_run(conn, schema, llm_table: str, *, service: str, test_type: Any, exclude_run_name: str = "") -> Optional[BaselineRun]:
    """The pinned baseline of the service and type (``mode="pinned"``) unless it is ``exclude_run_name``."""
    if not service or not release_marks_ready(conn, schema):
        return None
    query = _baseline_query(
        conn, schema, llm_table,
        "m.baseline AND m.service = %s AND m.test_type = %s AND m.run_name <> %s", "LIMIT 1",
    )
    with conn.cursor() as cur:
        cur.execute(query, (service, mark_test_type(test_type), str(exclude_run_name or "")))
        row = cur.fetchone()
    return _baseline_from_row(row, BASELINE_MODE_PINNED) if row else None


def previous_release_run(
    conn,
    schema,
    llm_table: str,
    *,
    service: str,
    test_type: Any,
    before: Optional[datetime],
    exclude_run_name: str = "",
    exclude_release: str = "",
    project_area: Optional[str] = None,
) -> Optional[BaselineRun]:
    """The accepted run of another release of the service and type whose report precedes ``before`` (the newest),
    ``mode="previous_release"``. ``exclude_release`` (the release of the current run) is skipped; a report saved
    without a project area fits any area."""
    if not service or not release_marks_ready(conn, schema):
        return None
    where = "m.accepted AND m.release <> '' AND m.release <> %s AND m.service = %s AND m.test_type = %s AND m.run_name <> %s"
    params: list = [str(exclude_release or ""), service, mark_test_type(test_type), str(exclude_run_name or "")]
    if before is not None:
        where += " AND r.created_at < %s"
        params.append(before)
    area = str(project_area or "").strip()
    if area:
        where += " AND (COALESCE(r.project_area, '') = '' OR r.project_area = %s)"
        params.append(area)
    query = _baseline_query(conn, schema, llm_table, where, "ORDER BY r.created_at DESC LIMIT 1")
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        row = cur.fetchone()
    return _baseline_from_row(row, BASELINE_MODE_PREVIOUS_RELEASE) if row else None


def find_baseline_run(
    conn,
    schema,
    llm_table: str,
    *,
    mode: str,
    exclude_run_name: str,
    service: Optional[str] = None,
    services: Optional[Sequence[str]] = None,
    project_area: Optional[str] = None,
    before: Optional[datetime] = None,
    test_type: Optional[str] = None,
) -> Optional[BaselineRun]:
    """Baseline of a run by ``mode`` with the fallback chain pinned -> previous_release -> previous_success.

    ``BaselineRun.mode`` is the mode that found the run and ``release`` its release. ``previous_success`` and
    ``previous`` call ``AI.db_store.find_previous_run`` with the same arguments. Without ``service`` the release
    modes take service and type from the run's final report.
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"Unknown baseline mode '{mode}', expected one of {BASELINE_MODES}")
    fallback = mode
    if mode in (BASELINE_MODE_PINNED, BASELINE_MODE_PREVIOUS_RELEASE):
        fallback = BASELINE_MODE_PREVIOUS_SUCCESS
        found = _release_baseline(
            conn, schema, llm_table, mode=mode, exclude_run_name=exclude_run_name, service=service,
            project_area=project_area, before=before, test_type=test_type,
        )
        if found is not None:
            return found
    kwargs: Dict[str, Any] = {
        "exclude_run_name": exclude_run_name, "mode": fallback, "service": service, "before": before,
        "project_area": project_area, "test_type": test_type,
    }
    if services is not None:
        kwargs["services"] = services
    return db_store.find_previous_run(conn, schema, llm_table, **kwargs)


def _release_baseline(
    conn, schema, llm_table: str, *, mode: str, exclude_run_name: str, service: Optional[str],
    project_area: Optional[str], before: Optional[datetime], test_type: Optional[str],
) -> Optional[BaselineRun]:
    if not release_marks_ready(conn, schema):
        return None
    service = str(service or "").strip()
    if not service and exclude_run_name:
        with conn.cursor() as cur:
            final = _final_row(cur, schema, llm_table, exclude_run_name)
        if final:
            service = str(final[1] or "")
            if test_type is None:
                test_type = str(final[2] or "")
    if not service:
        return None
    if mode == BASELINE_MODE_PINNED:
        found = pinned_run(conn, schema, llm_table, service=service, test_type=test_type, exclude_run_name=exclude_run_name)
        if found is not None:
            return found
    mark = load_mark(conn, schema, exclude_run_name)
    return previous_release_run(
        conn, schema, llm_table, service=service, test_type=test_type, before=before,
        exclude_run_name=exclude_run_name, exclude_release=(mark or {}).get("release") or "",
        project_area=project_area,
    )
