# SPDX-License-Identifier: Apache-2.0
"""Release marks (AI/release_marks.py): the table, the accepted run rule, the baseline, the baseline lookup chain,
the API of loadlens_app/blueprints/releases.py, /runs with marks, deletion protection, the launch form release and
the watcher, /compare_baseline modes, the pipeline baseline context and the Confluence header.

Offline: a fake database keeps final reports and mark rows and answers the statements of the module by their text;
it enforces the unique indexes and checks of the table, so a wrong order of statements fails here.
"""

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import db_store, release_marks  # noqa: E402
from AI.db_store import BaselineRun  # noqa: E402

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def flat(text) -> str:
    return " ".join(str(text).split())


def render(query) -> str:
    from psycopg2 import sql

    if isinstance(query, str):
        return query
    if isinstance(query, sql.Composed):
        return "".join(render(part) for part in query.seq)
    if isinstance(query, sql.Identifier):
        return ".".join('"' + name.replace('"', '""') + '"' for name in query.strings)
    if isinstance(query, sql.SQL):
        return query.string
    if isinstance(query, sql.Literal):
        return repr(query.wrapped)
    raise TypeError(f"unexpected query part {query!r}")


class IntegrityError(Exception):
    pass


class MarksCursor:
    rowcount = 0

    def __init__(self, db):
        self.db = db
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def execute(self, query, params=None):
        text = flat(render(query))
        params = tuple(params or ())
        self.db.log.append((text, params))
        self.rows = []
        db = self.db
        if "to_regclass" in text:
            self.rows = [("public.release_marks" if db.table_exists else None,)]
        elif text.startswith("CREATE TABLE"):
            db.table_exists = True
        elif text.startswith("CREATE"):
            pass
        elif "pg_advisory_xact_lock" in text:
            db.locks += 1
        elif text.startswith("SELECT COALESCE(run_id, ''), COALESCE(service, ''), COALESCE(test_type, ''), created_at FROM"):
            final = db.finals.get(params[0])
            self.rows = [(final["run_id"], final["service"], final["test_type"], final["created_at"])] if final else []
        elif text.startswith("SELECT run_name, run_id, service, test_type, release, accepted, accepted_manual, baseline"):
            mark = db.marks.get(params[0])
            self.rows = [tuple(mark[c] for c in release_marks.MARK_COLUMNS)] if mark else []
        elif text.startswith("SELECT run_name, accepted_manual FROM"):
            release, service, test_type, exclude = params
            self.rows = [(m["run_name"], m["accepted_manual"]) for m in db.marks.values()
                         if (m["release"], m["service"], m["test_type"]) == (release, service, test_type)
                         and m["accepted"] and m["run_name"] != exclude][:1]
        elif text.startswith("SELECT m.run_name FROM") and "ORDER BY r.created_at DESC NULLS LAST" in text:
            pair = params[:3]
            rows = [m for m in db.marks.values() if (m["release"], m["service"], m["test_type"]) == pair]
            rows.sort(key=lambda m: db.finals.get(m["run_name"], {}).get("created_at") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            self.rows = [(rows[0]["run_name"],)] if rows else []
        elif text.startswith("UPDATE") and "SET accepted = false, accepted_manual = false, updated_at = now() WHERE run_name" in text:
            db.update(params[0], accepted=False, accepted_manual=False)
        elif text.startswith("UPDATE") and "SET accepted = true, accepted_manual = false" in text:
            db.update(params[0], accepted=True, accepted_manual=False)
        elif text.startswith("INSERT INTO") and "VALUES (%s, %s, %s, %s, %s, %s, %s, false, %s, %s)" in text:
            run_name, run_id, service, test_type, release, accepted, manual, note, author, note_given = params
            old = db.marks.get(run_name)
            row = dict(old) if old else db.blank(run_name)
            row.update(run_id=run_id, service=service, test_type=test_type, release=release, accepted=accepted,
                       accepted_manual=manual, author=author)
            if old is None or note_given:
                row["note"] = note
            db.put(row)
        elif text.startswith("INSERT INTO") and "VALUES (%s, %s, %s, %s, '', false, false, true, %s, %s)" in text:
            run_name, run_id, service, test_type, note, author, note_given = params
            old = db.marks.get(run_name)
            row = dict(old) if old else db.blank(run_name)
            row.update(run_id=run_id, service=service, test_type=test_type, baseline=True, author=author)
            if old is None or note_given:
                row["note"] = note
            db.put(row)
        elif text.startswith("DELETE FROM") and "AND baseline AND release = '' AND run_name <> %s" in text:
            service, test_type, exclude = params
            for name in [n for n, m in db.marks.items() if (m["service"], m["test_type"]) == (service, test_type)
                         and m["baseline"] and m["release"] == "" and n != exclude]:
                del db.marks[name]
        elif text.startswith("UPDATE") and "SET baseline = false, updated_at = now() WHERE service" in text:
            service, test_type, exclude = params
            for name in [n for n, m in db.marks.items() if (m["service"], m["test_type"]) == (service, test_type)
                         and m["baseline"] and n != exclude]:
                db.update(name, baseline=False)
        elif text.startswith("UPDATE") and "SET release = '', accepted = false" in text:
            db.update(params[1], release="", accepted=False, accepted_manual=False, author=params[0])
        elif text.startswith("UPDATE") and "SET baseline = false, author = %s" in text:
            db.update(params[1], baseline=False, author=params[0])
        elif text.startswith("DELETE FROM") and text.endswith("WHERE run_name = %s"):
            db.marks.pop(params[0], None)
        elif text.startswith("DELETE FROM") and text.endswith("WHERE service = %s"):
            gone = [n for n, m in db.marks.items() if m["service"] == params[0]]
            for name in gone:
                del db.marks[name]
            self.rowcount = len(gone)
        elif text.startswith("SELECT run_name FROM") and "(accepted OR baseline)" in text:
            self.rows = [(n,) for n in params[0] if n in db.marks and (db.marks[n]["accepted"] or db.marks[n]["baseline"])]
        elif "JOIN LATERAL" in text and "WHERE m.baseline AND" in text:
            service, test_type, exclude = params
            self.rows = [db.baseline_row(m) for m in db.marks.values()
                         if m["baseline"] and (m["service"], m["test_type"]) == (service, test_type)
                         and m["run_name"] != exclude and m["run_name"] in db.finals][:1]
        elif "JOIN LATERAL" in text and "WHERE m.accepted AND" in text:
            exclude_release, service, test_type, exclude = params[:4]
            rest = list(params[4:])
            before = rest.pop(0) if "r.created_at < %s" in text else None
            area = rest.pop(0) if "r.project_area = %s" in text else None
            rows = [m for m in db.marks.values()
                    if m["accepted"] and m["release"] and m["release"] != exclude_release
                    and (m["service"], m["test_type"]) == (service, test_type) and m["run_name"] != exclude
                    and m["run_name"] in db.finals
                    and (before is None or db.finals[m["run_name"]]["created_at"] < before)
                    and (area is None or db.finals[m["run_name"]].get("project_area", "") in ("", area))]
            rows.sort(key=lambda m: db.finals[m["run_name"]]["created_at"], reverse=True)
            self.rows = [db.baseline_row(m) for m in rows[:1]]
        elif "ARRAY_AGG" in text:
            services = params[0] if params else None
            groups = {}
            for m in db.marks.values():
                if not m["release"] or (services is not None and m["service"] not in services):
                    continue
                groups.setdefault(m["release"], []).append(m)
            out = []
            for release, items in groups.items():
                newest = max((db.finals.get(m["run_name"], {}).get("created_at") or m["created_at"]) for m in items)
                out.append((release, len(items), sum(1 for m in items if m["accepted"]),
                            sorted({m["service"] for m in items}), newest))
            out.sort(key=lambda r: (r[4], r[0]), reverse=True)
            self.rows = out
        else:
            raise AssertionError(f"unexpected statement: {text}")


class MarksDb:
    def __init__(self, table_exists=True):
        self.table_exists = table_exists
        self.finals = {}
        self.marks = {}
        self.log = []
        self.locks = 0
        self.autocommit = False
        self.commits = 0
        self.clock = T0

    def report(self, run_name, service="checkout", test_type="step", minutes=0, verdict="pass", area=""):
        self.finals[run_name] = {"run_id": f"rid-{run_name}", "service": service, "test_type": test_type,
                                 "created_at": T0 + timedelta(minutes=minutes), "verdict": verdict, "project_area": area}
        return self

    def blank(self, run_name):
        self.clock += timedelta(seconds=1)
        return {"run_name": run_name, "run_id": "", "service": "", "test_type": "", "release": "", "accepted": False,
                "accepted_manual": False, "baseline": False, "note": "", "author": "", "created_at": self.clock,
                "updated_at": self.clock}

    def update(self, run_name, **values):
        if run_name in self.marks:
            row = dict(self.marks[run_name])
            row.update(values)
            self.put(row)

    def put(self, row):
        if not (row["release"] or row["baseline"]):
            raise IntegrityError("check: release <> '' OR baseline")
        if row["accepted"] and not row["release"]:
            raise IntegrityError("check: NOT accepted OR release <> ''")
        for other in self.marks.values():
            if other["run_name"] == row["run_name"]:
                continue
            if row["accepted"] and other["accepted"] and (other["release"], other["service"], other["test_type"]) == (
                    row["release"], row["service"], row["test_type"]):
                raise IntegrityError("unique: accepted")
            if row["baseline"] and other["baseline"] and (other["service"], other["test_type"]) == (row["service"], row["test_type"]):
                raise IntegrityError("unique: baseline")
        self.marks[row["run_name"]] = row

    def baseline_row(self, mark):
        final = self.finals[mark["run_name"]]
        return (mark["run_name"], mark["service"], final["verdict"], final["created_at"], final["run_id"], mark["release"])

    def state(self, run_name):
        m = self.marks.get(run_name)
        return None if m is None else (m["release"], m["accepted"], m["accepted_manual"], m["baseline"])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.commits += 1
        return False

    def cursor(self):
        return MarksCursor(self)

    def rollback(self):
        pass

    def close(self):
        pass


def statements(db):
    return [text for text, _ in db.log]


# ---- table and schema ----------------------------------------------------------------------------------------


def test_deployment_schema_app_role_and_run_tables_know_the_table():
    schema_sql = flat((ROOT_DIR / "docker" / "initdb" / "02_schema.sql").read_text(encoding="utf-8"))
    assert "CREATE TABLE IF NOT EXISTS public.release_marks ( run_name TEXT PRIMARY KEY," in schema_sql
    assert "CHECK (release <> '' OR baseline), CHECK (NOT accepted OR release <> '') );" in schema_sql
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS idx_release_marks_accepted ON public.release_marks "
            "(release, service, test_type) WHERE accepted;") in schema_sql
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS idx_release_marks_baseline ON public.release_marks "
            "(service, test_type) WHERE baseline;") in schema_sql
    assert "CREATE INDEX IF NOT EXISTS idx_release_marks_release ON public.release_marks (release);" in schema_sql
    # The lazy DDL of the module creates the same columns.
    for column in release_marks.MARK_COLUMNS:
        assert f" {column} " in flat(release_marks._CREATE_TABLE)
    role = (ROOT_DIR / "docker" / "initdb" / "03_app_role.sh").read_text(encoding="utf-8")
    assert " release_marks " in role.split("LOADLENS_TABLES=", 1)[1].splitlines()[0]
    _, tables = core._run_tables({})
    assert release_marks.RELEASE_MARKS_TABLE in tables  # deleted and renamed with the run


def test_ensure_creates_the_table_once_and_the_probe_finds_it(monkeypatch):
    monkeypatch.setattr(release_marks, "_READY", set())
    db = MarksDb(table_exists=False)
    assert release_marks.release_marks_ready(db, "public") is False
    release_marks.ensure_release_marks_table(db, "public")
    texts = statements(db)
    assert any(t.startswith('CREATE TABLE IF NOT EXISTS "public"."release_marks"') for t in texts)
    assert 'CREATE UNIQUE INDEX IF NOT EXISTS "idx_release_marks_accepted" ON "public"."release_marks" (release, service, test_type) WHERE accepted;' in texts
    assert 'CREATE UNIQUE INDEX IF NOT EXISTS "idx_release_marks_baseline" ON "public"."release_marks" (service, test_type) WHERE baseline;' in texts
    assert db.autocommit is False and "public" in release_marks._READY
    release_marks.ensure_release_marks_table(db, "public")
    assert len(db.log) == len(texts)  # once per process
    assert release_marks.release_marks_ready(db, "public") is True


def test_cte_has_every_column_and_no_rows_without_the_table():
    ready = flat(release_marks.release_marks_cte("perf", True))
    assert ready == ('release_marks AS (SELECT run_name, run_id, service, test_type, release, accepted, accepted_manual, '
                     'baseline, note, author, created_at, updated_at FROM "perf".release_marks)')
    missing = flat(release_marks.release_marks_cte("perf", False))
    assert missing.startswith("release_marks AS (SELECT NULL::text AS run_name") and missing.endswith("WHERE FALSE)")
    assert "NULL::boolean AS accepted" in missing and "NULL::timestamptz AS updated_at" in missing


@pytest.mark.parametrize("raw, key", [
    ("", "server.release.name_required"),
    ("   ", "server.release.name_required"),
    (None, "server.release.name_required"),
    ("x" * 81, "server.release.name_too_long"),
    ("25.11\nnext", "server.release.name_invalid"),
    ("25\t11", "server.release.name_invalid"),
])
def test_release_names_are_checked(raw, key):
    with pytest.raises(ValueError) as exc:
        release_marks.validate_release_name(raw)
    assert exc.value.key == key and exc.value.args[0] == key
    assert str(exc.value) and str(exc.value) != key  # rendered text


def test_release_names_keep_case_and_inner_spaces():
    assert release_marks.validate_release_name("  Release 25.11 RC  ") == "Release 25.11 RC"
    assert release_marks.validate_release_name("x" * 80) == "x" * 80
    assert release_marks.mark_test_type("Stability") == "soak" and release_marks.mark_test_type("") == "step"
    assert release_marks.mark_test_type("custom") == "step"


# ---- accepted run rule -----------------------------------------------------------------------------------------


def test_the_first_marked_run_is_accepted_and_copies_service_and_type():
    db = MarksDb().report("r1", service="orders", test_type="Stability")
    state = release_marks.set_release(db, "public", "llm_reports", "r1", " 25.11 ", note=" первый ", author="engineer1")
    assert state["release"] == "25.11" and state["accepted"] is True and state["accepted_manual"] is False
    assert (state["service"], state["test_type"], state["run_id"]) == ("orders", "soak", "rid-r1")
    assert state["note"] == "первый" and state["author"] == "engineer1" and state["baseline"] is False
    assert db.locks == 1


def test_an_automatic_acceptance_moves_to_the_newly_marked_run_and_a_manual_one_stays():
    db = MarksDb().report("r1", minutes=0).report("r2", minutes=10).report("r3", minutes=20)
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11")
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.11")
    assert db.state("r1") == ("25.11", False, False, False) and db.state("r2") == ("25.11", True, False, False)
    # Accepting by hand takes the acceptance and keeps it against newer marks.
    state = release_marks.set_release(db, "public", "llm_reports", "r1", "25.11", accepted=True)
    assert state["accepted"] is True and state["accepted_manual"] is True
    assert db.state("r2") == ("25.11", False, False, False)
    release_marks.set_release(db, "public", "llm_reports", "r3", "25.11")
    assert db.state("r3") == ("25.11", False, False, False) and db.state("r1") == ("25.11", True, True, False)
    # Another type or service is another pair.
    db.report("soak-1", test_type="soak", minutes=30)
    assert release_marks.set_release(db, "public", "llm_reports", "soak-1", "25.11")["accepted"] is True


def test_marking_again_keeps_the_acceptance_and_a_missing_note_keeps_the_note():
    db = MarksDb().report("r1").report("r2", minutes=5)
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11", note="база")
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.11", accepted=True)
    state = release_marks.set_release(db, "public", "llm_reports", "r2", "25.11", note=None)
    assert state["accepted"] is True and state["accepted_manual"] is True
    assert release_marks.set_release(db, "public", "llm_reports", "r1", "25.11", note=None)["note"] == "база"
    assert release_marks.set_release(db, "public", "llm_reports", "r1", "25.11", note="")["note"] == ""


def test_removing_the_accepted_run_hands_the_acceptance_to_the_newest_report():
    db = MarksDb().report("r1", minutes=0).report("r2", minutes=20).report("r3", minutes=10)
    for name in ("r1", "r2", "r3"):
        release_marks.set_release(db, "public", "llm_reports", name, "25.11")
    assert db.state("r3")[1] is True  # the last marked
    state = release_marks.clear_release(db, "public", "r3", "engineer2")
    assert state["release"] == "" and state["accepted"] is False and db.state("r3") is None
    assert db.state("r2") == ("25.11", True, False, False)  # newest report, not the newest mark
    with pytest.raises(LookupError):
        release_marks.clear_release(db, "public", "r3")


def test_unchecking_manual_acceptance_restores_the_newest_report():
    db = MarksDb().report("older").report("newer", minutes=10)
    release_marks.set_release(db, "public", "llm_reports", "older", "25.11", accepted=True)
    release_marks.set_release(db, "public", "llm_reports", "newer", "25.11")
    release_marks.set_baseline(db, "public", "llm_reports", "older")
    state = release_marks.set_release(db, "public", "llm_reports", "older", "25.11", accepted=False)
    assert (state["accepted"], state["accepted_manual"], state["baseline"]) == (False, False, True)
    assert db.state("newer") == ("25.11", True, False, False)


def test_unchecking_the_only_run_keeps_it_accepted_automatically():
    db = MarksDb().report("only")
    release_marks.set_release(db, "public", "llm_reports", "only", "25.11", accepted=True)
    state = release_marks.set_release(db, "public", "llm_reports", "only", "25.11", accepted=False)
    assert (state["accepted"], state["accepted_manual"]) == (True, False)


def test_moving_with_acceptance_unchecked_preserves_another_manual_choice():
    db = MarksDb().report("old").report("moving", minutes=10).report("chosen", minutes=20)
    release_marks.set_release(db, "public", "llm_reports", "old", "25.10")
    release_marks.set_release(db, "public", "llm_reports", "moving", "25.10", accepted=True)
    release_marks.set_release(db, "public", "llm_reports", "chosen", "25.11", accepted=True)
    release_marks.set_release(db, "public", "llm_reports", "moving", "25.11", accepted=False)
    assert db.state("old") == ("25.10", True, False, False)
    assert db.state("moving") == ("25.11", False, False, False)
    assert db.state("chosen") == ("25.11", True, True, False)
    release_marks.set_release(db, "public", "llm_reports", "moving", "25.11", accepted=False)
    assert db.state("chosen") == ("25.11", True, True, False)


def test_moving_a_run_to_another_release_frees_its_acceptance():
    db = MarksDb().report("r1").report("r2", minutes=5)
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.10")
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.10", accepted=True)
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.11")
    assert db.state("r2") == ("25.11", True, False, False)
    assert db.state("r1") == ("25.10", True, False, False)


def test_marks_need_a_final_report_and_a_valid_note():
    db = MarksDb()
    with pytest.raises(LookupError):
        release_marks.set_release(db, "public", "llm_reports", "ghost", "25.11")
    with pytest.raises(LookupError):
        release_marks.set_baseline(db, "public", "llm_reports", "ghost")
    db.report("r1")
    with pytest.raises(ValueError) as exc:
        release_marks.set_release(db, "public", "llm_reports", "r1", "25.11", note="x" * 1001)
    assert exc.value.key == "server.release.note_too_long"
    assert db.marks == {}


# ---- baseline ------------------------------------------------------------------------------------------------


def test_a_new_baseline_unpins_the_previous_one_of_the_service_and_type():
    db = MarksDb().report("r1").report("r2", minutes=5).report("r3", minutes=10).report("o1", service="orders")
    release_marks.set_baseline(db, "public", "llm_reports", "r1", note="эталон")
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.11")
    release_marks.set_baseline(db, "public", "llm_reports", "o1")
    state = release_marks.set_baseline(db, "public", "llm_reports", "r2", author="engineer1")
    assert state["baseline"] is True and state["release"] == "25.11" and state["accepted"] is True
    assert db.state("r1") is None  # a baseline without a release leaves no row
    assert db.state("o1") == ("", False, False, True)  # another service keeps its baseline
    release_marks.set_baseline(db, "public", "llm_reports", "r3")
    assert db.state("r2") == ("25.11", True, False, False)  # the release mark stays


def test_clearing_the_baseline_keeps_the_release_and_clearing_the_release_keeps_the_baseline():
    db = MarksDb().report("r1")
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11")
    release_marks.set_baseline(db, "public", "llm_reports", "r1")
    state = release_marks.clear_release(db, "public", "r1")
    assert (state["release"], state["baseline"], state["accepted"]) == ("", True, False)
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11")
    state = release_marks.clear_baseline(db, "public", "r1")
    assert (state["release"], state["baseline"]) == ("25.11", False)
    release_marks.clear_release(db, "public", "r1")
    assert db.marks == {}
    with pytest.raises(LookupError):
        release_marks.clear_baseline(db, "public", "r1")


def test_protection_mark_state_and_service_cleanup():
    db = MarksDb().report("r1").report("r2", minutes=5).report("o1", service="orders")
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11")
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.11", accepted=False)
    release_marks.set_baseline(db, "public", "llm_reports", "o1")
    assert db.state("r2")[1] is True  # accepted=False is not a refusal: the rule applies (auto handover)
    assert release_marks.is_protected(db, "public", "r2") is True
    assert release_marks.is_protected(db, "public", "r1") is False
    assert release_marks.protected_runs(db, "public", ["r1", "r2", "o1", "x"]) == {"r2", "o1"}
    unmarked = release_marks.mark_state(db.report("plain"), "public", "llm_reports", "plain")
    assert unmarked == release_marks.empty_mark("plain", "rid-plain", "checkout", "step")
    assert release_marks.mark_state(db, "public", "llm_reports", "ghost") is None
    assert release_marks.delete_service_marks(db, "public", "checkout") == 2
    assert set(db.marks) == {"o1"}


def test_list_releases_counts_runs_and_filters_services():
    db = MarksDb().report("r1", minutes=0).report("r2", minutes=50).report("o1", service="orders", minutes=60)
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.10")
    release_marks.set_release(db, "public", "llm_reports", "r2", "25.11")
    release_marks.set_release(db, "public", "llm_reports", "o1", "25.11")
    items = release_marks.list_releases(db, "public", None, "llm_reports")
    assert [(i["release"], i["runs"], i["accepted"], i["services"]) for i in items] == [
        ("25.11", 2, 2, ["checkout", "orders"]), ("25.10", 1, 1, ["checkout"]),
    ]
    assert items[0]["last_created_at"] == (T0 + timedelta(minutes=60)).isoformat()
    only = release_marks.list_releases(db, "public", ["checkout"], "llm_reports")
    assert [(i["release"], i["services"]) for i in only] == [("25.11", ["checkout"]), ("25.10", ["checkout"])]
    text, params = db.log[-1]
    assert "AND m.service = ANY(%s)" in text and params == (["checkout"],)
    assert "LEFT JOIN LATERAL" in text and "ORDER BY last_created_at DESC NULLS LAST, m.release DESC" in text
    assert release_marks.list_releases(db, "public", [], "llm_reports") == []
    db.log.clear()
    release_marks.list_releases(db, "public", None)
    assert "LATERAL" not in db.log[-1][0] and "MAX(m.created_at)" in db.log[-1][0]


# ---- baseline lookup chain -------------------------------------------------------------------------------------


@pytest.fixture
def previous_runs(monkeypatch):
    calls = []

    def fake(conn, schema, table, **kwargs):
        calls.append(kwargs)
        return BaselineRun("prev-ok", "checkout", "pass", T0, mode=kwargs["mode"], run_id="rid-prev")

    monkeypatch.setattr(db_store, "find_previous_run", fake)
    return calls


def test_pinned_takes_the_baseline_then_the_previous_release_then_the_previous_success(previous_runs):
    db = MarksDb().report("base", minutes=0).report("rel-a", minutes=10).report("rel-b", minutes=20)
    db.report("current", minutes=30)
    release_marks.set_baseline(db, "public", "llm_reports", "base")
    release_marks.set_release(db, "public", "llm_reports", "rel-a", "25.10")
    release_marks.set_release(db, "public", "llm_reports", "rel-b", "25.11")
    before = T0 + timedelta(minutes=31)
    kwargs = dict(mode="pinned", exclude_run_name="current", service="checkout", before=before, test_type="step")

    found = release_marks.find_baseline_run(db, "public", "llm_reports", **kwargs)
    assert (found.run_name, found.mode, found.release, found.run_id, found.verdict) == ("base", "pinned", "", "rid-base", "pass")

    # The baseline itself looks further: the newest accepted run of another release.
    found = release_marks.find_baseline_run(db, "public", "llm_reports", **{**kwargs, "exclude_run_name": "base"})
    assert (found.run_name, found.mode, found.release) == ("rel-b", "previous_release", "25.11")

    # A marked run skips its own release.
    release_marks.set_release(db, "public", "llm_reports", "current", "25.11", accepted=True)
    release_marks.clear_baseline(db, "public", "base")
    found = release_marks.find_baseline_run(db, "public", "llm_reports", **kwargs)
    assert (found.run_name, found.mode, found.release) == ("rel-a", "previous_release", "25.10")

    # An earlier report only: nothing before rel-a.
    found = release_marks.find_baseline_run(db, "public", "llm_reports", **{**kwargs, "before": T0 + timedelta(minutes=5)})
    assert (found.run_name, found.mode) == ("prev-ok", "previous_success")
    assert previous_runs == [{"exclude_run_name": "current", "mode": "previous_success", "service": "checkout",
                              "before": T0 + timedelta(minutes=5), "project_area": None, "test_type": "step"}]


def test_previous_modes_pass_through_and_unknown_modes_fail(previous_runs):
    db = MarksDb()
    found = release_marks.find_baseline_run(db, "public", "llm_reports", mode="previous", exclude_run_name="x",
                                            services=["a", "b"], service="a")
    assert found.mode == "previous" and previous_runs[0]["services"] == ["a", "b"]
    assert db.log == []  # no release lookups
    with pytest.raises(ValueError):
        release_marks.find_baseline_run(db, "public", "llm_reports", mode="best", exclude_run_name="x")


def test_find_previous_run_refuses_the_release_modes():
    with pytest.raises(ValueError):
        db_store.find_previous_run(MarksDb(), "public", "llm_reports", mode="pinned", exclude_run_name="x")


def test_release_modes_without_the_table_fall_back_at_once(monkeypatch, previous_runs):
    monkeypatch.setattr(release_marks, "_READY", set())
    db = MarksDb(table_exists=False)
    found = release_marks.find_baseline_run(db, "public", "llm_reports", mode="previous_release",
                                            exclude_run_name="cur", service="checkout")
    assert found.mode == "previous_success" and statements(db) == ["SELECT to_regclass(%s)"]


def test_release_modes_take_service_and_type_of_the_run_without_a_service(previous_runs):
    db = MarksDb().report("base", test_type="soak").report("cur", test_type="stability", minutes=5)
    release_marks.set_baseline(db, "public", "llm_reports", "base")
    found = release_marks.find_baseline_run(db, "public", "llm_reports", mode="pinned", exclude_run_name="cur")
    assert found.run_name == "base" and found.mode == "pinned"


def test_baseline_sql_reads_the_latest_final_report_and_the_manual_verdict():
    db = MarksDb().report("rel-a")
    release_marks.set_release(db, "public", "llm_reports", "rel-a", "25.10")
    db.log.clear()
    release_marks.previous_release_run(db, "public", "llm_reports", service="checkout", test_type="step", before=T0,
                                       exclude_run_name="cur", exclude_release="25.11", project_area="shop")
    text, params = db.log[-1]
    assert text.startswith("WITH manual_verdicts AS (")
    assert 'JOIN LATERAL (SELECT f.created_at, f.run_id, f.sla_verdict, f.verdict, f.project_area FROM "public"."llm_reports" f' in text
    assert "WHERE f.run_name = m.run_name AND f.domain = 'final' ORDER BY f.created_at DESC LIMIT 1) r ON TRUE" in text
    assert "LEFT JOIN manual_verdicts mv ON mv.run_name = m.run_name" in text
    assert ("WHERE m.accepted AND m.release <> '' AND m.release <> %s AND m.service = %s AND m.test_type = %s "
            "AND m.run_name <> %s AND r.created_at < %s AND (COALESCE(r.project_area, '') = '' OR r.project_area = %s) "
            "ORDER BY r.created_at DESC LIMIT 1") in text
    assert params == ("25.11", "checkout", "step", "cur", T0, "shop")
    assert "COALESCE(mv.verdict, r.sla_verdict, r.verdict)" in text


def test_find_previous_run_reports_its_mode_and_run_id():
    class Conn:
        def __init__(self):
            self.log = []

        def cursor(self):
            conn = self

            class Cur:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, query, params=None):
                    conn.log.append(flat(render(query)))

                def fetchone(self):
                    return ("old", "checkout", "Успешно", T0, "rid-old")

            return Cur()

    conn = Conn()
    found = db_store.find_previous_run(conn, "public", "llm_reports", exclude_run_name="cur", service="checkout", before=T0)
    assert found == BaselineRun("old", "checkout", "pass", T0, mode="previous_success", run_id="rid-old")
    assert ", r.created_at, COALESCE(r.run_id, '') FROM" in conn.log[-1]


# ---- API ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app.blueprints import dashboard, releases

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    db = MarksDb().report("r1").report("r2", minutes=5).report("o1", service="orders", minutes=7)
    for module in (releases, dashboard):
        monkeypatch.setattr(module, "_ts_conn", lambda: db)
    monkeypatch.setattr(releases, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(releases, "_resolve_services_filter", lambda area: ["checkout"])
    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), db=db, app=app)


def test_api_state_set_reset_and_names(api):
    resp = api.client.get("/runs/r1/release")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["release"] == "" and body["baseline"] is False and body["releases"] == []
    assert (body["service"], body["test_type"], body["run_id"]) == ("checkout", "step", "rid-r1")
    assert body["name_max"] == 80 and body["note_max"] == 1000

    resp = api.client.put("/runs/r1/release", json={"release": "25.11", "note": "ночной"})
    assert resp.status_code == 200 and resp.get_json()["accepted"] is True and resp.get_json()["note"] == "ночной"
    resp = api.client.put("/runs/r2/release", json={"release": "25.11"})
    assert resp.get_json()["accepted"] is True and api.db.state("r1")[1] is False
    resp = api.client.put("/runs/r1/release", json={"release": "25.11", "accepted": True})
    assert resp.get_json()["accepted_manual"] is True and api.db.state("r2")[1] is False
    api.client.put("/runs/o1/release", json={"release": "25.12"})  # another area
    assert api.client.get("/runs/r1/release").get_json()["releases"] == ["25.11"]
    assert [item["release"] for item in api.client.get("/releases_data").get_json()] == ["25.11"]

    resp = api.client.delete("/runs/r1/release")
    assert resp.status_code == 200 and resp.get_json()["release"] == ""
    assert api.db.state("r2") == ("25.11", True, False, False)
    assert api.client.delete("/runs/r1/release").status_code == 404

    resp = api.client.put("/runs/r1/baseline", json={})
    assert resp.status_code == 200 and resp.get_json()["baseline"] is True
    assert api.client.put("/runs/r1/baseline").status_code == 200  # no body
    assert api.client.delete("/runs/r1/baseline").get_json()["baseline"] is False
    assert api.client.delete("/runs/r1/baseline").status_code == 404


def test_api_refusals(api):
    assert api.client.get("/runs/ghost/release").status_code == 404
    assert api.client.put("/runs/ghost/release", json={"release": "25.11"}).status_code == 404
    assert api.client.put("/runs/ghost/baseline", json={}).status_code == 404
    for body in ({"release": ""}, {"release": "x" * 81}, {"release": "a\nb"}, {"release": "25.11", "note": "x" * 1001}):
        resp = api.client.put("/runs/r1/release", json=body)
        assert resp.status_code == 400 and resp.get_json()["error"]
    assert "имя релиза" in api.client.put("/runs/r1/release", json={"release": " "}).get_json()["error"].lower()
    for body in ({"release": 5}, {"release": "25.11", "note": 1}, {"release": "25.11", "accepted": "yes"}, [1]):
        assert api.client.put("/runs/r1/release", json=body).status_code == 400
    assert api.client.put("/runs/r1/baseline", json={"note": 5}).status_code == 400
    assert api.db.marks == {}


def test_api_distinguishes_omitted_acceptance_from_unchecked(api):
    api.client.put("/runs/r1/release", json={"release": "25.11", "accepted": True})
    api.client.put("/runs/r2/release", json={"release": "25.11"})
    unchanged = api.client.put("/runs/r1/release", json={"release": "25.11", "note": "edited"})
    assert unchanged.get_json()["accepted_manual"] is True
    reset = api.client.put("/runs/r1/release", json={"release": "25.11", "accepted": False})
    assert reset.status_code == 200 and reset.get_json()["accepted_manual"] is False
    assert reset.get_json()["accepted"] is False
    assert api.db.state("r2") == ("25.11", True, False, False)


def test_api_conflicts_answer_409(api, monkeypatch):
    import psycopg2

    def busy(*args, **kwargs):
        raise psycopg2.IntegrityError("duplicate key value violates unique constraint")

    monkeypatch.setattr(release_marks, "set_release", busy)
    resp = api.client.put("/runs/r1/release", json={"release": "25.11"})
    assert resp.status_code == 409 and resp.get_json()["code"] == "conflict"


def test_api_audit_and_dashboard_cache(api, monkeypatch):
    from flask import g

    from loadlens_app import response_cache
    from loadlens_app.blueprints import releases

    seen = []
    monkeypatch.setattr(response_cache, "invalidate", lambda name: seen.append(name))
    with api.app.test_request_context("/runs/r1/release", method="PUT", json={"release": "25.11"}):
        resp = releases.set_run_release("r1")
        assert resp[1] == 200
        assert g.audit_extra == {"run_name": "r1", "manual_accept": False, "release": "25.11", "accepted": True, "baseline": False}
    assert seen == [response_cache.DASHBOARD]


def test_policy_rules_of_the_release_routes():
    from loadlens_app.auth.models import Role
    from loadlens_app.auth.policy import POLICY

    assert POLICY["releases.get_run_release"].role == Role.VIEWER
    assert POLICY["releases.list_releases"].role == Role.VIEWER
    for endpoint, audit in (("set_run_release", "report.release"), ("reset_run_release", "report.release_reset"),
                            ("set_run_baseline", "report.baseline"), ("reset_run_baseline", "report.baseline_reset")):
        rule = POLICY[f"releases.{endpoint}"]
        assert rule.role == Role.ENGINEER and rule.audit == audit and not rule.session_only  # API tokens may mark


# ---- /runs and deletion ------------------------------------------------------------------------------------------


def test_runs_join_the_marks_filter_and_sort_by_release(api, monkeypatch):
    from tests.test_smoke_api import RecordingConnection
    from loadlens_app.blueprints import dashboard

    row = ("r1", T0, T0, "checkout", "pass", "pass", None, T0, "step", "rid-r1", "", 10.0, 1, None, None, None, None,
           "25.11", True, True)
    conn = RecordingConnection(fetchall_results=[[row]])
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_ensure_run_index_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])

    body = api.client.get("/runs", query_string={"release": "*", "sort": "release"}).get_json()
    assert body[0]["release"] == "25.11" and body[0]["release_accepted"] is True and body[0]["baseline"] is True
    (text, params), = conn.log
    assert 'release_marks AS (SELECT run_name, run_id, service, test_type, release, accepted' in text
    assert "LEFT JOIN release_marks rm ON rm.run_name = b.run_name" in text
    assert "COALESCE(rm.release, '') AS release, COALESCE(rm.accepted, false) AS release_accepted" in text
    assert "CASE WHEN rm.release <> '' THEN MAX(f.created_at) OVER (PARTITION BY rm.release) END AS release_newest" in text
    assert "manual_changed_at, release, release_accepted, baseline FROM joined WHERE release <> ''" in text
    assert "ORDER BY release_newest DESC NULLS LAST, release ASC, report_created_at DESC NULLS LAST, run_name ASC" in text

    conn.log.clear()
    conn._cursor._fetchall = [[row[:13] + (None,) * 4 + ([],) + ("", False, False)]]
    body = api.client.get("/runs", query_string={"release": "25.11", "include": "sla"}).get_json()
    (text, params), = conn.log
    assert "WHERE release = %s" in text and params[-3] == "25.11"
    assert "NULL::timestamptz AS release_newest" in text  # no window without sort=release
    assert "AS sla_checks, release, release_accepted, baseline FROM joined" in text
    assert body[0]["release"] == "" and body[0]["baseline"] is False


def test_runs_item_reads_the_marks_after_the_manual_verdict():
    from loadlens_app.blueprints import dashboard

    base = ("r1", T0, T0, "checkout", "pass", "pass", None, T0, "step", "rid", "", None, 1, None, None, None, None)
    assert dashboard._runs_item(base)["release"] == "" and dashboard._runs_item(base)["baseline"] is False
    item = dashboard._runs_item(base + ("25.11", True, False))
    assert (item["release"], item["release_accepted"], item["baseline"]) == ("25.11", True, False)
    item = dashboard._runs_item(base + ([], "25.12", False, True), 18)
    assert (item["release"], item["release_accepted"], item["baseline"]) == ("25.12", False, True)


def test_deleting_a_protected_run_needs_force(api, monkeypatch):
    from loadlens_app.blueprints import dashboard

    deleted = []
    monkeypatch.setattr(dashboard, "_delete_run_data", deleted.append)
    api.client.put("/runs/r1/release", json={"release": "25.11"})
    resp = api.client.delete("/runs/r1")
    assert resp.status_code == 409 and resp.get_json()["code"] == "release_protected" and deleted == []
    assert "r1" in resp.get_json()["error"]
    assert api.client.delete("/runs/r1", query_string={"force": "1"}).status_code == 200 and deleted == ["r1"]
    assert api.client.delete("/runs/r2").status_code == 200 and deleted == ["r1", "r2"]  # not marked


def test_bulk_deletion_keeps_protected_reports(monkeypatch):
    from loadlens_app.blueprints import dashboard
    from loadlens_app.load_tests import service

    refs = {"rid-a": {"run_id": "rid-a", "run_name": "a", "service": "s", "protected": True},
            "rid-b": {"run_id": "rid-b", "run_name": "b", "service": "s", "protected": False}}
    monkeypatch.setattr(dashboard, "_lookup_report_refs", lambda ids: {i: refs[i] for i in ids if i in refs})
    deleted = []
    monkeypatch.setattr(core, "_delete_run_data", deleted.append)
    runs = [types.SimpleNamespace(run_id="x", report_url="/reports/rid-a", report_job_id=None, options={}),
            types.SimpleNamespace(run_id="y", report_url="/reports/rid-b", report_job_id=None, options={})]
    monkeypatch.setattr(service, "report_id_of", lambda run: {"x": "rid-a", "y": "rid-b"}[run.run_id])
    done, failed = service._delete_reports(runs)
    assert done == ["b"] and deleted == ["b"]
    assert failed == [{"run_id": "x", "run_name": "a", "code": "release_protected", "message": failed[0]["message"]}]
    assert "a" in failed[0]["message"]


def test_report_refs_carry_the_protection(monkeypatch):
    from loadlens_app.blueprints import dashboard

    db = MarksDb().report("r1").report("r2")
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11")

    class Cur(MarksCursor):
        def execute(self, query, params=None):
            text = flat(render(query))
            if "SAVEPOINT" in text:
                return
            if "run_index WHERE run_id = ANY" in text:
                self.rows = [("rid-r1", "r1", "checkout"), ("rid-r2", "r2", "checkout")]
                return
            super().execute(query, params)

    db.cursor = lambda: Cur(db)
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: db)
    refs = dashboard._lookup_report_refs(["rid-r1", "rid-r2"])
    assert refs["rid-r1"]["protected"] is True and refs["rid-r2"]["protected"] is False


def test_service_deletion_removes_marks_in_its_own_savepoint(monkeypatch):
    from loadlens_app.load_tests import store as lt_store
    from tests.test_perf_run_index import Conn

    monkeypatch.setattr(core, "_load_settings_runtime_data", lambda: {})
    monkeypatch.setattr(core, "_load_metrics_runtime_data", lambda: {})
    monkeypatch.setattr(lt_store, "_store", lt_store.MemoryLoadTestStore())
    conn = Conn(("FROM \"public\".\"release_marks\"", RuntimeError("relation does not exist")))
    monkeypatch.setattr(core, "_ts_conn", lambda: conn)
    core._delete_service_data("shop", "checkout")  # survives the failure
    texts = conn.texts()
    assert 'DELETE FROM "public"."release_marks" WHERE service = %s' in texts
    assert texts.count("SAVEPOINT service_delete") == 4
    assert "ROLLBACK TO SAVEPOINT release_marks_delete" in texts and conn.commits == 1


# ---- launch form and watcher ------------------------------------------------------------------------------------


def test_plan_run_trims_and_checks_the_release(monkeypatch):
    from loadlens_app.load_tests import service

    # Only the release part of plan_run: it is checked before the instances, whose empty card fails next.
    base = {"service": "checkout", "instances": [{}]}
    monkeypatch.setattr(service, "load_catalog", lambda: service.Catalog(tools={}, generators={}))
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    with pytest.raises(service.LaunchError) as exc:
        service.plan_run({**base, "release": "x" * 81}, "engineer1")
    assert exc.value.key == "server.release.name_too_long"
    with pytest.raises(service.LaunchError) as exc:
        service.plan_run({**base, "release": "25.11"}, "engineer1")
    assert exc.value.key == "lt.launch.choose_generator"  # the release passed


def test_run_views_show_the_release():
    from loadlens_app.load_tests import service
    from loadlens_app.load_tests.store import RUN_DONE, RunRecord

    run = RunRecord(run_id="x", run_tag="t-x", run_name="nightly", service="checkout", project_area="shop", test_type="step",
                    status=RUN_DONE, owner="engineer1", options={"release": "25.11"})
    cat = service.Catalog(tools={}, generators={})
    assert service.run_view(run, cat, with_report=False)["release"] == "25.11"
    assert service.history_view(run, cat, {}, "engineer1", True, False)["release"] == "25.11"
    run.options = {}
    assert service.run_view(run, cat, with_report=False)["release"] == ""


def test_the_watcher_marks_the_finished_report(monkeypatch):
    from loadlens_app import jobs
    from loadlens_app.load_tests import watcher
    from loadlens_app.load_tests.store import RUN_REPORTING, RunRecord

    marks = []
    monkeypatch.setattr(watcher, "_set_release_mark", lambda name, release, author: marks.append((name, release, author)))
    job = types.SimpleNamespace(status=jobs.JOB_STATUS_DONE, page_url="/reports/rid", report_url="")
    monkeypatch.setattr(jobs.job_store, "get", lambda job_id: job)
    run = RunRecord(run_id="x", run_tag="t-x", run_name="nightly-x", service="checkout", project_area="shop", test_type="step",
                    status=RUN_REPORTING, owner="engineer1", options={"release": "25.11"}, report_job_id="job-1")
    w = watcher.Watcher.__new__(watcher.Watcher)
    w._follow_report(run)
    assert marks == [("nightly-x", "25.11", "engineer1")] and run.report_url == "/reports/rid"

    # Without a release nothing is marked; a failure is logged and the run stays done.
    run2 = RunRecord(run_id="y", run_tag="t-y", run_name="n2", service="checkout", project_area="", test_type="step",
                     status=RUN_REPORTING, owner="", options={}, report_job_id="job-1")
    w._follow_report(run2)
    assert len(marks) == 1

    def broken(*args):
        raise RuntimeError("database is down")

    monkeypatch.setattr(watcher, "_set_release_mark", broken)
    run3 = RunRecord(run_id="z", run_tag="t-z", run_name="n3", service="checkout", project_area="", test_type="step",
                     status=RUN_REPORTING, owner="", options={"release": "25.11"}, report_job_id="job-1")
    w._follow_report(run3)
    assert run3.status == watcher.RUN_DONE


def test_the_watcher_mark_uses_the_rule_of_the_module(monkeypatch):
    from loadlens_app.load_tests import watcher

    db = MarksDb().report("nightly-x")
    monkeypatch.setattr(core, "_ts_conn", lambda: db)
    import AI.db_store as store_module

    monkeypatch.setattr(store_module, "_ensure_llm_reports_table", lambda conn, cfg: None)
    state = watcher._set_release_mark("nightly-x", "25.11", "engineer1")
    assert state["accepted"] is True and state["author"] == "engineer1"


# ---- /compare_baseline ------------------------------------------------------------------------------------------


def test_compare_baseline_answers_the_mode_that_found_the_run(monkeypatch):
    from app import create_app
    from tests.test_smoke_api import RecordingConnection
    from loadlens_app.blueprints import compare

    created = T0 + timedelta(hours=1)
    conn = RecordingConnection(fetchone_results=[("checkout", created, "step", "")])
    monkeypatch.setattr(compare, "_ts_conn", lambda: conn)
    monkeypatch.setattr(compare, "_resolve_services_filter", lambda area: [])
    calls = []

    def fake(conn, schema, table, **kwargs):
        calls.append(kwargs)
        return BaselineRun("rel-a", "checkout", "pass", T0, mode="previous_release", release="25.10", run_id="rid-a")

    monkeypatch.setattr(release_marks, "find_baseline_run", fake)
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    resp = client.get("/compare_baseline", query_string={"run_name": "cur", "mode": "pinned"})
    assert resp.status_code == 200
    assert resp.get_json() == {"run_name": "rel-a", "service": "checkout", "verdict": "pass",
                               "created_at": T0.isoformat(), "mode": "previous_release", "release": "25.10"}
    assert calls[0]["mode"] == "pinned" and calls[0]["test_type"] == "step" and calls[0]["service"] == "checkout"

    conn._cursor._fetchone = [("checkout", created, "step", "")]
    monkeypatch.setattr(release_marks, "find_baseline_run", lambda *a, **k: None)
    resp = client.get("/compare_baseline", query_string={"run_name": "cur", "mode": "previous_release"})
    assert resp.status_code == 404 and resp.get_json()["mode"] == "previous_release"
    assert "принятый прогон прошлого релиза" in resp.get_json()["error"]
    bad = client.get("/compare_baseline", query_string={"run_name": "cur", "mode": "best"}).get_json()["error"]
    assert "pinned, previous_release, previous_success, previous" in bad


# ---- pipeline baseline context ----------------------------------------------------------------------------------


def test_baseline_context_names_the_baseline_by_its_mode(monkeypatch):
    from AI import pipeline

    found = {}

    def fake(conn, schema, table, **kwargs):
        found.update(kwargs)
        return BaselineRun("base", "checkout", "pass", T0, mode="pinned", run_id="rid-base")

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(pipeline, "find_baseline_run", fake)
    monkeypatch.setattr(pipeline, "_db_connect", lambda cfg: Conn())
    monkeypatch.setattr(pipeline, "load_run_metric_stats", lambda *args: {})
    meta = {"service": "checkout", "run_name": "cur", "test_type": "step", "project_area": "shop"}
    ctx = pipeline._load_baseline_context({"llm": {}}, meta, {"schema": "public"}, True, T0.timestamp())
    assert found["mode"] == "pinned"  # the default
    assert ctx["mode"] == "pinned" and ctx["release"] == "" and ctx["run_id"] == "rid-base"
    assert ctx["basis"] == "закреплённый эталон сервиса"

    monkeypatch.setattr(pipeline, "find_baseline_run", lambda *a, **k: BaselineRun(
        "rel", "checkout", "pass", T0, mode="previous_release", release="25.10"))
    ctx = pipeline._load_baseline_context({"llm": {}}, meta, {"schema": "public"}, True, T0.timestamp())
    assert ctx["basis"] == "принятый прогон релиза «25.10»" and ctx["release"] == "25.10"
    domain = pipeline._baseline_for_domain({**ctx, "stats": {}}, "jvm", {"sections": []})
    assert domain["mode"] == "previous_release" and domain["release"] == "25.10" and domain["basis"] == ctx["basis"]

    monkeypatch.setattr(pipeline, "find_baseline_run", lambda *a, **k: BaselineRun(
        "old", "checkout", "pass", T0, mode="previous_success"))
    ctx = pipeline._load_baseline_context({"llm": {}}, meta, {"schema": "public"}, True, T0.timestamp())
    assert "предыдущий успешный прогон" in ctx["basis"]
    ctx = pipeline._load_baseline_context({"llm": {"baseline": {"mode": "previous_success"}}}, meta, {"schema": "public"},
                                          True, T0.timestamp())
    assert ctx["basis"] == "" and "basis" not in pipeline._baseline_for_overall({**ctx, "stats": {}}, {})

    monkeypatch.setattr(pipeline, "find_baseline_run", lambda *a, **k: None)
    ctx = pipeline._load_baseline_context({"llm": {}}, meta, {"schema": "public"}, True, T0.timestamp())
    assert ctx["available"] is False and "успешный" in ctx["reason"]


def test_baseline_settings_default_to_pinned_and_reject_unknown_modes():
    from AI import pipeline

    assert pipeline._baseline_settings({"llm": {}})["mode"] == "pinned"
    assert pipeline._baseline_settings({"llm": {"baseline": {"mode": "Previous"}}})["mode"] == "previous"
    assert pipeline._baseline_settings({"llm": {"baseline": {"mode": "best"}}})["mode"] == "pinned"


# ---- Confluence header -----------------------------------------------------------------------------------------


def test_confluence_header_shows_the_marks():
    from loadlens_app import confluence_export

    rows = confluence_export._mark_rows({"release": "25.11", "accepted": True, "baseline": True})
    assert rows == [("Релиз", "25.11, принятый прогон"), ("Эталон", "закреплённый эталон сервиса")]
    assert confluence_export._mark_rows({"release": "25.11", "accepted": False, "baseline": False}) == [("Релиз", "25.11")]
    assert confluence_export._mark_rows(None) == []
    body = confluence_export.build_page_body(
        run_name="r1", service="checkout", source_url="", llm_rows=[], engineer_html="", charts=[],
        step_table=None, run_meta={"service": "checkout", "start_ms": 0, "end_ms": 0},
        mark={"release": "25.11", "accepted": True, "baseline": False},
    )
    assert "25.11, принятый прогон" in body
    plain = confluence_export.build_page_body(
        run_name="r1", service="checkout", source_url="", llm_rows=[], engineer_html="", charts=[],
        step_table=None, run_meta={"service": "checkout", "start_ms": 0, "end_ms": 0},
    )
    assert "Релиз" not in plain


def test_confluence_reads_the_mark_and_survives_a_broken_database(monkeypatch):
    from loadlens_app import confluence_export

    db = MarksDb().report("r1")
    release_marks.set_release(db, "public", "llm_reports", "r1", "25.11")
    monkeypatch.setattr(confluence_export, "_ts_conn", lambda: db)
    assert confluence_export.load_release_mark("r1")["release"] == "25.11"

    def broken():
        raise RuntimeError("down")

    monkeypatch.setattr(confluence_export, "_ts_conn", broken)
    assert confluence_export.load_release_mark("r1") is None
