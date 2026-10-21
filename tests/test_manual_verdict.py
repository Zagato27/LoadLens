# SPDX-License-Identifier: Apache-2.0
"""Manual verdicts of reports: the report_verdicts table, the verdict in force in every reader (dashboard, /runs,
/llm_reports, baseline lookup, forecast list, projects, Confluence) and the /runs/<run>/verdict API.
Offline: a fake database keeps the final row and the manual changes of one run.

Verdicts are codes (AI/verdicts.py). The fake database keeps the Russian labels where a database written before
the codes holds them (the final row, older history rows), so every reader is checked on legacy rows as well."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import db_store  # noqa: E402
from AI.verdicts import code_sql  # noqa: E402

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def flat(text) -> str:
    return " ".join(str(text).split())


class VerdictCursor:
    def __init__(self, db):
        self.db = db
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = flat(query if isinstance(query, str) else render(query))
        self.db.log.append((text, params))
        self.rows = []
        if "to_regclass" in text:
            self.rows = [("public.report_verdicts" if self.db.table_exists else None,)]
        elif text.startswith("CREATE TABLE"):
            if self.db.create_fails:
                raise RuntimeError("permission denied for schema public")
            self.db.table_exists = True
        elif "domain = 'final' ORDER BY created_at DESC LIMIT 1" in text:
            if self.db.final is not None and params[0] == self.db.run_name:
                self.rows = [self.db.final]
        elif "SELECT created_at, verdict, previous_verdict" in text:
            rows = [r for r in self.db.changes if r["run_name"] == params[0]]
            self.rows = [
                (r["created_at"], r["verdict"], r["previous_verdict"], r["auto_verdict"], r["reason"], r["author"])
                for r in sorted(rows, key=lambda r: r["created_at"], reverse=True)
            ][: params[1]]
        elif text.startswith("INSERT INTO"):
            run_name, run_id, verdict, previous, automatic, reason, author = params
            self.db.changes.append({
                "run_name": run_name, "run_id": run_id, "verdict": verdict, "previous_verdict": previous,
                "auto_verdict": automatic, "reason": reason, "author": author,
                "created_at": T0 + timedelta(minutes=len(self.db.changes)),
            })

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class VerdictDb:
    def __init__(self, auto="Провал", run_name="nightly-7", table_exists=True, create_fails=False):
        self.run_name = run_name
        self.final = (auto, "rid-7") if auto is not None else None
        self.changes = []
        self.table_exists = table_exists
        self.create_fails = create_fails
        self.log = []
        self.autocommit = False
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.commits += 1
        return False

    def cursor(self):
        return VerdictCursor(self)

    def rollback(self):
        pass

    def close(self):
        pass


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


# ---- SQL helpers and the table ----------------------------------------------------------------------------


def test_cte_reads_the_newest_change_and_is_empty_without_the_table():
    ready = flat(db_store.manual_verdicts_cte("perf", True))
    assert 'FROM "perf".report_verdicts ORDER BY run_name, created_at DESC, id DESC' in ready
    assert "DISTINCT ON (run_name)" in ready and "WHERE verdict IS NOT NULL" in ready  # a reset row hides the manual one
    missing = flat(db_store.manual_verdicts_cte("perf", False))
    assert missing.startswith("manual_verdicts AS (SELECT NULL::text AS run_name") and "WHERE FALSE" in missing
    # Codes: old labels in any column map to their code and a run without any verdict is no_data.
    assert db_store.effective_verdict_sql("f") == code_sql("COALESCE(mv.verdict, f.sla_verdict, f.verdict)")
    assert db_store.effective_verdict_sql("r", fallback=False) == code_sql("COALESCE(mv.verdict, r.sla_verdict, r.verdict)")
    assert "WHEN 'Провал' THEN 'fail'" in db_store.effective_verdict_sql("f") and "ELSE 'no_data' END" in db_store.effective_verdict_sql("f")


def test_ensure_creates_the_table_and_marks_it_ready(monkeypatch):
    monkeypatch.setattr(db_store, "_MANUAL_VERDICTS_READY", set())
    db = VerdictDb(table_exists=False)
    db_store.ensure_report_verdicts_table(db, "public")
    statements = [text for text, _ in db.log]
    assert any('CREATE TABLE IF NOT EXISTS "public"."report_verdicts"' in s and "previous_verdict TEXT" in s for s in statements)
    assert any('ON "public"."report_verdicts" (run_name, created_at DESC, id DESC)' in s for s in statements)
    assert db.autocommit is False and "public" in db_store._MANUAL_VERDICTS_READY
    db_store.ensure_report_verdicts_table(db, "public")
    assert len(db.log) == len(statements)  # once per process


def test_a_failed_create_leaves_the_probe_to_decide(monkeypatch):
    monkeypatch.setattr(db_store, "_MANUAL_VERDICTS_READY", set())
    db = VerdictDb(table_exists=False, create_fails=True)
    db_store.ensure_report_verdicts_table(db, "public")
    assert "public" not in db_store._MANUAL_VERDICTS_READY
    assert db_store.manual_verdicts_ready(db, "public") is False
    db.table_exists = True  # created later by an administrator
    assert db_store.manual_verdicts_ready(db, "public") is True
    probes = [params for text, params in db.log if "to_regclass" in text]
    assert probes == [('"public".report_verdicts',)] * 2
    assert db_store.manual_verdicts_ready(db, "public") is True and len([t for t, _ in db.log if "to_regclass" in t]) == 2


def test_deployment_schema_and_app_role_know_the_table():
    schema_sql = flat((ROOT_DIR / "docker" / "initdb" / "02_schema.sql").read_text(encoding="utf-8"))
    assert "CREATE TABLE IF NOT EXISTS public.report_verdicts (" in schema_sql
    assert "CREATE INDEX IF NOT EXISTS idx_report_verdicts_run_created ON public.report_verdicts (run_name, created_at DESC, id DESC);" in schema_sql
    role = (ROOT_DIR / "docker" / "initdb" / "03_app_role.sh").read_text(encoding="utf-8")
    assert " report_verdicts " in role.split("LOADLENS_TABLES=", 1)[1].splitlines()[0]
    _, tables = core._run_tables({})
    assert db_store.REPORT_VERDICTS_TABLE in tables  # deleted and renamed with the run


# ---- save and load --------------------------------------------------------------------------------------


def test_set_change_and_reset_keep_the_history():
    db = VerdictDb(auto="Провал")  # a final row saved before the codes
    state = db_store.load_verdict_state(db, "public", "llm_reports", "nightly-7")
    assert state == {"run_name": "nightly-7", "run_id": "rid-7", "automatic": "fail", "verdict": "fail", "manual": None, "history": []}

    # A legacy label is accepted on input; all three columns store codes.
    state = db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", "Успешно", "  сбой генератора  ", "ivanov")
    assert state["verdict"] == "pass" and state["automatic"] == "fail"
    assert state["manual"] == {"verdict": "pass", "previous_verdict": "fail", "auto_verdict": "fail",
                               "reason": "сбой генератора", "author": "ivanov", "changed_at": T0.isoformat()}
    assert (db.changes[0]["verdict"], db.changes[0]["previous_verdict"], db.changes[0]["auto_verdict"]) == ("pass", "fail", "fail")

    state = db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", "risk", "уточнили", "petrov")
    assert state["verdict"] == "risk" and state["manual"]["previous_verdict"] == "pass"

    state = db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", None, "", "ivanov")
    assert state["verdict"] == "fail" and state["manual"] is None
    assert [(h["verdict"], h["previous_verdict"]) for h in state["history"]] == [
        (None, "risk"), ("risk", "pass"), ("pass", "fail"),
    ]
    assert db.changes[0]["run_id"] == "rid-7"


def test_history_rows_saved_with_labels_are_read_as_codes():
    db = VerdictDb(auto="Провал")
    db.changes = [
        {"run_name": "nightly-7", "run_id": "rid-7", "verdict": "Есть риски", "previous_verdict": "Провал",
         "auto_verdict": "Провал", "reason": "старая запись", "author": "ivanov", "created_at": T0},
        {"run_name": "nightly-7", "run_id": "rid-7", "verdict": None, "previous_verdict": "Успешно",
         "auto_verdict": "", "reason": "", "author": "petrov", "created_at": T0 - timedelta(days=1)},
    ]
    state = db_store.load_verdict_state(db, "public", "llm_reports", "nightly-7")
    assert state["verdict"] == "risk" and state["manual"]["verdict"] == "risk"
    assert [(h["verdict"], h["previous_verdict"], h["auto_verdict"]) for h in state["history"]] == [
        ("risk", "fail", "fail"), (None, "pass", ""),
    ]
    # The same verdict given as a code or a label changes nothing.
    for same in ("risk", "Есть риски", " есть  РИСКИ "):
        with pytest.raises(ValueError):
            db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", same, "x", "ivanov")


@pytest.mark.parametrize(
    ("verdict", "reason", "error"),
    [
        ("Отлично", "x", ValueError),
        ("passed", "x", ValueError),  # only codes and the old labels, no heuristic on input
        ("Успешно", "   ", ValueError),
        ("pass", "x" * (db_store.MANUAL_VERDICT_REASON_MAX + 1), ValueError),
        ("Провал", "уже провал", ValueError),  # the verdict in force already (automatic «Провал»)
        ("fail", "уже провал", ValueError),  # the same as a code
        (None, "", ValueError),  # nothing to return: the verdict is automatic
    ],
)
def test_changes_that_change_nothing_or_lack_a_reason_are_refused(verdict, reason, error):
    db = VerdictDb(auto="Провал")
    with pytest.raises(error):
        db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", verdict, reason, "ivanov")
    assert db.changes == []


def test_a_run_without_a_final_report_has_no_verdict_to_change():
    db = VerdictDb(auto=None)
    assert db_store.load_verdict_state(db, "public", "llm_reports", "nightly-7") is None
    with pytest.raises(LookupError):
        db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", "Успешно", "x", "ivanov")


def test_without_the_table_the_state_has_no_history(monkeypatch):
    monkeypatch.setattr(db_store, "_MANUAL_VERDICTS_READY", set())
    db = VerdictDb(auto="Есть риски", table_exists=False)
    state = db_store.load_verdict_state(db, "public", "llm_reports", "nightly-7")
    assert state["verdict"] == "risk" and state["history"] == []
    assert not any("report_verdicts WHERE run_name" in text for text, _ in db.log)


# ---- readers ------------------------------------------------------------------------------------------


class Scripted:
    """Logs statements and answers fetchone/fetchall from scripted lists (in order)."""

    def __init__(self, one=None, many=None):
        self.log = []
        self.one = list(one or [])
        self.many = list(many or [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        outer = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, query, params=None):
                outer.log.append((flat(query if isinstance(query, str) else render(query)), params))

            def fetchone(self):
                return outer.one.pop(0) if outer.one else None

            def fetchall(self):
                return outer.many.pop(0) if outer.many else []

        return Cursor()

    def rollback(self):
        pass

    def close(self):
        pass


def test_the_baseline_lookup_counts_the_manual_verdict():
    conn = Scripted(one=[("release-2.4", "demo", "Успешно", T0)])  # a legacy row
    previous = db_store.find_previous_run(conn, "public", "llm_reports", exclude_run_name="nightly-8", service="demo", before=T0)
    assert previous.verdict == "pass"
    text = conn.log[0][0]
    assert text.startswith("WITH manual_verdicts AS (") and 'FROM "public".report_verdicts' in text
    assert "LEFT JOIN manual_verdicts mv ON mv.run_name = r.run_name" in text
    # previous_success: the code of the verdict in force, so «Успешно» rows and «pass» rows both qualify.
    assert "AND " + flat(code_sql("COALESCE(mv.verdict, r.sla_verdict, r.verdict)")) + " = 'pass'" in text

    conn = Scripted(one=[None])
    db_store.find_previous_run(conn, "public", "llm_reports", exclude_run_name="nightly-8")
    text = conn.log[0][0]
    assert "), manual_verdicts AS (" in text and "FROM \"public\".\"llm_reports\" r CROSS JOIN current c LEFT JOIN manual_verdicts mv" in text


def test_the_forecast_list_shows_the_verdict_in_force():
    conn = Scripted(many=[[("rid", "nightly-7", "demo", "step", "", T0, "Успешно", None)]])
    rows = db_store.list_final_reports(conn, "public", "llm_reports", ["demo"])
    assert rows[0].verdict == "pass"
    text = conn.log[0][0]
    assert flat(code_sql("COALESCE(mv.verdict, latest.sla_verdict, latest.verdict)")) + " AS verdict" in text
    assert "LEFT JOIN manual_verdicts mv ON mv.run_name = latest.run_name WHERE latest.rn = 1" in text


def test_projects_read_the_verdict_in_force(monkeypatch):
    from loadlens_app import projects

    conn = Scripted(many=[[("demo", T0, "Успешно")]])
    monkeypatch.setattr(projects, "_ts_conn", lambda: conn)
    stats = projects._report_stats(["demo"])
    assert stats["demo"].last_verdict == "pass"
    text = conn.log[0][0]
    assert text.startswith("WITH manual_verdicts AS (") and flat(code_sql("COALESCE(mv.verdict, latest.sla_verdict, latest.verdict)")) in text


def test_confluence_page_names_who_changed_the_verdict_and_why():
    from loadlens_app import confluence_export

    row = {"domain": "final", "verdict": "Провал", "llm_verdict": "Провал", "sla_verdict": "Провал",
           "parsed": {"verdict": "Провал", "verdict_rationale": "p95 выше порога"},
           "manual_verdict": {"verdict": "Есть риски", "reason": "известная деградация стенда", "author": "ivanov",
                              "changed_at": "2026-10-17T12:00:00+00:00"}}
    body = confluence_export.render_llm_domain_body(row)
    assert "<th>Вердикт</th><td>Есть риски</td>" in body
    assert "<th>Изменён вручную</th><td>ivanov, 2026-10-17 12:00 UTC. Причина: известная деградация стенда</td>" in body
    assert "<th>Автоматический вердикт</th><td>Провал</td>" in body
    plain = confluence_export.render_llm_domain_body({**row, "manual_verdict": None})
    assert "<th>Вердикт</th><td>Провал</td>" in plain and "Вердикт LLM" not in plain  # the same verdict twice says nothing
    assert "«Недостаточно данных»" in confluence_export.sla_criteria_block({"sla_details": {}})
    # A report saved with codes renders the same page.
    coded = {"domain": "final", "verdict": "fail", "llm_verdict": "fail", "sla_verdict": "fail",
             "parsed": {"verdict": "fail", "verdict_rationale": "p95 выше порога"},
             "manual_verdict": {**row["manual_verdict"], "verdict": "risk"}}
    assert confluence_export.render_llm_domain_body(coded) == body


# ---- HTTP -----------------------------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app import response_cache
    from loadlens_app.blueprints import dashboard

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    db = VerdictDb(auto="Провал")
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: db)
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    invalidated = []
    monkeypatch.setattr(response_cache, "invalidate", lambda name: invalidated.append(name))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), db, invalidated


def test_verdict_api_sets_reads_and_resets(api):
    client, db, invalidated = api
    state = client.get("/runs/nightly-7/verdict").get_json()
    assert state["verdict"] == "fail" and state["automatic"] == "fail" and state["reason_max"] == 1000
    assert state["choices"] == list(db_store.MANUAL_VERDICTS) == ["pass", "risk", "fail", "no_data"]

    # The old label of the README example still works; the answer and the stored row carry codes.
    resp = client.put("/runs/nightly-7/verdict", json={"verdict": "Успешно", "reason": "сбой генератора"})
    assert resp.status_code == 200 and resp.get_json()["manual"]["verdict"] == "pass" and resp.get_json()["verdict"] == "pass"
    assert db.changes[-1]["verdict"] == "pass" and db.changes[-1]["previous_verdict"] == "fail"
    assert db.commits >= 1 and invalidated  # the dashboard cache is dropped
    assert db.changes[-1]["author"] == ""  # sign-in is off in tests

    assert client.put("/runs/nightly-7/verdict", json={"verdict": "pass", "reason": "ещё раз"}).status_code == 400
    assert client.put("/runs/nightly-7/verdict", json={"verdict": "Успешно", "reason": "ещё раз"}).status_code == 400
    assert client.put("/runs/nightly-7/verdict", json={"verdict": "fail", "reason": ""}).status_code == 400
    assert client.put("/runs/nightly-7/verdict", json={"verdict": "maybe", "reason": "x"}).status_code == 400
    assert client.put("/runs/nightly-7/verdict", json={"verdict": 1, "reason": "x"}).status_code == 400
    assert client.put("/runs/nightly-7/verdict", data="x", content_type="text/plain").status_code == 400
    assert client.put("/runs/other/verdict", json={"verdict": "Провал", "reason": "x"}).status_code == 404
    assert client.get("/runs/other/verdict").status_code == 404

    resp = client.delete("/runs/nightly-7/verdict", json={})  # the report page sends an empty object
    assert resp.status_code == 200 and resp.get_json()["verdict"] == "fail" and resp.get_json()["manual"] is None
    assert client.delete("/runs/nightly-7/verdict").status_code == 400  # already automatic
    assert client.put("/runs/nightly-7/verdict", json={"verdict": "risk", "reason": "x"}).status_code == 200
    assert client.delete("/runs/nightly-7/verdict", json={"reason": 5}).status_code == 400
    assert client.delete("/runs/nightly-7/verdict", json={"reason": "проверили"}).status_code == 200
    assert db.changes[-1]["reason"] == "проверили" and db.changes[-1]["verdict"] is None


def test_verdict_endpoints_are_for_engineers_with_audit():
    from loadlens_app.auth.models import Role
    from loadlens_app.auth.policy import POLICY

    assert POLICY["dashboard.get_run_verdict"].role == Role.VIEWER
    assert POLICY["dashboard.set_run_verdict"].role == Role.ENGINEER and POLICY["dashboard.set_run_verdict"].audit == "report.verdict"
    assert POLICY["dashboard.reset_run_verdict"].audit == "report.verdict_reset"


def test_report_rows_carry_the_manual_verdict(api, monkeypatch):
    client, db, _ = api
    from loadlens_app.blueprints import dashboard

    db_store.save_manual_verdict(db, "public", "llm_reports", "nightly-7", "Есть риски", "деградация стенда", "ivanov")
    rows = [("nightly-7", "demo", 1, 2, "final", "Провал", "{}", {"verdict": "Провал"}, None, "Провал", {}, None, T0, "step"),
            ("nightly-7", "demo", 1, 2, "jvm", "Провал", "{}", {"verdict": "Провал"}, None, None, {}, None, T0, "step")]
    conn = Scripted(many=[rows])
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)
    monkeypatch.setattr(dashboard, "load_verdict_state", lambda c, s, t, run: db_store.load_verdict_state(db, s, t, run))
    data = client.get("/llm_reports", query_string={"run_name": "nightly-7"}).get_json()
    final = next(item for item in data if item["domain"] == "final")
    assert final["verdict"] == "risk" and final["auto_verdict"] == "fail"
    assert final["llm_verdict"] == "fail" and final["sla_verdict"] == "fail" and final["parsed"]["verdict"] == "fail"
    assert final["manual_verdict"]["reason"] == "деградация стенда" and final["manual_verdict"]["author"] == "ivanov"
    assert final["manual_verdict"]["verdict"] == "risk"
    jvm = next(item for item in data if item["domain"] == "jvm")
    assert jvm["verdict"] == "fail" and jvm["sla_verdict"] is None and "manual_verdict" not in jvm


def test_runs_items_and_dashboard_rows_carry_the_manual_verdict():
    from loadlens_app.blueprints import dashboard

    row = ("nightly-7", T0, T0, "demo", "Есть риски", "Провал", "Провал", T0, "step", "rid", "", 120, 1,
           "Есть риски", "деградация стенда", "ivanov", T0)
    item = dashboard._runs_item(row)
    assert item["verdict"] == "risk" and item["sla_verdict"] == "fail" and item["llm_verdict"] == "fail"
    assert item["manual_verdict"] == {"verdict": "risk", "reason": "деградация стенда", "author": "ivanov", "changed_at": T0.isoformat()}
    assert dashboard._runs_item(row[:13])["manual_verdict"] is None  # an automatic verdict
    assert dashboard._manual_verdict_view([None, None, None, None]) is None
