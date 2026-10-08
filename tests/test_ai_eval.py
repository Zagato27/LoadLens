"""Offline eval harness: fixture metrics and DB loading without a live GigaChat call."""

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AI.eval import evaluate_analysis, evaluate_fixture, evaluate_run_records, load_eval_records
from tests.test_smoke_api import RecordingConnection

FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "ai_eval" / "sample.json"


def test_eval_fixture_metrics_are_deterministic():
    metrics = evaluate_fixture(FIXTURE_PATH)
    assert metrics["domain"] == "final"
    assert metrics["verdict"] == "Есть риски"
    assert metrics["verdict_match"] == 1.0
    assert metrics["findings_total"] == 3
    assert metrics["findings_verified"] == 1
    assert metrics["findings_unverified"] == 1
    assert metrics["findings_qualitative"] == 1
    assert metrics["findings_verified_share"] == pytest.approx(1 / 3)
    assert metrics["findings_unverified_share"] == pytest.approx(1 / 3)
    assert metrics["findings_qualitative_share"] == pytest.approx(1 / 3)
    assert metrics["findings_agree"] == 0
    assert metrics["findings_disagree"] == 1
    assert metrics["findings_agree_share"] == pytest.approx(0.0)
    assert metrics["findings_disagree_share"] == pytest.approx(1 / 3)
    assert metrics["actions_total"] == 3
    assert metrics["actions_empty"] == 2
    assert metrics["actions_empty_share"] == pytest.approx(2 / 3)
    assert metrics["usage"] == {
        "calls": 3,
        "prompt_tokens": 1500,
        "completion_tokens": 400,
        "elapsed_ms": 1200,
        "cost": None,
    }


def test_eval_without_feedback_leaves_verdict_match_empty():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    metrics = evaluate_analysis(payload["parsed"], feedback=[], scores=payload["scores"], domain="final")
    assert metrics["verdict_match"] is None
    assert metrics["findings_agree"] == 0
    assert metrics["findings_disagree"] == 0


def test_load_eval_records_reads_context_and_feedback():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    reports = [
        (
            "final",
            payload["parsed"],
            payload["scores"],
            payload["context"],
            "2024-01-01T12:00:00+00:00",
        )
    ]
    feedback = [
        ("nightly-1", "final", "verdict", "", "agree", ""),
        ("nightly-1", "final", "finding", "f2", "disagree", "p95 в отчёте выдуман"),
    ]
    conn = RecordingConnection(fetchall_results=[reports, feedback])
    records = load_eval_records(conn, "nightly-1")
    assert len(records) == 1
    assert records[0]["domain"] == "final"
    assert records[0]["context"]["load_steps"][1]["rps_level"] == 200
    sql_text = " ".join(str(conn.log[0][0]).split())
    assert "context" in sql_text
    assert conn.log[0][1] == ("nightly-1",)
    assert "llm_feedback" in " ".join(str(conn.log[1][0]).split())
    run_metrics = evaluate_run_records(records)
    assert run_metrics["verdict_match"] == 1.0
    assert run_metrics["actions_empty_share"] == pytest.approx(2 / 3)
    assert run_metrics["usage"]["calls"] == 3
    assert run_metrics["usage"]["prompt_tokens"] == 1500
