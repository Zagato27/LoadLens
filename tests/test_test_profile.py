"""Single test-profile source: aliases, modes, step detection, prompt guidance, baseline filter."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AI import pipeline as pipeline_module  # noqa: E402
from AI.db_store import find_previous_run  # noqa: E402
from AI.prompt_assembly import TEST_TYPE_OVERLAYS, build_analysis_guidance  # noqa: E402
from AI.test_profile import is_capacity_like, mode_of, normalize_test_type, profile_for, resolve_step_detection  # noqa: E402
from loadlens_app.forecast import forecast_availability  # noqa: E402
from tests.test_smoke_api import RecordingConnection  # noqa: E402


@pytest.mark.parametrize(
    "raw, test_type, mode, peak, forecast",
    [
        ("step", "step", "capacity", True, True),
        ("Ступенчатый", "step", "capacity", True, True),
        ("soak", "soak", "stability", False, False),
        ("Стабильность", "soak", "stability", False, False),
        ("night-stability-run", "soak", "stability", False, False),
        ("spike", "spike", "spike", False, False),
        ("Всплески", "spike", "spike", False, False),
        ("stress", "stress", "stress", True, True),
        ("", "", "capacity", True, True),
        ("garbage", "", "capacity", True, True),
    ],
)
def test_profile_for_normalizes_aliases_and_modes(raw, test_type, mode, peak, forecast):
    profile = profile_for(raw)
    assert profile.test_type == test_type
    assert profile.mode == mode
    assert profile.peak_performance_applicable is peak
    assert profile.forecast_allowed is forecast
    assert normalize_test_type(raw) == test_type
    stored = profile.to_dict()
    assert stored["mode"] == mode and stored["test_type"] == raw.strip()
    assert mode_of(stored) == mode


def test_stored_profiles_of_old_reports_keep_their_mode():
    assert mode_of({"mode": "stability", "test_type": "soak"}) == "stability"
    assert mode_of({"mode": "capacity", "test_type": "stress"}) == "capacity"
    assert mode_of({"test_type": "spike"}) == "spike"
    assert mode_of(None) == "capacity"
    assert is_capacity_like("stress") and is_capacity_like("capacity") and not is_capacity_like("spike")


def test_resolve_step_detection_respects_config_only_for_capacity_like_modes():
    enabled = {"step_detection_enabled": True}
    disabled = {"step_detection_enabled": False}
    assert resolve_step_detection(profile_for("step"), {}) is True
    assert resolve_step_detection(profile_for("step"), disabled) is False
    assert resolve_step_detection(profile_for("stress"), disabled) is False
    # soak and spike use their own steps whatever the config says
    assert resolve_step_detection(profile_for("soak"), enabled) is False
    assert resolve_step_detection(profile_for("spike"), enabled) is False
    # untyped legacy runs: the detector runs only when the config enables it
    assert resolve_step_detection(profile_for(""), {}) is False
    assert resolve_step_detection(profile_for(""), enabled) is True


def test_sla_guidance_degradation_rule_only_for_capacity_modes():
    sla = {"target_rps": 200, "max_p95_ms": 500, "max_performance_query": "LT RPS"}
    rule = "даже если после него была деградация"
    assert rule in build_analysis_guidance(sla, None, "step")
    assert rule in build_analysis_guidance(sla, None, "stress")
    soak = build_analysis_guidance(sla, None, "soak")
    spike = build_analysis_guidance(sla, None, "spike")
    assert rule not in soak and rule not in spike
    assert "удерживать весь тест" in soak and "hold_window" in soak
    assert "уровень всплеска" in spike and "recovery" in spike
    domain = build_analysis_guidance(sla, None, "soak", domain="jvm")
    assert "окне удержания" in domain and "sla_step" not in domain


def test_overlays_cover_every_type_and_no_placeholders():
    assert set(TEST_TYPE_OVERLAYS) == {"step", "soak", "spike", "stress"}
    for text in TEST_TYPE_OVERLAYS.values():
        assert "X%" not in text and "recovery_time_s" not in text


def test_legacy_prompt_layers_are_gone():
    import update_page  # imported late: importing it before loadlens_app hits the app's import cycle

    for name in ("_build_final_prompts", "_test_type_overlays", "_sla_prompt_block", "_LLM_PROMPT_DOMAINS"):
        assert not hasattr(update_page, name)
    for name in ("_test_profile_from_type", "_augment_prompt_with_test_profile"):
        assert not hasattr(pipeline_module, name)


@pytest.mark.parametrize("test_type, allowed", [("step", True), ("stress", True), ("soak", False), ("spike", False)])
def test_forecast_availability_follows_profile(test_type, allowed):
    status = forecast_availability(test_type, None, {})
    if allowed:
        assert status.code != "not_applicable"
    else:
        assert status.code == "not_applicable" and not status.available


def test_find_previous_run_filters_by_test_type():
    created = datetime(2024, 10, 31, 11, 0, tzinfo=timezone.utc)
    conn = RecordingConnection(fetchone_results=[("soak-1", "demo", "Успешно", created)])
    before = datetime(2024, 11, 1, tzinfo=timezone.utc)
    find_previous_run(
        conn, "public", "llm_reports", exclude_run_name="soak-2", service="demo",
        project_area="payments", before=before, test_type="Стабильность",
    )
    sql_text, params = conn.log[0]
    assert "LOWER(TRIM(COALESCE(r.test_type, ''))) = ANY(%s)" in sql_text
    assert sql_text.index("r.test_type") < sql_text.index("r.project_area")
    assert params[:3] == ("demo", "soak-2", before)
    assert "soak" in params[3] and "стабильность" in params[3] and "" not in params[3]
    assert params[4] == "payments"

    conn = RecordingConnection(fetchone_results=[None])
    find_previous_run(conn, "public", "llm_reports", exclude_run_name="x", service="demo", before=before, test_type="step")
    assert "" in conn.log[0][1][3]  # untyped legacy runs stay eligible for step baselines

    conn = RecordingConnection(fetchone_results=[None])
    find_previous_run(conn, "public", "llm_reports", exclude_run_name="x", service="demo", before=before)
    assert "r.test_type" not in conn.log[0][0] and len(conn.log[0][1]) == 3
