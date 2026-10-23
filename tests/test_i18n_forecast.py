# SPDX-License-Identifier: Apache-2.0
"""Localization of the forecasting pages (plans/i18n-plan.md 2.5, row 6): /forecasting and /forecasting/<id>.

The templates render in en without Russian text, the builders of forecast_report.js speak the UI language (plural
forms by LL.tn, numbers and dates by LL.fmt, labels of the server codes from the forecast namespace), and without a
cookie the Russian output stays as it was. Texts that come from the server payload (step labels, plan messages,
notes) are data here and stay as the server sent them.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from tests.i18n_helpers import assert_no_cyrillic, use_locale, visible_text
from tests.js_harness import run_js

FILES = ["static/js/common.js", "static/js/ui.js", "static/js/forecast_report.js"]
NBSP = "\u00a0"  # ru-RU group separator of Intl.NumberFormat


@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _page(client, url, lang):
    use_locale(client, lang)
    response = client.get(url)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1).strip()
    main = html[html.index('<main id="main"'):html.index("</main>")]
    return html, title, main[main.index(">") + 1:]


# ---- templates ---------------------------------------------------------------------------------------------


def test_forecasting_list_in_en_has_no_russian_text(client):
    html, title, page = _page(client, "/forecasting", "en")
    assert re.search(r'<html lang="en"', html)
    assert title == "Forecasting"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert "<h2>Forecasting</h2>" in page
    assert re.findall(r"<th>([^<]*)</th>", page) == ["Report", "Service", "Test type", "Report created", "Verdict", "Load model"]
    # The page declares the forecast namespace for forecasting.js.
    assert '"forecast.list.model_closed":"closed"' in html


def test_forecast_report_page_in_en_has_no_russian_text(client):
    html, title, page = _page(client, "/forecasting/abc", "en")
    assert re.search(r'<html lang="en"', html)
    assert title == "Capacity forecast"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert 'aria-label="Section navigation"' in page and '<a href="/forecasting">Forecasting</a>' in page
    assert '<h1 id="forecastTitle">Capacity forecast</h1>' in page
    assert ">Add report to Confluence</button>" in page and ">Open in Confluence</a>" in page
    assert '"forecast.answer.holds":"will hold"' in html


def test_forecasting_pages_stay_russian_without_a_cookie(client):
    html, title, page = _page(client, "/forecasting", None)
    assert re.search(r'<html lang="ru"', html) and title == "Прогнозирование"
    assert "<th>Отчёт создан</th>" in page and "<th>Модель нагрузки</th>" in page
    assert "(и число VU для закрытой модели). Выберите отчёт.</p>" in page
    html, title, page = _page(client, "/forecasting/abc", None)
    assert title == "Прогноз мощностей"
    assert 'aria-label="Навигация по разделу"' in page
    assert '<span class="current" aria-current="page">Отчёт</span>' in page
    assert ">Добавить отчёт в Confluence</button>" in page
    assert "Сколько нагрузки выдержит сервис по модели USL, построенной на ступенях теста." in page


# ---- forecast_report.js --------------------------------------------------------------------------------------


def _services():
    rows = [
        ("api", 1, 2, 0.8, 0.1, 0.74, 0.82, 0.46, False),
        ("db", 2, 3, 0.8, 0.2, 0.7, 0.9, 0.6, True),
        ("web", 1, 1, 0.8, 0.05, 0.21, 0.23, None, True),
        ("auth", 1, 1, 0.8, 0.05, 0.13, 0.14, None, True),
        ("queue", 1, 1, 0.8, 0.04, 0.08, 0.085, None, True),
        ("cache", 1, 1, 0.8, 0.03, 0.06, 0.07, None, True),
        ("gw", 1, 1, 0.8, 0.02, 0.05, 0.06, None, True),
    ]
    keys = ("name", "instances", "instances_needed", "cpu_ceiling", "cpu_base", "cpu_measured", "cpu_at_target", "cpu_after", "reliable")
    return [dict(zip(keys, row)) for row in rows]


def _step(number, rps, response_ms, stalls=0, after_drop=False, degraded=False):
    return {
        "number": number, "label": f"Step {number}", "source": "step_detector", "rps": rps, "concurrency": rps * response_ms / 1000,
        "response_ms": response_ms, "p95_ms": response_ms * 2.2, "stalls": stalls, "samples": 10, "after_drop": after_drop,
        "degraded": degraded,
    }


def _point(minute, rps, response_ms, role, vus=None):
    return {
        "time_iso": f"2026-10-01T13:{minute:02d}:00+03:00", "rps": rps, "response_ms": response_ms, "p95_ms": response_ms * 2.2,
        "concurrency": rps * response_ms / 1000, "vus": vus, "role": role,
    }


# A forecast as GET /forecast/<id> returns it (loadlens_app/forecast.py), with every block filled.
PAYLOAD = {
    "load_model": "open",
    "window_start_iso": "2026-10-01T13:00:00+03:00",
    "cutoff_iso": "2026-10-01T13:55:00+03:00",
    "cutoff_kind": "rps_drop",
    "bin_minutes": 1.0,
    "fit": {"sigma": 0.012, "kappa": 0.00023, "r2": 0.874, "n_peak": 70.2, "x_max": 880.4},
    "tested_rps": 600.0,
    "peak_rps": 1612.0,
    "samples_used": 41,
    "excluded_edges": 2,
    "excluded_stalls": 9,
    "think_time_s": 0.0,
    "knee": {"rps": 420.0, "response_ms": 35.0},
    "instability": {
        "onset_iso": "2026-10-01T13:41:00+03:00", "onset_rps": 500.0, "onset_step": 5, "onset_label": "Step 5", "stalls": 9,
        "samples": 21, "worst_response_ms": 1300.0, "worst_p95_ms": 2600.0, "period_min": 2.0,
    },
    "limit": {"rps": 500.0, "kind": "instability"},
    "limit_applies": True,
    "safe": {"rps": 437.5, "source": "cpu", "service": "api"},
    "scale": {
        "end_rps": 636.0,
        "tested_rps": 600.0,
        "zones": [
            {"from_rps": 0.0, "to_rps": 437.5, "tone": "ok"},
            {"from_rps": 437.5, "to_rps": 500.0, "tone": "warn"},
            {"from_rps": 500.0, "to_rps": 636.0, "tone": "fail"},
        ],
        "marks": [{"rps": 437.5, "kind": "safe"}, {"rps": 500.0, "kind": "limit"}, {"rps": 600.0, "kind": "tested"}],
    },
    "target": {"rps": 450, "headroom_pct": 10.0, "concurrency": 14.2, "response_ms": 31.5, "p95_ms": 69.3},
    "answer": {
        "verdict": "scale",
        "confidence": "low",
        "confidence_reasons": ["low_r2", "instability", "extrapolation", "cpu_fit", "cpu_extrapolation", "scaling_untested"],
    },
    "resources": {
        "status": "ok", "message": "", "ceiling_pct": 80.0, "ceiling_source": "sla", "target_rps": 1450, "last_step": 4,
        "steps_used": 4, "services": _services(),
        "nodes": [{"name": "n1", "cpu_measured": 0.62, "cpu_at_target": 0.86}, {"name": "n2", "cpu_measured": 0.2, "cpu_at_target": 0.21}],
        "nodes_message": "", "capacity_rps": 437.5, "bottleneck": "api", "scaled_capacity_rps": 875.0, "next_bottleneck": "db",
        "limit_service": "api", "limit_cpu": 0.9, "limit_cause": "api",
    },
    "points": [
        _point(0, 20.0, 1000.0, "edge"),
        _point(5, 100.0, 30.0, "fit", vus=5),
        _point(25, 300.0, 33.0, "fit"),
        _point(41, 500.0, 3000.0, "stall"),
        _point(58, 150.0, 30.0, "after_cutoff"),
    ],
    "steps": [
        _step(1, 100.0, 30.0), _step(2, 200.0, 31.0), _step(3, 300.0, 33.0), _step(4, 400.0, 34.0),
        _step(5, 500.0, 165.0, stalls=5), _step(6, 150.0, 30.0, after_drop=True),
    ],
    "curve": [{"concurrency": n, "rps": n * 30.0, "response_ms": 33.0 + n, "p95_ms": (33.0 + n) * 2.2} for n in (3.0, 10.0, 20.0, 40.0)],
    "sla_target_rps": 400.0,
    "sla_p95_ms": 9000.0,
    "notes": [],
}

FINDING = {"ref": "F1", "domain": "jvm", "start_time": "2026-10-01T13:40:00", "end_time": "2026-10-01T13:45:00", "summary": "GC pauses"}
VIEW = {
    "status": "ready",
    "explanation": {
        "headline": "Headline", "cause": "Cause", "confidence": "medium", "key_facts": [{"value": "1", "label": "fact"}],
        "scaling": {"verdict": "partly", "reason": "Reason"}, "scaling_risks": ["a", "b", "c", "d"], "next_checks": [],
        "evidence": [{"finding_id": "F1", "role": "cause"}, {"finding_id": "F9", "role": "context"}],
    },
    "findings": [FINDING],
    "model": "model",
    "generated_at": "2026-10-06T07:30:00Z",
    "check": {"status": "verified", "claims_matched": 3, "claims_total": 4, "unmatched": [], "unknown_findings": ["F9"]},
    "llm_problem": "",
    "message": "",
}


def _blocked():
    """Stuck service (CPU above the ceiling with no load): the answer is blocked, no reasons."""
    data = copy.deepcopy(PAYLOAD)
    db = data["resources"]["services"][1]
    db.update(instances_needed=None, cpu_ceiling=0.6, cpu_base=0.65, cpu_after=None)
    data["answer"] = {"verdict": "blocked", "confidence": "high", "confidence_reasons": []}
    data["limit"] = {"rps": 500.0, "kind": "model"}
    return data


def _variants():
    out = [PAYLOAD, _blocked()]
    for verdict in ("holds", "tight"):
        data = copy.deepcopy(PAYLOAD)
        data["answer"] = {"verdict": verdict, "confidence": "medium", "confidence_reasons": ["low_r2"]}
        data["instability"] = None
        data["knee"] = None
        data["fit"].update(kappa=0, x_max=None)
        data["load_model"] = "closed"
        data["think_time_s"] = 1.25
        data["cutoff_kind"] = "test_end"
        for step in data["steps"]:
            step["source"] = "time_buckets"
        out.append(data)
    no_plan = copy.deepcopy(PAYLOAD)
    no_plan["answer"] = None
    no_plan["target"] = None
    no_plan["limit"] = {"rps": None, "kind": "none"}
    no_plan["resources"] = dict(no_plan["resources"], status="missing_setting", message="Server message")
    out.append(no_plan)
    out.append(_above_maximum())
    return out


def _above_maximum():
    """The report maximum is step 4; steps 5 and 6 are evenly slow (no stalls), so the limit is the report SLA."""
    data = copy.deepcopy(PAYLOAD)
    data.update(
        instability=None, excluded_stalls=0, excluded_degraded=20, cutoff_kind="test_end", tested_rps=400.0,
        limit={"rps": 500.0, "kind": "sla"}, limit_applies=True, safe={"rps": 400.0, "source": "limit", "service": None},
        answer={"verdict": "blocked", "confidence": "medium", "confidence_reasons": ["extrapolation"]},
        target={"rps": 550, "headroom_pct": None, "concurrency": None, "response_ms": None, "p95_ms": None},
    )
    data["fit"].update(kappa=0, x_max=None, n_peak=None)
    data["resources"] = dict(data["resources"], status="missing_setting", message="Server message", limit_cause=None)
    data["scale"] = {
        "end_rps": 636.0,
        "tested_rps": 400.0,
        "zones": [
            {"from_rps": 0.0, "to_rps": 400.0, "tone": "ok"},
            {"from_rps": 400.0, "to_rps": 500.0, "tone": "warn"},
            {"from_rps": 500.0, "to_rps": 636.0, "tone": "fail"},
        ],
        "marks": [{"rps": 400.0, "kind": "safe"}, {"rps": 400.0, "kind": "tested"}, {"rps": 500.0, "kind": "limit"}],
    }
    data["points"] = [_point(5, 100.0, 30.0, "fit"), _point(25, 300.0, 33.0, "fit"), _point(45, 500.0, 150.0, "degraded")]
    data["steps"] = [
        _step(1, 100.0, 30.0), _step(2, 200.0, 31.0), _step(3, 300.0, 33.0), _step(4, 400.0, 34.0),
        _step(5, 500.0, 150.0, degraded=True), _step(6, 600.0, 160.0, degraded=True),
    ]
    data["sla_ceiling"] = {
        "rps": 400.0, "step_number": 4, "step_label": "Step 4", "end_iso": "2026-10-01T13:40:00+03:00", "degraded_rps": 500.0,
        "degraded_checks": [{"name": "p95_latency", "title": "p95 latency"}], "limit_rps": 500.0,
    }
    return data


BUILDERS = """
(() => {
  const I = LL.forecastReport._internals;
  const P = %s;
  const view = (v) => [v.caption].concat(v.datasets.map((d) => d.label), v.datasets.flatMap((d) => d.data.flatMap((p) => p.tip || [])));
  const pages = P.map((p) => [
    p.answer ? I.answerHtml(p, 20) : '', I.scaleHtml(p), I.scaleCaption(p), I.stepsHtml(p), I.resourcesHtml(p),
    I.factItems(p), p.answer ? I.confidenceHtml(p) : '', I.methodItems(p), p.answer ? view(I.latencyView(p)) : [],
    p.answer ? view(I.throughputView(p)) : [], I.controlsHtml(p, 450), I.shellHtml(p, 450)
  ]);
  const view0 = %s;
  const views = [view0, Object.assign({}, view0, { explanation: null, findings: [] }), Object.assign({}, view0, { llm_problem: 'LLM is off' })];
  return { pages: pages, explain: views.map(I.explanationHtml), unavailable: I.unavailable('Server error') };
})()
"""


def _render(locale):
    return run_js(FILES, BUILDERS % (json.dumps(_variants()), json.dumps(VIEW)), locale=locale)


def test_page_builders_in_en_have_no_russian_text():
    out = _render("en")
    offending = sorted(set(re.findall(r'[^"<>]*[\u0400-\u04ff][^"<>]*', json.dumps(out, ensure_ascii=False))))
    assert not offending, offending[:20]
    page = out["pages"][0]
    assert "Target 450 RPS: will hold after scaling" in page[0] and "Confidence: low" in page[0]
    assert "Add: api 1 → 2 pods, pod CPU at the target ≈ 46% instead of 82%." in page[0]
    assert '<th class="num">Pod CPU, step 4</th>' in page[4]
    assert "2 more services stay at or below 8.5% pod CPU at the target." in page[4]
    assert "No stalls: steps 1–4, 100 → 400 RPS, mean 30–34 ms." in page[5]
    assert ("Stalls began at 13:41 (Step 5, ≈ 500 RPS): 9 of 21 points of 1 min, mean up to 1.3 s, p95 up to 2.6 s. "
            "They recur about every 2 min.") in page[5]
    assert "Peak RPS: 500 averaged per step (Step 5), 1,612 per 1 min." in page[5]
    assert "κ = 2.30e-4: the cost of keeping parallel requests coherent; because of κ, RPS falls after the peak. The peak is at concurrency ≈ 70." in page[7]
    assert "+25% over SLA" in page[10] and 'title="CPU load of one pod above which another pod is needed"' in page[10]
    blocked = out["pages"][1]
    assert "Target 450 RPS: will not hold" in blocked[0] and "pods will not help" in blocked[4]
    assert "Target 450 RPS: at the limit" in out["pages"][3][0]
    assert "Intervals: mean" in out["pages"][2][8]
    explained = out["explain"][0]
    assert "<strong>Adding pods will help partly.</strong>" in explained and "Model confidence: medium" in explained
    assert "numbers checked against the data: 3 of 4; links to unknown findings: F9" in explained
    assert "06/10/2026, 07:30" in explained and "F9: not in the report" in explained
    assert "Analyse with AI" in out["explain"][1]
    assert "Forecast unavailable" in out["unavailable"]


def test_page_builders_in_ru_keep_the_russian_text():
    out = _render("ru")
    page = out["pages"][0]
    assert "Цель 450 RPS: выдержит после масштабирования" in page[0] and "Достоверность: низкая" in page[0]
    assert "Добавить: api 1 → 2 пода, CPU пода при цели ≈ 46 % вместо 82 %." in page[0]
    assert "Потолок CPU пода 80 % (из SLA); CPU пода — прямая по 4 ступеням без провалов." in page[4]
    assert "Ещё 2 сервиса при цели не выше 8,5 % CPU пода." in page[4]
    assert ("С 13:41 (Step 5, ≈ 500 RPS) начались провалы: 9 из 21 точки по 1 мин, среднее до 1,3 с, p95 до 2,6 с. "
            "Повторяются примерно каждые 2 мин.") in page[5]
    assert f"Максимум RPS: в среднем за ступень — 500 (Step 5), за 1 мин — 1{NBSP}612." in page[5]
    assert "κ = 2,30e-4 — потери на согласование параллельных запросов: из-за κ после пика RPS падает. Пик — при параллелизме ≈ 70." in page[7]
    assert "+25 % к SLA" in page[10]
    assert "Интервалы: среднее" in out["pages"][2][8]
    assert "<strong>Добавление подов поможет частично.</strong>" in out["explain"][0]
    assert "06.10.2026, 07:30" in out["explain"][0]


@pytest.mark.parametrize(
    "locale, texts",
    [
        ("en", {
            "steps": ["✓ no stalls", "✗ above the report maximum"],
            "answer": ["Limit ≈ 500 RPS: the report shows the SLA thresholds broken from this load (p95 latency)."],
            "scale": ["above the report maximum", "SLA broken"],
            "caption": ["red: above the report maximum, where the test broke the SLA thresholds"],
            "facts": [
                "No stalls: steps 1–4, 100 → 400 RPS, mean 30–34 ms.",
                "Report maximum: step 4, 400 RPS, the last load at which the SLA latency and error thresholds hold. "
                "Above it (step 5, 500 RPS) p95 latency broke; those points are left out of the model.",
            ],
            "method": ["above the report maximum (20)"],
            "latency": ["Red fill: load above the report maximum", "Above the report maximum", "Limit: report SLA",
                        "above the report maximum: SLA thresholds broken"],
        }),
        ("ru", {
            "steps": ["✓ без провалов", "✗ выше максимума отчёта"],
            "answer": ["Предел ≈ 500 RPS: с этой нагрузки в отчёте нарушены пороги SLA (p95 latency)."],
            "scale": ["выше максимума отчёта", "нарушен SLA"],
            "caption": ["красная — выше максимума отчёта: там в тесте нарушены пороги SLA"],
            "facts": [
                "Без провалов: ступени 1–4, 100 → 400 RPS, среднее 30–34 мс.",
                "Максимум отчёта — ступень 4, 400 RPS: последняя нагрузка, на которой держатся пороги задержек и ошибок SLA. "
                "Выше (ступень 5, 500 RPS) нарушено: p95 latency — эти точки в модель не вошли.",
            ],
            "method": ["выше максимума отчёта — 20"],
            "latency": ["Красная заливка — нагрузка выше максимума отчёта", "Выше максимума отчёта", "Предел: SLA отчёта",
                        "выше максимума отчёта: пороги SLA нарушены"],
        }),
    ],
)
def test_steps_above_the_report_maximum_are_red(locale, texts):
    page = _render(locale)["pages"][5]
    steps = page[3]
    cards = re.findall(r'<div class="forecast-step is-(\w+)"', steps)
    assert cards == ["ok", "ok", "ok", "ok", "fail", "fail"]
    blocks = {
        "steps": steps, "answer": page[0], "scale": page[1], "caption": page[2], "facts": "\n".join(page[5]),
        "method": "\n".join(page[7]), "latency": "\n".join(page[8]),
    }
    for name, expected in texts.items():
        for text in expected:
            assert text in blocks[name], (name, text, blocks[name])
    assert re.search(r'forecast-scale-zone is-fail[^>]*title="[^"]*' + re.escape(texts["scale"][0]), page[1])


@pytest.mark.parametrize(
    "locale, expected",
    [
        ("ru", ["0 подов", "1 под", "2 пода", "5 подов", "11 подов", "21 под", "22 пода", "112 подов"]),
        ("en", ["0 pods", "1 pod", "2 pods", "5 pods", "11 pods", "21 pods", "22 pods", "112 pods"]),
    ],
)
def test_pod_plural_follows_the_locale(locale, expected):
    expr = "[0, 1, 2, 5, 11, 21, 22, 112].map(LL.forecastReport._internals.podsText)"
    assert run_js(FILES, expr, locale=locale) == expected


@pytest.mark.parametrize(
    "locale, expected",
    [
        ("ru", [["0,5 мс", "12 мс", "1,5 с", "—"], ["5,6 %", "12 %"], f"1{NBSP}234{NBSP}567,89"]),
        ("en", [["0.5 ms", "12 ms", "1.5 s", "—"], ["5.6%", "12%"], "1,234,567.89"]),
    ],
)
def test_numbers_follow_the_locale(locale, expected):
    expr = """(() => {
      const I = LL.forecastReport._internals;
      return [[0.5, 12.2, 1500, null].map(I.formatMs), [5.55, 12.3].map(I.formatPct), I.formatNumber(1234567.891, 2)];
    })()"""
    assert run_js(FILES, expr, locale=locale) == expected


def test_stored_step_labels_reach_the_forecast_in_the_reader_language(client, monkeypatch):
    """A report stores its step labels in its own language; the forecast page (outside the report body) shows them
    in the reader's language, and without a cookie the Russian labels come back unchanged (s9 e2e finding)."""
    import pandas as pd

    from tests.test_smoke_api import _forecast_frames, _forecast_report, _patch_forecast

    frames = _forecast_frames()
    start = frames["RPS"].index[0]
    end = frames["RPS"].index[-1] + pd.Timedelta(minutes=1)
    middle = start + (end - start) / 2
    report = _forecast_report(start, end)
    report.context["load_steps"] = [
        {"label": "ступень 1 · ≈120 RPS", "start_iso": start.isoformat(), "end_iso": middle.isoformat(), "source": "step_detector"},
        {"label": "ступень 2 · ≈240 RPS", "start_iso": middle.isoformat(), "end_iso": end.isoformat(), "source": "step_detector"},
    ]
    _patch_forecast(monkeypatch, report, frames)
    labels = {}
    for lang in (None, "en"):
        use_locale(client, lang)
        response = client.get("/forecast/abc")
        assert response.status_code == 200
        labels[lang] = [step["label"] for step in response.get_json()["steps"]]
    assert labels[None] == ["ступень 1 · ≈120 RPS", "ступень 2 · ≈240 RPS"]
    assert labels["en"] == ["step 1 · ≈120 RPS", "step 2 · ≈240 RPS"]
