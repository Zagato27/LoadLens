# SPDX-License-Identifier: Apache-2.0
"""settings_runtime.json over settings.py: sections the pages save whole replace, the others merge key by key."""

import copy
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before update_page)
import update_page  # noqa: E402
from AI.data_sources import validate_data_sources, validate_domain_sources  # noqa: E402

GRAFANA = {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "bearer", "token": "t"}}}
INFLUX = {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "o", "token": "t"}}
PROM = {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}}
SETTINGS = {
    "llm": {"provider": "gigachat", "max_domain_workers": 3},
    "data_sources": {"grafana": GRAFANA, "influx": INFLUX},
    "domain_sources": {
        "default": {"source": "grafana", "datasource_uid": "prom-uid", "datasource_name": "Prometheus"},
        "kafka": {"source": "influx", "bucket": "telegraf"},
    },
    "per_area": {"shop": {
        "domain_sources": {"default": {"source": "grafana", "datasource_uid": "shop-uid"}},
        "queries": {"jvm": {"title": "Shop JVM"}},
    }},
}
# What the settings pages write: whole sections; the Kafka binding and the Influx source were deleted.
RUNTIME = {
    "llm": {"provider": "openai"},
    "data_sources": {"grafana": GRAFANA, "prom": PROM},
    "domain_sources": {"default": {"source": "prom"}},
    "per_area": {"shop": {
        "domain_sources": {"default": {"source": "prom"}},
        "queries": {"jvm": {"labels": []}},
    }},
}


def test_sections_saved_whole_replace_the_settings_file():
    merged = core.overlay_runtime(copy.deepcopy(SETTINGS), copy.deepcopy(RUNTIME))
    assert merged["domain_sources"] == {"default": {"source": "prom"}}
    assert merged["data_sources"] == {"grafana": GRAFANA, "prom": PROM}
    assert merged["per_area"]["shop"]["domain_sources"] == {"default": {"source": "prom"}}
    # The rest still merges key by key.
    assert merged["llm"] == {"provider": "openai", "max_domain_workers": 3}
    assert merged["per_area"]["shop"]["queries"] == {"jvm": {"title": "Shop JVM", "labels": []}}
    # The result is a valid catalogue with valid bindings: saving the bindings card again works.
    validate_data_sources(merged["data_sources"])
    validate_domain_sources(merged["domain_sources"], merged["data_sources"])
    validate_domain_sources(merged["per_area"]["shop"]["domain_sources"], merged["data_sources"])


def test_settings_file_sections_stay_when_the_runtime_has_none():
    merged = core.overlay_runtime(copy.deepcopy(SETTINGS), {"llm": {"provider": "openai"}, "per_area": {"shop": {"title": "Shop"}}})
    assert merged["domain_sources"] == SETTINGS["domain_sources"]
    assert merged["data_sources"] == SETTINGS["data_sources"]
    assert merged["per_area"]["shop"]["domain_sources"] == SETTINGS["per_area"]["shop"]["domain_sources"]
    assert merged["per_area"]["shop"]["title"] == "Shop"


def test_overlay_does_not_share_the_runtime_objects():
    runtime = copy.deepcopy(RUNTIME)
    merged = core.overlay_runtime(copy.deepcopy(SETTINGS), runtime)
    merged["domain_sources"]["default"]["source"] = "changed"
    merged["per_area"]["shop"]["domain_sources"]["default"]["source"] = "changed"
    assert runtime["domain_sources"]["default"]["source"] == "prom"
    assert runtime["per_area"]["shop"]["domain_sources"]["default"]["source"] == "prom"


def test_worker_refresh_replaces_bindings_too(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps(RUNTIME), encoding="utf-8")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    for key in ("llm", "data_sources", "domain_sources", "per_area"):
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(SETTINGS[key]))
    assert update_page.refresh_runtime_config() is True
    assert core.CONFIG["domain_sources"] == {"default": {"source": "prom"}}
    assert "influx" not in core.CONFIG["data_sources"]
    assert core.CONFIG["llm"]["max_domain_workers"] == 3
    once = json.dumps(core.CONFIG, sort_keys=True, default=str)
    assert update_page.refresh_runtime_config() is True
    assert json.dumps(core.CONFIG, sort_keys=True, default=str) == once
