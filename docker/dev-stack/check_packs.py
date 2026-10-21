# SPDX-License-Identifier: Apache-2.0
"""Runs the query packs against docker/dev-stack and reports which queries return data.

Every query goes through LoadLens's own preview (the «Проверить» button): Prometheus directly,
InfluxQL through the Grafana proxy, Flux directly against InfluxDB 1.8. Usage:

    python docker/dev-stack/check_packs.py                    # all packs of the running profiles
    python docker/dev-stack/check_packs.py --packs node-exporter,db-redis-exporter --minutes 30
    python docker/dev-stack/check_packs.py --json results.json

Exit code 1 when a required query of a checked pack returned nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loadlens_app  # noqa: E402,F401  (before update_page users)
from AI.query_packs import load_catalog, render_pack  # noqa: E402
from loadlens_app.query_preview import preview_metric_query  # noqa: E402

SOURCES = {
    "prom": {"title": "dev Prometheus", "type": "prometheus", "prometheus": {"url": "http://127.0.0.1:19090"}},
    "graf": {
        "title": "dev Grafana",
        "type": "grafana_proxy",
        "grafana": {"base_url": "http://127.0.0.1:13000", "auth": {"method": "basic", "username": "admin", "password": "devstack-admin"}},
    },
    "influx": {"title": "dev InfluxDB", "type": "influxdb", "influxdb": {"url": "http://127.0.0.1:18086", "org": "devstack", "token": "devstack:devstack"}},
    "vm": {"title": "dev VictoriaMetrics", "type": "prometheus", "prometheus": {"url": "http://127.0.0.1:18428"}},
    "vm-labels": {"title": "dev VictoriaMetrics, relabeled", "type": "prometheus", "prometheus": {"url": "http://127.0.0.1:18429"}},
}
BINDINGS = {
    "prom": {"source": "prom"},
    "graf-influxql": {"source": "graf", "datasource_uid": "devstack-influx-k6", "database": "k6"},
    "influx-flux": {"source": "influx", "bucket": "k6/autogen"},
    "vm": {"source": "vm"},
    "vm-labels": {"source": "vm-labels"},
}
# pack -> (binding, variables, compose profile that produces its metrics)
PLAN: dict[str, tuple[str, dict[str, str], str]] = {
    "lt-k6-influxql": ("graf-influxql", {}, "load"),
    "lt-k6-influx2-flux": ("influx-flux", {}, "load"),
    "lt-k6-prometheus-rw": ("prom", {"matchers": 'testid="devstack"'}, "load"),
    "lt-k6-prometheus-native": ("prom", {"matchers": 'testid="devstack-native"'}, "load"),
    "lt-locust-prometheus": ("prom", {}, "locust"),
    "lt-gatling-graphite-victoriametrics": ("vm", {"simulation": "checkoutsimulation"}, "gatling"),
    "lt-gatling-graphite-victoriametrics-labels": ("vm-labels", {}, "gatling"),
    "jvm-micrometer": ("prom", {"service_label": "job", "matchers": 'job="keycloak"'}, "core"),
    "http-micrometer": ("prom", {"service_label": "job", "matchers": 'job="keycloak"'}, "core"),
    "runtime-go": ("prom", {"matchers": 'job=~"prometheus|node|postgres|pgbouncer|redis"'}, "core"),
    "http-go": ("prom", {"metric_prefix": "prometheus_http_request_duration_seconds", "matchers": 'job="prometheus"'}, "core"),
    "runtime-python": ("prom", {"matchers": 'job="pyapp"'}, "core"),
    "http-python-fastapi": ("prom", {"matchers": 'job="pyapp"'}, "core"),
    "node-exporter": ("prom", {"matchers": 'job="node"'}, "core"),
    # The stand has no separate load generator host: the Docker VM node stands in for it.
    "lt-generators-node-exporter": ("prom", {"matchers": 'job="node"'}, "core"),
    "db-postgres-exporter": ("prom", {}, "core"),
    "db-pgbouncer": ("prom", {}, "core"),
    "db-redis-exporter": ("prom", {}, "core"),
    "kafka-exporter": ("prom", {}, "kafka"),
    "db-arangodb": ("prom", {"matchers": 'job="arangodb"'}, "arangodb"),
}


def check_pack(pack_id: str, start_ts: float, end_ts: float) -> dict:
    binding, variables, profile = PLAN[pack_id]
    pack = load_catalog().pack(pack_id)
    cfg = {
        "data_sources": SOURCES,
        "domain_sources": {"default": {"source": "prom"}, pack.domain: BINDINGS[binding]},
        "default_params": {"step": "30s"},
    }
    results = []
    for query in render_pack(pack, variables).queries:
        preview = preview_metric_query(
            cfg, domain=pack.domain, lang=pack.language, query=query.query,
            label_keys=list(query.keys), start_ts=start_ts, end_ts=end_ts,
        )
        results.append({
            "label": query.label,
            "required": query.required,
            "ok": preview.ok,
            "series": preview.series_count,
            "points": preview.point_count,
            "sample": preview.series[:3],
            "message": preview.message,
        })
    return {
        "pack": pack_id,
        "profile": profile,
        "ok": all(item["ok"] for item in results if item["required"]),
        "empty": [item["label"] for item in results if not item["ok"]],
        "queries": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packs", default="", help="comma-separated pack ids (default: every pack of the plan)")
    parser.add_argument("--profiles", default="core,load,kafka,locust,gatling,arangodb", help="compose profiles that are running")
    parser.add_argument("--minutes", type=float, default=15.0, help="window that ends now")
    parser.add_argument("--json", default="", help="write the full results to this file")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    profiles = {item.strip() for item in args.profiles.split(",") if item.strip()}
    wanted = [item.strip() for item in args.packs.split(",") if item.strip()] or [
        pack_id for pack_id, (_, _, profile) in PLAN.items() if profile in profiles
    ]
    end_ts = time.time()
    start_ts = end_ts - args.minutes * 60
    reports = [check_pack(pack_id, start_ts, end_ts) for pack_id in wanted]
    for report in reports:
        status = "OK  " if report["ok"] else "FAIL"
        print(f"{status} {report['pack']}")
        for item in report["queries"]:
            mark = "+" if item["ok"] else ("!" if item["required"] else "-")
            detail = f"{item['series']} series, {item['points']} points" if item["ok"] else item["message"]
            print(f"   {mark} {item['label']}: {detail}")
    if args.json:
        Path(args.json).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [report["pack"] for report in reports if not report["ok"]]
    print(f"\npacks: {len(reports)}, required queries failed in: {', '.join(failed) or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
