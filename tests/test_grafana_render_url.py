# SPDX-License-Identifier: Apache-2.0
"""Grafana render URLs reach Grafana as configured: only a missing timeout is appended.

Repeated query keys (var-KAFKA_TOPICS=a&var-KAFKA_TOPICS=b) select several values of a multi-value
dashboard variable; rebuilding the query through a dict once kept only the last one.
"""

from data_collectors import grafana_collector as gc

BASE = "http://grafana:3000/render/d-solo/uid/kafka-lag"


def test_repeated_variables_are_kept_in_order():
    query = (
        "orgId=1&var-KAFKA_TOPICS=a.saved&var-KAFKA_TOPICS=b.saved&var-KAFKA_TOPICS=c.saved"
        "&panelId=4&tz=Europe%2FMoscow&from=1&to=2"
    )
    prepared = gc._prepare_render_url(f"{BASE}?{query}")
    assert prepared == f"{BASE}?{query}&timeout={gc._GRAFANA_RENDER_QUERY_TIMEOUT_SEC}"


def test_existing_timeout_leaves_url_untouched():
    url = f"{BASE}?var-X=a&var-X=b&timeout=30&from=1"
    assert gc._prepare_render_url(url) == url


def test_url_without_query_gets_timeout():
    assert gc._prepare_render_url(BASE) == f"{BASE}?timeout={gc._GRAFANA_RENDER_QUERY_TIMEOUT_SEC}"


def test_non_render_url_is_untouched():
    url = "http://grafana:3000/d/uid/dash?var-X=a&var-X=b"
    assert gc._prepare_render_url(url) == url
