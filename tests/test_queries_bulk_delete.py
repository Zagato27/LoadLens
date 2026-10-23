# SPDX-License-Identifier: Apache-2.0
"""Bulk deletion of query rows on the settings page (static/js/settings.js): the row remover shared by the row
button and the selection bar, run under Node (tests/js_harness.py), and the markup, strings and styles of the bar."""

from __future__ import annotations

import json
import re

from tests.js_harness import REPO_ROOT, run_js

SETTINGS_JS = ["static/js/common.js", "static/js/ui.js", "static/js/settings.js"]
INTERNALS = "LL.settings._internals"

BLOCK = {
    "labels": ["A", "B", "C", "D"],
    "promql_queries": ["qa", "qb", "qc", "qd"],
    "label_keys_list": [["a"], ["b"], ["c"], ["d"]],
    "influxql_queries": ["", "", "ic", ""],
    "roles": {"A": "lt.rps_total", "B": "lt.errors", "C": "lt.p95"},
    "descriptions": {"A": "first", "D": "last"},
    "sources": {"B": {"source": "prom2"}},
    "title": "Tool",
}


def _remove(block: dict, indexes: list) -> dict:
    expr = (
        f"(() => {{ const block = {json.dumps(block)}; "
        f"const removed = {INTERNALS}.removeQueryRowsFromBlock(block, {json.dumps(indexes)}); "
        "return { block, removed }; })()"
    )
    return run_js(SETTINGS_JS, expr)


def test_removes_the_rows_in_any_order_and_keeps_the_arrays_aligned():
    out = _remove(BLOCK, [0, 2])
    assert out["removed"] == ["A", "C"]
    block = out["block"]
    assert block["labels"] == ["B", "D"]
    assert block["promql_queries"] == ["qb", "qd"]
    assert block["label_keys_list"] == [["b"], ["d"]]
    assert block["influxql_queries"] == ["", ""]
    assert block["roles"] == {"B": "lt.errors"}
    assert block["descriptions"] == {"D": "last"}
    assert block["sources"] == {"B": {"source": "prom2"}}
    assert block["title"] == "Tool"
    # The positions are those before the removal, whatever their order.
    assert _remove(BLOCK, [2, 0])["block"] == block


def test_ignores_repeats_and_positions_outside_the_block():
    out = _remove(BLOCK, [3, 3, 9, -1, 1.5])
    assert out["removed"] == ["D"]
    assert out["block"]["labels"] == ["A", "B", "C"] and "D" not in out["block"]["descriptions"]
    untouched = _remove(BLOCK, [])
    assert untouched["removed"] == [] and untouched["block"] == BLOCK


def test_keeps_the_role_and_note_of_a_name_another_row_still_carries():
    out = _remove({**BLOCK, "labels": ["A", "A", "C", "D"]}, [0])
    assert out["removed"] == ["A"]
    assert out["block"]["labels"] == ["A", "C", "D"]
    assert out["block"]["roles"]["A"] == "lt.rps_total" and out["block"]["descriptions"]["A"] == "first"


def test_a_shorter_array_is_cut_only_where_the_position_exists():
    block = {"labels": ["A", "B", "C"], "promql_queries": ["qa"], "label_keys_list": [[]]}
    out = _remove(block, [2])
    assert out["removed"] == ["C"]
    assert out["block"] == {"labels": ["A", "B"], "promql_queries": ["qa"], "label_keys_list": [[]]}


def test_the_list_has_a_tick_per_row_and_a_selection_bar():
    source = (REPO_ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert "line.append(queryTick(item, row), head);" in source
    assert "renderQuerySelectBar(rows);" in source
    # The row button and the bulk action share one remover.
    assert "editQueryBlock(domain, (block) => { removeQueryRowsFromBlock(block, indexes); });" in source
    assert "spliceQueryRows(domain, [index]);" in source and "spliceQueryRows(domain, indexes);" in source
    css = (REPO_ROOT / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert re.search(r"\.q-item-row > input\.q-select \{[^}]*width: auto", css)
    assert re.search(r"\.q-select-bar \{[^}]*flex-wrap: wrap", css)
    for lang in ("ru", "en"):
        messages = json.loads((REPO_ROOT / "locales" / lang / "settings.json").read_text(encoding="utf-8"))
        for key in ("select_row", "select_all", "selected_count", "delete_selected", "clear_selection",
                    "delete_selected_title", "delete_selected_sla", "delete_selected_more", "delete_selected_confirm.one"):
            assert f"settings.queries.{key}" in messages, (lang, key)


def test_the_confirmation_counts_the_rows_in_both_languages():
    expr = """[
        LL.tn('settings.queries.delete_selected_confirm', 1, { list: 'A' }),
        LL.tn('settings.queries.delete_selected_confirm', 3, { list: 'A, B, C' }),
        LL.tn('settings.queries.delete_selected_confirm', 5, { list: 'x' }),
    ]"""
    ru = run_js([], expr)
    assert ru[0].startswith("Удалить 1 запрос: A?")
    assert ru[1].startswith("Удалить 3 запроса: A, B, C?")
    assert ru[2].startswith("Удалить 5 запросов: x?")
    en = run_js([], expr, locale="en")
    assert en[0].startswith("Delete 1 query: A?") and en[1].startswith("Delete 3 queries: A, B, C?")
