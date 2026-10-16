"""Pure helpers of the command palette (static/js/palette.js) and the archive (static/js/archive.js), run under Node."""

import json

import pytest

from tests.js_harness import run_js

PALETTE = ["static/js/common.js", "static/js/ui.js", "static/js/palette.js"]
ARCHIVE = ["static/js/common.js", "static/js/ui.js", "static/js/archive.js"]
NOW = 1792058400000  # 2026-10-15T10:00:00Z


def js(value):
    return json.dumps(value, ensure_ascii=False)


def palette(expression, **kwargs):
    return run_js(PALETTE, f"const P = LL.palette._internals; {expression}", **kwargs)


def archive(expression, **kwargs):
    return run_js(ARCHIVE, f"const A = LL.archive._internals; {expression}", **kwargs)


# ---- palette: matching ----------------------------------------------------------------------

def test_match_rank_prefers_prefix_then_word_then_substring_then_keywords():
    items = [
        {"label": "Новый отчёт", "keywords": "создать"},
        {"label": "Архив", "keywords": "отчёты список"},
        {"label": "Отчёт по тесту"},
        {"label": "Сравнение тестов", "keywords": "compare"},
    ]
    ranks = palette(f"{js(items)}.map((it) => P.matchRank(it, 'отч'))")
    assert ranks == [1, 3, 0, -1]


def test_filter_items_is_stable_and_normalizes_case_yo_and_spaces():
    items = [{"label": "Ёлка"}, {"label": "Архив"}, {"label": "Елочный отчёт"}, {"label": "Запуск теста"}]
    labels = palette(f"P.filterItems({js(items)}, '  ЕЛ ').map((it) => it.label)")
    assert labels == ["Ёлка", "Елочный отчёт"]
    assert palette(f"P.filterItems({js(items)}, '').length") == 4


def test_every_query_word_must_match_label_keywords_or_meta():
    items = [{"label": "syn-payments-02", "meta": "payments · вчера · Провал"}, {"label": "syn-payments-06", "meta": "payments · Успешно"}]
    assert palette(f"P.filterItems({js(items)}, 'payments провал').map((it) => it.label)") == ["syn-payments-02"]


# ---- palette: sources -----------------------------------------------------------------------

def test_page_items_skip_new_report_and_duplicates_and_get_keywords():
    links = [
        {"href": "/", "label": "Дэшборд", "icon": "dashboard"},
        {"href": "/new", "label": "Новый отчёт", "icon": "file-plus"},
        {"href": "/reports", "label": "Архив", "icon": "archive"},
        {"href": "/reports", "label": "Архив", "icon": "archive"},
        {"href": "", "label": "Пусто"},
    ]
    pages = palette(f"P.pageItems({js(links)})")
    assert [p["href"] for p in pages] == ["/", "/reports"]
    assert pages[1]["key"] == "page:/reports" and pages[1]["kind"] == "page" and "отчёты" in pages[1]["keywords"]


def test_action_items_follow_role_and_theme():
    dark = palette("P.actionItems({ canCreate: true, theme: 'dark', themeMode: 'dark' })")
    assert [a["key"] for a in dark] == ["action:new", "action:theme", "action:theme-system"]
    assert dark[0]["href"] == "/new" and dark[1]["label"] == "Включить светлую тему"
    assert dark[2]["label"] == "Тема как в системе"
    # Already following the system: only the explicit switch is offered.
    light = palette("P.actionItems({ canCreate: false, theme: 'light', themeMode: 'system' })")
    assert [a["label"] for a in light] == ["Включить тёмную тему"]


def test_area_items_mark_the_active_area_and_hide_when_nothing_to_switch():
    areas = palette("P.areaItems(['demo', { id: 'devstack', title: 'Dev stack' }, { id: 'demo' }], 'devstack')")
    assert [(a["area"], a["label"], a["current"], a["meta"]) for a in areas] == [
        ("", "Все области", False, ""), ("demo", "demo", False, ""), ("devstack", "Dev stack", True, "текущая"),
    ]
    assert palette("P.areaItems([], '')") == []
    assert palette("P.areaItems(null, '')") == []
    # A cookie area missing from the configuration still shows up as the current one.
    legacy = palette("P.areaItems([], 'payments')")
    assert [(a["area"], a["current"]) for a in legacy] == [("", False), ("payments", True)]


def test_run_items_link_by_id_and_describe_the_run():
    runs = [
        {"run_name": "nightly-1", "run_id": "abc", "service": "checkout", "end_time": "2026-10-15T09:55:00Z", "verdict": "Провал"},
        {"run_name": "old run", "service": "", "verdict": ""},
        {"service": "no-name"},
    ]
    items = palette(f"P.runItems({js(runs)}, {NOW})")
    assert [i["label"] for i in items] == ["nightly-1", "old run"]
    assert items[0]["href"] == "/reports/abc" and items[0]["meta"] == "checkout · 5 мин назад · Провал"
    assert items[1]["href"] == "/reports/old%20run" and items[1]["meta"] == ""


def test_build_groups_order_notes_and_empty_groups():
    sources = {
        "pages": [{"key": "page:/", "label": "Дэшборд"}, {"key": "page:/compare", "label": "Сравнение тестов"}],
        "actions": [{"key": "action:theme", "label": "Включить светлую тему"}],
        "areas": [],
        "runs": {"status": "loading", "items": []},
    }
    groups = palette(f"P.buildGroups({js(sources)}, '')")
    assert [(g["id"], g["title"], len(g["items"]), g["note"]) for g in groups] == [
        ("pages", "Страницы", 2, ""), ("actions", "Действия", 1, ""), ("runs", "Недавние отчёты", 0, "Загружаем отчёты…"),
    ]
    sources["runs"] = {"status": "ready", "items": []}
    searching = palette(f"P.buildGroups({js(sources)}, 'сравн')")
    assert [(g["id"], g["title"], g["note"]) for g in searching] == [("pages", "Страницы", ""), ("runs", "Отчёты", "Отчётов не найдено")]
    sources["runs"] = {"status": "error", "items": []}
    assert palette(f"P.buildGroups({js(sources)}, 'zzz')") == [
        {"id": "runs", "title": "Отчёты", "items": [], "note": "Не удалось загрузить отчёты"},
    ]
    sources["runs"] = {"status": "off", "items": []}
    assert palette(f"P.buildGroups({js(sources)}, 'zzz')") == []
    # Ready recent runs with an empty query and no rows: the group disappears.
    sources["runs"] = {"status": "ready", "items": []}
    assert [g["id"] for g in palette(f"P.buildGroups({js(sources)}, '')")] == ["pages", "actions"]


# ---- palette: keyboard ----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("index", "delta", "count", "expected"),
    [(0, 1, 3, 1), (2, 1, 3, 2), (0, -1, 3, 0), (-1, 1, 3, 0), (1, 5, 3, 2), (0, 1, 0, -1)],
)
def test_move_index_clamps(index, delta, count, expected):
    assert palette(f"P.moveIndex({index}, {delta}, {count})") == expected


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"key": "k", "ctrlKey": True, "code": "KeyK"}, True),
        ({"key": "K", "metaKey": True, "code": "KeyK"}, True),
        ({"key": "л", "ctrlKey": True, "code": "KeyK"}, True),  # Russian layout
        ({"key": "л", "ctrlKey": True}, True),
        ({"key": "k", "code": "KeyK"}, False),
        ({"key": "k", "ctrlKey": True, "shiftKey": True, "code": "KeyK"}, False),
        ({"key": "k", "ctrlKey": True, "altKey": True, "code": "KeyK"}, False),
        ({"key": "j", "ctrlKey": True, "code": "KeyJ"}, False),
    ],
)
def test_open_shortcut(event, expected):
    assert palette(f"P.isOpenShortcut({js(event)})") is expected


def test_slash_shortcut_ignores_text_fields_and_modifiers():
    expression = """
      const el = (tag, type, extra) => Object.assign({ nodeType: 1, tagName: tag, type: type || '' }, extra || {});
      [
        P.isSlashShortcut({ key: '/', target: el('BODY') }),
        P.isSlashShortcut({ key: '/', target: el('BUTTON') }),
        P.isSlashShortcut({ key: '/', target: el('INPUT', 'checkbox') }),
        P.isSlashShortcut({ key: '/', target: el('INPUT', 'search') }),
        P.isSlashShortcut({ key: '/', target: el('INPUT', '') }),
        P.isSlashShortcut({ key: '/', target: el('TEXTAREA') }),
        P.isSlashShortcut({ key: '/', target: el('SELECT') }),
        P.isSlashShortcut({ key: '/', target: el('DIV', '', { isContentEditable: true }) }),
        P.isSlashShortcut({ key: '/', ctrlKey: true, target: el('BODY') }),
        P.isSlashShortcut({ key: '?', target: el('BODY') }),
      ]
    """
    assert palette(expression) == [True, True, True, False, False, False, False, False, False, False]


# ---- archive: URL state ---------------------------------------------------------------------

def test_archive_parse_query_defaults():
    assert archive("A.parseQuery('')") == {
        "page": 1, "sort": "end_time", "dir": "desc", "q": "", "service": "", "verdict": "", "testType": "",
    }


def test_archive_parse_query_reads_and_validates_every_field():
    state = archive("A.parseQuery('?q=%20night%20&service=checkout&verdict=%D0%9F%D1%80%D0%BE%D0%B2%D0%B0%D0%BB&test_type=Soak&sort=max_rps&page=3')")
    assert state == {"page": 3, "sort": "max_rps", "dir": "desc", "q": "night", "service": "checkout", "verdict": "Провал", "testType": "soak"}
    bad = archive("A.parseQuery('?verdict=maybe&sort=drop&dir=up&page=0')")
    assert (bad["verdict"], bad["sort"], bad["dir"], bad["page"]) == ("", "end_time", "desc", 1)
    assert archive("A.parseQuery('?sort=run_name').dir") == "asc"
    assert archive("A.parseQuery('?sort=service&dir=desc').dir") == "desc"
    assert archive("A.parseQuery('?page=2.5').page") == 1
    assert archive("A.parseQuery('?page=999999').page") == 1
    assert len(archive("A.parseQuery('?q=' + 'x'.repeat(500)).q")) == 200


def test_archive_build_query_omits_defaults_and_round_trips():
    assert archive("A.buildQuery(A.defaultState())") == ""
    state = {"page": 2, "sort": "run_name", "dir": "desc", "q": "50% x", "service": "НСИ", "verdict": "Успешно", "testType": "spike"}
    query = archive(f"A.buildQuery({js(state)})")
    assert query.startswith("?q=50%25+x&service=")
    assert "sort=run_name" in query and "dir=desc" in query and "page=2" in query
    assert archive(f"A.parseQuery(A.buildQuery({js(state)}))") == state
    # The default direction of a sort column is not written.
    assert archive("A.buildQuery({ ...A.defaultState(), sort: 'max_rps', dir: 'desc' })") == "?sort=max_rps"
    assert archive("A.buildQuery({ ...A.defaultState(), sort: 'service', dir: 'asc' })") == "?sort=service"


def test_archive_runs_url():
    url = archive("A.buildRunsUrl({ ...A.defaultState(), page: 3, q: 'a b', testType: 'soak' })")
    assert url == "/runs?q=a+b&test_type=soak&offset=40&limit=20&sort=end_time&dir=desc"


def test_archive_chips_and_removing_a_filter():
    state = {"page": 4, "sort": "end_time", "dir": "desc", "q": "night", "service": "svc", "verdict": "Провал", "testType": "soak"}
    chips = archive(f"A.activeChips({js(state)}, {{ service: (id) => 'Сервис ' + id, testType: (k) => LL.testTypeLabel(k) }})")
    assert chips == [
        {"key": "q", "label": "Поиск", "value": "«night»"},
        {"key": "service", "label": "Сервис", "value": "Сервис svc"},
        {"key": "verdict", "label": "Вердикт", "value": "Провал"},
        {"key": "testType", "label": "Тип теста", "value": "Долговременный (soak)"},
    ]
    assert archive("A.activeChips(A.defaultState())") == []
    removed = archive(f"A.withoutFilter({js(state)}, 'service')")
    assert removed["service"] == "" and removed["page"] == 1 and removed["q"] == "night"
    assert archive(f"A.hasFilters({js(removed)})") is True
    assert archive("A.hasFilters(A.defaultState())") is False


def test_archive_next_sort():
    assert archive("A.nextSort({ sort: 'end_time', dir: 'desc' }, 'end_time')") == {"sort": "end_time", "dir": "asc"}
    assert archive("A.nextSort({ sort: 'end_time', dir: 'desc' }, 'service')") == {"sort": "service", "dir": "asc"}
    assert archive("A.nextSort({ sort: 'service', dir: 'asc' }, 'max_rps')") == {"sort": "max_rps", "dir": "desc"}


# ---- archive: selection ---------------------------------------------------------------------

def test_archive_compare_targets_put_the_newer_run_first():
    older = {"run_name": "old", "end_time": "2026-10-01T10:00:00Z"}
    newer = {"run_name": "new one", "end_time": "2026-10-10T10:00:00Z"}
    assert archive(f"A.compareTargets([{js(older)}, {js(newer)}])") == {"runA": "new one", "runB": "old"}
    assert archive(f"A.compareTargets([{js(newer)}, {js(older)}])") == {"runA": "new one", "runB": "old"}
    assert archive(f"A.compareUrl([{js(older)}, {js(newer)}])") == "/compare?run_a=new%20one&run_b=old"
    assert archive(f"A.compareTargets([{js(older)}])") is None
    assert archive(f"A.compareUrl([{js(older)}, {js(newer)}, {js(newer)}])") == ""
    # Runs without dates keep the selection order.
    assert archive("A.compareTargets([{ run_name: 'a' }, { run_name: 'b' }])") == {"runA": "a", "runB": "b"}


def test_archive_selection_texts():
    assert archive("[0, 1, 2, 3].map((n) => [A.selectionLabel(n), A.selectionHint(n)])") == [
        ["Выбрано 0", ""],
        ["Выбрано 1", "Отметьте ещё один запуск, чтобы сравнить"],
        ["Выбрано 2", ""],
        ["Выбрано 3", "Сравнить можно ровно два запуска — снимите лишние отметки"],
    ]


# ---- archive: cells -------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("null", "—"), ("undefined", "—"), ("''", "—"), ("true", "—"), ("'abc'", "—"), ("Infinity", "—"),
        ("0", "0"), ("95.46", "95,5"), ("99.96", "100"), ("100", "100"), ("1234.5", "1 235"),
        ("1234567", "1 234 567"), ("'249.3'", "249"), ("-1500", "-1 500"), ("12.0", "12"),
    ],
)
def test_archive_format_rps(value, expected):
    assert archive(f"A.formatRps({value})") == expected


def test_archive_time_cell_and_duration():
    cell = archive(f"A.timeCell('2026-10-15T09:00:00Z', {NOW})")
    assert cell == {"text": "1 ч назад", "title": "2026-10-15 09:00"}
    assert archive(f"A.timeCell(null, {NOW})") == {"text": "—", "title": ""}
    assert archive(f"A.timeCell('garbage', {NOW})") == {"text": "—", "title": ""}
    run = {"start_time": "2026-10-15T08:00:00Z", "end_time": "2026-10-15T09:05:00Z"}
    assert archive(f"A.durationText({js(run)})") == "1 ч 05 мин"
    assert archive("A.durationText({ start_time: null, end_time: '2026-10-15T09:05:00Z' })") == ""


def test_archive_short_type_label():
    assert archive("['step', 'soak', 'spike', 'stress', '', 'custom'].map(A.shortTypeLabel)") == [
        "Ступенчатый", "Долговременный", "Всплески", "Стресс", "—", "custom",
    ]
