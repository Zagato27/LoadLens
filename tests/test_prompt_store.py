# SPDX-License-Identifier: Apache-2.0
"""Prompt files of both languages and AI/prompt_store.py (plans/i18n-plan.md, sections 3.3, 3.5 and 3.7)."""

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: E402,F401  (import order: loadlens_app before update_page / the pipeline)
from AI import prompt_store  # noqa: E402
from AI.i18n import with_locale  # noqa: E402
from AI.prompt_store import (  # noqa: E402
    DOMAIN_PROMPT_FILES,
    PROMPTS_ROOT,
    PromptMissing,
    default_prompts,
    detect_language,
    effective_override,
    load_prompt,
    prompt_files,
    text,
)
from AI.verdicts import CODES  # noqa: E402

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
PLACEHOLDER = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}|\{[A-Za-z_][A-Za-z0-9_]*\}")
JSON_KEY = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:')
RU_DIR = PROMPTS_ROOT / "ru"
EN_DIR = PROMPTS_ROOT / "en"
TEXT_FILES = [name for name in prompt_files("ru") if name.endswith(".txt")]
# Files the code loads by name (besides the editable domains).
LOADED_FILES = {
    "_context_guide.txt", "_response_format.txt", "verify_prompt.txt", "judge_safe_prompt.txt", "_system.txt",
    "_system_json.txt", "_no_sla.txt", "forecast_limit_prompt.txt", "context_import_prompt.txt",
    "judge_rubrics/common.txt", "judge_rubrics/final.txt", "_strings.json",
    *(f"profiles/{name}.txt" for name in prompt_store.TEST_PROFILES),
}
# Lines of the ru files that may differ from the files before the move: the verdict contract (codes, labels,
# threshold null). Everything else stays byte-identical.
CONTRACT_MARKERS = (
    "verdict", "Успешно", "Есть риски", "Провал", "Недостаточно данных", "не задан", "threshold", "Статус", "подпись",
    "недопустимо",
)


def _load_stub():
    spec = importlib.util.spec_from_file_location("llm_stub_for_prompts", ROOT_DIR / "tools" / "stubs" / "llm_stub.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chat(system: str, user: str) -> dict:
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


@pytest.fixture
def prompts_root(tmp_path, monkeypatch):
    """An empty prompts tree with ru and en folders; the store reads it instead of AI/prompts."""
    (tmp_path / "ru").mkdir()
    (tmp_path / "en").mkdir()
    monkeypatch.setattr(prompt_store, "PROMPTS_ROOT", tmp_path)
    prompt_store.reload_prompts()
    yield tmp_path
    prompt_store.reload_prompts()


# ---- files of both languages ------------------------------------------------------------------------------


def test_every_language_has_the_same_files():
    assert prompt_files("ru") == prompt_files("en")
    names = set(prompt_files("ru"))
    assert LOADED_FILES <= names
    assert set(DOMAIN_PROMPT_FILES.values()) <= names
    assert {f"judge_rubrics/{name}.txt" for name in ("jvm", "database", "kafka", "microservices", "hard_resources",
                                                      "lt_framework", "lt_generators", "application_logs")} <= names
    # The old flat layout is gone: every prompt lives in a language folder.
    assert not [path for path in PROMPTS_ROOT.iterdir() if path.is_file()]


@pytest.mark.parametrize("name", TEXT_FILES)
def test_language_pairs_keep_placeholders_json_keys_and_format(name):
    ru_raw = (RU_DIR / name).read_bytes()
    en_raw = (EN_DIR / name).read_bytes()
    for raw in (ru_raw, en_raw):
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n"), name
    ru, en = ru_raw.decode("utf-8"), en_raw.decode("utf-8")
    assert Counter(PLACEHOLDER.findall(ru)) == Counter(PLACEHOLDER.findall(en))
    assert Counter(JSON_KEY.findall(ru)) == Counter(JSON_KEY.findall(en))
    assert ru.count("\\") == en.count("\\")
    assert not CYRILLIC.search(en), CYRILLIC.findall(en)[:10]
    assert CYRILLIC.search(ru) or name == "judge_rubrics/application_logs.txt"  # that rubric was English already


def _evaluation_form(body: str) -> dict:
    block = body.split("Evaluation Form", 1)[1]
    return json.loads(block[block.index("{"): block.rindex("}") + 1])


def test_judge_rubrics_keep_the_criteria_and_the_evaluation_form():
    from AI.scoring import JudgeRubricScores, JudgeScoreItem

    forms = {lang: _evaluation_form(load_prompt("judge_rubrics/common.txt", lang)) for lang in ("ru", "en")}
    assert forms["ru"] == forms["en"]
    item = JudgeScoreItem.parse_obj(forms["en"]["scores"][0])
    criteria = set(item.rubric.dict())
    assert criteria == set(JudgeRubricScores.__fields__)
    for lang in ("ru", "en"):
        common = load_prompt("judge_rubrics/common.txt", lang)
        assert all(name in common for name in criteria)
    for name in [n for n in TEXT_FILES if n.startswith("judge_rubrics/")]:
        ru_heads = [line for line in load_prompt(name, "ru").splitlines()
                    if line and not CYRILLIC.search(line) and not line.startswith(("-", " ", "{", "}", '"'))]
        en_lines = load_prompt(name, "en").splitlines()
        assert all(head in en_lines for head in ru_heads), name


def test_verdict_contract_in_both_languages():
    for lang, labels in (("ru", ("Успешно", "Есть риски", "Провал", "Недостаточно данных")),
                         ("en", ("Passed", "Risks found", "Failed", "Insufficient data"))):
        response_format = load_prompt("_response_format.txt", lang)
        assert '"verdict": "pass" | "risk" | "fail" | "no_data"' in response_format
        assert all(label in response_format for label in labels)  # the code -> label table for the rationale
        for name in ("_response_format.txt", "verify_prompt.txt", "critic_prompt.txt", "application_logs_prompt.txt"):
            body = load_prompt(name, lang)
            assert '"no_data"' in body, (lang, name)
            assert 'verdict="Недостаточно данных"' not in body and '"verdict": "Успешно"' not in body, (lang, name)
        assert "null" in response_format and '"не задан"' not in response_format and "'не задан'" not in response_format
    ru_status = load_prompt("_response_format.txt", "ru")
    assert "«Статус «<подпись>»" in ru_status and 'Status "<label>"' in load_prompt("_response_format.txt", "en")
    # The ru text fields stay Russian; verdict is a code now.
    assert "(verdict_rationale," in load_prompt("critic_prompt.txt", "ru")
    assert "ступенчат" not in load_prompt("_system.txt", "ru")  # the default system prompt fits every test type
    assert "pass | risk | fail | no_data" in load_prompt("_system_json.txt", "en")


def _head_text(path: str):
    try:
        done = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT_DIR, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8").replace("\r\n", "\n")


def test_ru_prompts_equal_the_files_before_the_move_except_the_verdict_contract():
    import difflib

    compared = 0
    for name in TEXT_FILES:
        old = _head_text(f"AI/prompts/{name}")
        if old is None:
            continue  # a file extracted from Python, or the move is committed already
        compared += 1
        old_lines = old.splitlines()
        new_lines = (RU_DIR / name).read_text(encoding="utf-8").splitlines()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            hunk = old_lines[i1:i2] + new_lines[j1:j2]
            # Every changed block is about the verdict contract.
            assert any(marker in line for line in hunk for marker in CONTRACT_MARKERS), (name, hunk)
    if not compared:
        pytest.skip("no pre-move copies of the prompt files in HEAD (committed already, or no git)")


# ---- _strings.json -----------------------------------------------------------------------------------------


def _strings_file(lang: str) -> tuple[str, dict]:
    raw = (PROMPTS_ROOT / lang / "_strings.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    body = raw.decode("utf-8")
    return body, json.loads(body)


def test_strings_are_canonical_with_the_same_keys_and_placeholders():
    ru_body, ru = _strings_file("ru")
    en_body, en = _strings_file("en")
    for body, data in ((ru_body, ru), (en_body, en)):
        assert body == json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        assert all(isinstance(value, str) and value for value in data.values())
    assert set(ru) == set(en)
    for key in ru:
        assert Counter(PLACEHOLDER.findall(ru[key])) == Counter(PLACEHOLDER.findall(en[key])), key
        assert not CYRILLIC.search(en[key]), key
    # Families the code builds keys of.
    from AI.prompt_assembly import SLA_FIELDS

    for code in CODES:
        assert f"rationale.head_{code}" in ru
    for field in SLA_FIELDS:
        assert f"guidance.sla_field.{field}" in ru and f"guidance.sla_unit.{field}" in ru
    for domain in ("jvm", "database", "kafka", "microservices", "hard_resources", "application_logs", "lt_generators"):
        assert f"result_header.{domain}" in ru


def _literal_keys() -> set[str]:
    pattern = re.compile(r"""(?:prompt_text|_prompt_text|\btext)\(\s*"([a-z_]+(?:\.[a-z_]+)+)\"""")
    keys = set()
    for rel in ("AI/pipeline.py", "AI/scoring.py", "AI/providers.py", "AI/prompt_assembly.py",
                "loadlens_app/forecast_explanation.py", "loadlens_app/context_import.py"):
        keys |= set(pattern.findall((ROOT_DIR / rel).read_text(encoding="utf-8")))
    return keys


def test_every_key_the_code_uses_exists():
    keys = _literal_keys()
    assert len(keys) > 40
    assert keys <= set(prompt_store.strings("ru")), keys - set(prompt_store.strings("ru"))


# Russian text of the strings moved out of Python: the ru output stays as before.
RU_MOVED = {
    "system_context.none": "Контекст системы не задан.",
    "system_context.prefix": (
        "СПРАВОЧНЫЙ КОНТЕКСТ ТЕСТИРУЕМОЙ СИСТЕМЫ:\n{brief}\n\nИспользуй этот контекст только как справочную информацию. "
        "Не подменяй им факты из метрик и не делай выводов, которые не подтверждаются данными."
    ),
    "guidance.header": "ОРИЕНТИРЫ АНАЛИЗА (сформированы автоматически из настроек сервиса):",
    "peak.designated": (
        "ЕДИНСТВЕННЫЙ источник peak_performance.max_rps. "
        "НЕ использовать RPS из домена microservices для определения максимальной производительности системы."
    ),
    "judge.system": (
        "Вы опытный инженер по нагрузочному тестированию и выступаете независимым судьёй. "
        "Используйте предоставленный контекст метрик, чтобы беспристрастно оценить кандидатов. "
        "Верните только JSON согласно запросу."
    ),
    "judge.no_data": "нет данных",
    "rationale.head_risk": "Статус «{label}»: есть подтверждённые отклонения, но без провала целевого результата.",
    "rationale.derived_findings": "Вердикт «{label}» сформирован по ключевым наблюдениям отчёта.\n{bullets}",
    "analysis_error.invalid_json": "Ответ модели не удалось разобрать как JSON.",
    "forecast.system": "Вы инженер по производительности. Отвечайте строго в JSON по заданной схеме, все тексты — на русском языке.",
    "forecast.kind_stalls": "провалы времени отклика",
    "forecast.kind_usl": "потолок модели USL",
    "context_import.system": (
        "Вы переносите описание программной системы из документа в структуру JSON. "
        "Используйте только сведения из документа и возвращайте только JSON."
    ),
    "context_import.document_open": "<<<ДОКУМЕНТ",
}


def test_ru_text_of_the_strings_moved_from_python_is_unchanged():
    for key, value in RU_MOVED.items():
        assert prompt_store.strings("ru")[key] == value, key
    assert text("verdict.sla_step_adjusted", "ru", step="ступень 5", label="Провал") == (
        "Статус скорректирован по ступени SLA: на ступени «ступень 5» пороги SLA выполнены, "
        "серьёзные проблемы домена начались после неё. Исходная оценка модели — «Провал»."
    )


def test_text_fills_placeholders_and_follows_the_locale():
    assert text("guidance.min_stable", "en", minutes="5") == "- Minimum duration of a stable step: 5 min"
    assert text("guidance.min_stable", "ru", minutes="5") == "- Минимальная длительность стабильной ступени: 5 мин"
    with with_locale("en"):
        assert text("judge.no_data") == "no data"
        assert load_prompt("_system.txt").startswith("You are a load testing engineer")
    assert text("judge.no_data") == "нет данных"  # tests run in ru
    assert text("system_context.system", "en", value="{value} {n}") == "- System: {value} {n}"  # values are not re-read
    with pytest.raises(PromptMissing):
        text("no.such_key", "en")


# ---- loader: fallback and cache ----------------------------------------------------------------------------


def test_missing_file_of_a_language_falls_back_to_ru_with_one_warning(prompts_root, caplog):
    (prompts_root / "ru" / "only_ru.txt").write_text("русский", encoding="utf-8")
    (prompts_root / "ru" / "both.txt").write_text("русский", encoding="utf-8")
    (prompts_root / "en" / "both.txt").write_text("english", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="AI.prompt_store"):
        assert load_prompt("only_ru.txt", "en") == "русский"
        assert load_prompt("only_ru.txt", "en") == "русский"
    warnings = [record for record in caplog.records if "only_ru.txt" in record.getMessage()]
    assert len(warnings) == 1
    # Cached per language: the same name gives each language its own text.
    assert load_prompt("both.txt", "ru") == "русский" and load_prompt("both.txt", "en") == "english"
    assert load_prompt("both.txt", "EN-gb") == "english"
    with pytest.raises(PromptMissing):
        load_prompt("nowhere.txt", "en")
    with pytest.raises(PromptMissing):
        load_prompt("../outside.txt", "ru")


def test_strings_fall_back_to_ru_and_files_are_reread_after_a_change(prompts_root, caplog):
    (prompts_root / "ru" / "_strings.json").write_text(json.dumps({"a.b": "ру {n}"}), encoding="utf-8")
    (prompts_root / "en" / "_strings.json").write_text(json.dumps({}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="AI.prompt_store"):
        assert text("a.b", "en", n=2) == "ру 2"
    assert any("a.b" in record.getMessage() for record in caplog.records)
    path = prompts_root / "ru" / "x.txt"
    path.write_text("one", encoding="utf-8")
    assert load_prompt("x.txt", "ru") == "one"
    path.write_text("two", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert load_prompt("x.txt", "ru") == "two"


def test_default_prompts_by_language():
    ru = default_prompts("ru")
    en = default_prompts("en")
    assert set(ru) == set(en) == set(DOMAIN_PROMPT_FILES)
    assert CYRILLIC.search(ru["jvm"]) and not CYRILLIC.search(en["jvm"])
    assert ru["jvm"] == (RU_DIR / "jvm_prompt.txt").read_text(encoding="utf-8")


# ---- overrides and language detection -------------------------------------------------------------------------


def test_effective_override_ignores_empty_texts_and_copies_of_any_default():
    ru = load_prompt("jvm_prompt.txt", "ru")
    en = load_prompt("jvm_prompt.txt", "en")
    for value in (None, "", "  \n", ru, en, ru.replace("\n", "\r\n"), ru.strip() + "\n\n", "  " + en):
        assert effective_override(value, "jvm") is None
    assert effective_override(en, "jvm_prompt.txt") is None
    custom = ru + "\nДополнительное правило."
    assert effective_override(custom, "jvm") == custom
    assert effective_override(ru, "database") == ru  # another domain's default is a real override here
    assert effective_override("Analyse JVM only.", "jvm") == "Analyse JVM only."


def test_detect_language_of_a_text():
    assert detect_language(load_prompt("overall_prompt.txt", "ru")) == "ru"
    assert detect_language(load_prompt("overall_prompt.txt", "en")) == "en"
    assert detect_language("Проанализируй heap_used и p95 на ступени 3") == "ru"
    assert detect_language("p95 300 CPU_usage 12.5") is None
    assert detect_language("") is None


# ---- candidates in the report language ------------------------------------------------------------------------

RU_ANSWER = {
    "verdict": "risk",
    "verdict_rationale": "Статус «Есть риски»: задержка растёт быстрее нагрузки на ступени 3.",
    "findings": [{
        "id": "f1", "summary": "Рост p95 latency на ступени 3 до 450 ms", "severity": "high", "component": "orders-api",
        "evidence_summary": "Серия http_server_requests_seconds растёт", "evidence_items": [
            {"metric": "http_server_requests_seconds", "observed_value": "450 ms", "threshold": None, "note": "ступень 3"},
        ],
    }],
    "recommended_actions": [{"summary": "Проверить пул соединений", "details": "Увеличить пул и повторить тест.", "for_finding_ids": ["f1"]}],
}
EN_ANSWER = {
    "verdict": "risk",
    "verdict_rationale": 'Status "Risks found": the latency grows faster than the load on step 3.',
    "findings": [{
        "id": "f1", "summary": "p95 latency grows to 450 ms on step 3", "severity": "high", "component": "сервис-заказов",
        "evidence_summary": "The http_server_requests_seconds series grows", "evidence_items": [
            {"metric": "задержка_ответа", "observed_value": "450 ms", "threshold": None, "note": "step 3"},
        ],
    }],
    "recommended_actions": [{"summary": "Check the connection pool", "details": "Raise the pool size and repeat the test.", "for_finding_ids": ["f1"]}],
}


def test_language_match_measures_the_free_text_only():
    from AI.scoring import LLMAnalysis, language_match

    assert language_match(RU_ANSWER, "ru") >= 0.5 > language_match(RU_ANSWER, "en")
    # Component and metric names are data: a Cyrillic component does not make an English answer Russian.
    assert language_match(EN_ANSWER, "en") == pytest.approx(1.0)
    assert language_match(LLMAnalysis.parse_obj(EN_ANSWER), "ru") == pytest.approx(0.0)
    assert language_match({"verdict": "pass"}, "en") == 1.0  # nothing to judge


def test_best_candidate_is_the_one_in_the_report_language(monkeypatch):
    from AI import scoring
    from AI.scoring import LLMAnalysis

    candidates = [
        (json.dumps(RU_ANSWER, ensure_ascii=False), LLMAnalysis.parse_obj(RU_ANSWER)),
        (json.dumps(EN_ANSWER), LLMAnalysis.parse_obj(EN_ANSWER)),
    ]
    # The judge prefers the Russian answer; in an en run the English one still wins.
    monkeypatch.setattr(scoring, "judge_candidates_with_llm", lambda *a, **k: ({0: {"overall": 1.0}, 1: {"overall": 0.0}}, {}))
    _text, parsed, info = scoring._select_best_candidate(candidates, "{}", domain_key="jvm", lang="en")
    assert info["selected_index"] == 1 and info["language_match"] == pytest.approx(1.0)
    _text, parsed, info = scoring._select_best_candidate(candidates, "{}", domain_key="jvm", lang="ru")
    assert info["selected_index"] == 0 and info["language_match"] >= 0.5
    with with_locale("en"):
        _text, _parsed, info = scoring._select_best_candidate(candidates[:1], "{}", domain_key="jvm")
    assert info["language_match"] < 0.5  # written into score_info for one candidate too


def test_gigachat_schema_names_the_verdict_codes():
    from AI.scoring import LLMAnalysis

    schema = LLMAnalysis.schema()
    assert schema["properties"]["verdict"]["description"] == "pass | risk | fail | no_data"


# ---- LLM stub ----------------------------------------------------------------------------------------------


def test_llm_stub_tells_english_requests_apart_and_answers_in_their_language():
    from AI.providers import default_system_prompt
    from AI.scoring import judge_system_prompt, parse_llm_analysis_strict
    from loadlens_app import context_import

    stub = _load_stub()
    assert stub.classify(_chat(judge_system_prompt("en"), "[]")) == ("judge", "")
    assert stub.classify(_chat(context_import.system_prompt("en"), "<<<DOCUMENT\ntext\nDOCUMENT>>>")) == ("context_import", "")
    body_en = _chat(default_system_prompt("en"), 'Analyse the domain\n\n{"domain": "kafka", "sections": []}')
    body_ru = _chat(default_system_prompt("ru"), 'Проанализируй домен\n\n{"domain": "kafka", "sections": []}')
    assert stub.classify(body_en) == ("analysis", "kafka")
    content, kind, domain = stub.answer_text(body_en)
    assert (kind, domain) == ("analysis", "kafka") and not CYRILLIC.search(content)
    assert parse_llm_analysis_strict(content).verdict == "pass"
    content_ru, _kind, _domain = stub.answer_text(body_ru)
    assert CYRILLIC.search(content_ru) and json.loads(content_ru)["verdict"] == "pass"
    forced, _kind, _domain = stub.answer_text(body_ru, "en")
    assert not CYRILLIC.search(forced)
    assert json.loads(stub.answer_text(_chat(default_system_prompt("en"), '{"domain": "hard_resources"}'))[0])["verdict"] == "risk"
    assert stub.parse_args(["--lang", "en"]).lang == "en" and stub.parse_args([]).lang == "auto"
