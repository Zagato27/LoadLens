# SPDX-License-Identifier: Apache-2.0
"""Ratchet of Cyrillic user-facing literals: tools/i18n_extract.py against locales/baseline.json.

A file's count may only go down. After translating a file run
``python tools/i18n_extract.py --write-baseline <files>``; literals that must stay Russian (input synonyms, regexes
of stored ru text, language names) go to locales/allowlist.json with a reason.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_extractor():
    spec = importlib.util.spec_from_file_location("i18n_extract_guard", ROOT_DIR / "tools" / "i18n_extract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


extract = _load_extractor()


def _canonical(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_no_file_has_more_cyrillic_literals_than_its_baseline():
    baseline = extract.load_baseline()
    allow = extract.load_allowlist()
    report = []
    for path in extract.scanned_files():
        found = extract.counted(path, allow)
        limit = baseline.get(path, 0)
        if len(found) > limit:
            sample = "\n".join(f"    {item}" for item in found[:15])
            report.append(f"{path}: {len(found)} Cyrillic literals, baseline {limit}\n{sample}")
    assert not report, (
        "user-facing Russian text was added; use t()/LL.t() with a dictionary key, or allowlist a literal that must "
        "stay Russian (locales/allowlist.json)\n" + "\n".join(report)
    )


def test_baseline_and_allowlist_are_canonical():
    for name in ("baseline.json", "allowlist.json"):
        raw = (ROOT_DIR / "locales" / name).read_bytes()
        assert raw == _canonical(json.loads(raw.decode("utf-8"))), f"locales/{name} is not canonical"
    baseline = json.loads((ROOT_DIR / "locales" / "baseline.json").read_text(encoding="utf-8"))
    assert all(isinstance(count, int) and count >= 0 for count in baseline.values())
    assert set(baseline) <= set(extract.scanned_files()), "baseline names files that are not scanned: rewrite it"


def test_allowlist_entries_have_a_literal_and_a_reason():
    data = json.loads((ROOT_DIR / "locales" / "allowlist.json").read_text(encoding="utf-8"))
    scanned = set(extract.scanned_files())
    for path, entries in data.items():
        assert path == "*" or path in scanned or path.startswith("locales/en/"), f"allowlist path {path!r} is not scanned"
        for entry in entries:
            assert set(entry) == {"literal", "reason"} and entry["literal"] and entry["reason"].strip(), (path, entry)


# ---- the extractor --------------------------------------------------------------------------------------------


def _texts(items):
    return [(item.line, item.text) for item in items]


def test_js_literals_skip_comments_and_tell_regexes_from_division():
    source = "\n".join([
        "// Комментарий 'в кавычках'",
        "/* блок",
        "   'тоже комментарий' */",
        "const a = 'Привет', b = \"Мир\", c = 'hello';",
        "const r = text.replace(/ё/g, 'е');",
        "const d = total / count / 2; const e = x.y / 'z';",
        "const t = `Выбрано ${n} из ${items.map((i) => `«${i}»`).join(', ')}`;",
        "const u = `${a ? 'да' : 'no'}`;",
        "return /^Инстанс (\\d+):/.test(s);",
        "url = 'http://x//y'; // хвост",
    ])
    assert _texts(extract.js_literals("x.js", source)) == [
        (4, "Привет"), (4, "Мир"),
        (5, "/ё/g"), (5, "е"),
        (7, "Выбрано ${n} из ${items.map((i) => `«${i}»`).join(', ')}"),
        (8, "да"),
        (9, "/^Инстанс (\\d+):/"),
    ]
    stripped = extract.strip_js_comments(source)
    assert "Комментарий" not in stripped and "хвост" not in stripped and stripped.count("\n") == source.count("\n")


def test_template_literals_skip_comments_and_split_on_markup():
    source = "\n".join([
        "{# Комментарий шаблона #}",
        "<!-- HTML-комментарий -->",
        '<button title="Закрыть меню" class="x">Закрыть</button>',
        "{% set items = [('a', 'Дэшборд')] %}",
        "<p>Действует до {{ minutes }} минут</p>",
        "<style>/* стиль */ .x::after { content: 'Загрузка'; }</style>",
        "<script nonce=\"{{ csp_nonce }}\">// скрипт\nvar x = 1;</script>",
        "{{ t('nav.title') }}",
    ])
    assert _texts(extract.template_literals("x.html", source)) == [
        (3, "Закрыть меню"), (3, "Закрыть"), (4, "Дэшборд"), (5, "Действует до"), (5, "минут"), (6, "Загрузка"),
    ]


def test_python_literals_skip_docstrings_logs_cli_and_main():
    source = '''
"""Модуль."""
import logging
logger = logging.getLogger(__name__)


def f(name):
    """Докстринг."""
    "просто строка"
    logger.warning("Журнал %s", name)
    self.logger.info(f"Журнал {name}")
    print("Печать")
    recovery.log("Аудит")
    warnings.warn("Предупреждение")
    raise ValueError(f"Ошибка «{name}»: {'вложенная' if name else ''}")


def main():
    parser.add_argument("--x", help="Справка CLI")


if __name__ == "__main__":
    print("Запуск")
    sys.exit(main("Аргумент"))
'''
    assert _texts(extract.python_literals("pkg/mod.py", source)) == [
        (14, "Предупреждение"),
        (15, "Ошибка «{name}»: {'вложенная' if name else ''}"),
        (15, "вложенная"),
        (19, "Справка CLI"),
    ]
    extract.CLI_FUNCTIONS["pkg/cli_part.py"] = frozenset({"main"})
    try:
        assert _texts(extract.python_literals("pkg/cli_part.py", source))[-1] == (15, "вложенная")
    finally:
        del extract.CLI_FUNCTIONS["pkg/cli_part.py"]


def test_cli_modules_are_not_scanned():
    files = extract.scanned_files()
    assert {"loadlens_app/auth/cli.py", "AI/query_packs/__main__.py"} & set(files) == set()
    assert "update_page.py" in files and "templates/base.html" in files and "static/js/common.js" in files
    assert not [path for path in files if path.startswith("static/vendor/") or "__pycache__" in path]


def test_write_baseline_lowers_listed_files_and_refuses_growth(tmp_path, monkeypatch, capsys):
    target = tmp_path / "baseline.json"
    target.write_text(json.dumps({"a.js": 5, "b.js": 3, "c.js": 2}), encoding="utf-8")
    monkeypatch.setattr(extract, "BASELINE_PATH", target)
    fake = {"a.js": 1, "b.js": 4, "c.js": 2, "d.js": 7}
    monkeypatch.setattr(extract, "counts", lambda paths=None: {p: fake[p] for p in (fake if paths is None else paths)})

    assert extract.write_baseline(["a.js", "b.js"]) == 1
    assert json.loads(target.read_text(encoding="utf-8")) == {"a.js": 1, "b.js": 3, "c.js": 2}
    assert "b.js would grow from 3 to 4" in capsys.readouterr().out
    assert extract.write_baseline(["b.js"], force=True) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["b.js"] == 4

    fake["a.js"] = 0
    assert extract.write_baseline(None) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == {"a.js": 0, "b.js": 4, "c.js": 2, "d.js": 7}


def test_cli_lists_and_summarizes(capsys, monkeypatch):
    # base.html is translated: a stand-in literal plays the residue the listing prints.
    real = extract.literals_of
    monkeypatch.setattr(extract, "literals_of", lambda path: [extract.Literal(path, 1, "Пример")] if path == "templates/base.html" else real(path))
    assert extract.main(["templates/base.html"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed and all(line.startswith("templates/base.html:") for line in listed)
    assert extract.main(["--summary", "templates/base.html"]) == 0
    assert "total" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        extract.main(["no/such/file.js"])
