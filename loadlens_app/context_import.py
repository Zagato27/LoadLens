"""System context from a document: text extraction, one LLM call, and merging into the current context.

Nothing here stores anything: the settings page shows the result in the form and the user saves it.
Supported inputs are PDF with a text layer, Word (.docx), Markdown, plain text and pasted text.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from loadlens_app.core import _clean_string_list, _clean_text, _normalize_object_list

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_TEXT_CHARS = 120_000
MIN_TEXT_CHARS = 40
# Share of letters among non-space characters below which the extracted text is probably garbled.
MIN_LETTER_SHARE = 0.5
MODES = ("fill", "replace")
SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}
PROMPT_FILE = Path(__file__).resolve().parent.parent / "AI" / "prompts" / "context_import_prompt.txt"
SYSTEM_PROMPT = (
    "Вы переносите описание программной системы из документа в структуру JSON. "
    "Используйте только сведения из документа и возвращайте только JSON."
)

# The context structure: plain strings, lists of objects (string and list fields) and lists of strings.
STRING_FIELDS: dict[str, tuple[str, ...]] = {
    "system": ("name", "domain", "description", "test_goal"),
    "architecture": ("style",),
}
OBJECT_LISTS: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {
    ("architecture", "components"): (("id", "name", "role", "criticality"), ("technologies",)),
    ("architecture", "dependencies"): (("from", "to", "kind", "purpose"), ()),
    ("architecture", "data_stores"): (("id", "type", "purpose"), ("used_by",)),
    ("load_model", "entrypoints"): (("id", "name", "kind", "business_priority"), ()),
    ("load_model", "critical_user_flows"): (("id", "name"), ("steps", "success_signals")),
}
STRING_LISTS: tuple[tuple[str, str], ...] = (
    ("load_model", "expected_hotspots"),
    ("operational_context", "known_constraints"),
    ("operational_context", "known_risks"),
    ("operational_context", "normal_degradation_rules"),
    ("operational_context", "analysis_focus"),
)
GROUPS = ("system", "architecture", "load_model", "operational_context")

LEVELS = {
    "high": "high", "высокая": "high", "высокий": "high", "critical": "high", "критичная": "high", "критическая": "high",
    "medium": "medium", "средняя": "medium", "средний": "medium", "normal": "medium",
    "low": "low", "низкая": "low", "низкий": "low",
}
DEPENDENCY_KINDS = {"sync": "sync", "синхронная": "sync", "synchronous": "sync", "async": "async", "асинхронная": "async", "asynchronous": "async"}
ARCH_STYLES = {
    "microservices": "microservices", "микросервисы": "microservices", "микросервисная": "microservices",
    "monolith": "monolith", "монолит": "monolith",
    "modular-monolith": "modular-monolith", "modular monolith": "modular-monolith", "модульный монолит": "modular-monolith",
    "event-driven": "event-driven", "событийная": "event-driven",
    "serverless": "serverless",
}

FIELD_TITLES = {
    "system.name": "Название системы",
    "system.domain": "Предметная область",
    "system.description": "Описание",
    "system.test_goal": "Цель тестирования",
    "architecture.style": "Архитектурный стиль",
    "architecture.components": "Компоненты",
    "architecture.dependencies": "Зависимости",
    "architecture.data_stores": "Хранилища",
    "load_model.entrypoints": "Точки входа",
    "load_model.critical_user_flows": "Критичные сценарии",
    "load_model.expected_hotspots": "Ожидаемые узкие места",
    "operational_context.known_constraints": "Ограничения",
    "operational_context.known_risks": "Известные риски",
    "operational_context.normal_degradation_rules": "Нормальная деградация",
    "operational_context.analysis_focus": "Фокус анализа",
}


class ContextImportError(Exception):
    """The import cannot go on; the message is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class ExtractedText:
    text: str
    format: str
    chars: int
    pages: Optional[int] = None
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    def source(self, name: str, size: int) -> dict[str, Any]:
        out = asdict(self)
        out.pop("text")
        out.pop("warnings")
        return {"name": name, "bytes": size, **out}


# ---- text extraction ------------------------------------------------------------------------------


def _pdf_text(data: bytes, warnings: list[str]) -> tuple[str, int]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ContextImportError("PDF защищён паролем: снимите защиту или вставьте текст вручную")
        pages = len(reader.pages)
        if pages > MAX_PDF_PAGES:
            warnings.append(f"В PDF {pages} страниц, прочитаны первые {MAX_PDF_PAGES}")
        texts = [(page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES]]
    except ContextImportError:
        raise
    except Exception as exc:  # damaged files raise many kinds of errors
        logger.info("context import: unreadable PDF", exc_info=True)
        raise ContextImportError(f"Не удалось прочитать PDF: {str(exc)[:200]}") from exc
    return "\n\n".join(texts), pages


def _docx_text(data: bytes) -> str:
    import docx
    from docx.table import Table

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # not a .docx inside, or a damaged one
        logger.info("context import: unreadable Word document", exc_info=True)
        raise ContextImportError(f"Не удалось прочитать документ Word: {str(exc)[:200]}") from exc
    parts: list[str] = []
    # Paragraphs and tables in document order; tables often hold the list of components.
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                unique = [text for index, text in enumerate(cells) if text and text not in cells[:index]]
                if unique:
                    parts.append(" | ".join(unique))
        else:
            text = block.text.strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _plain_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _finish(text: str, fmt: str, warnings: list[str], pages: Optional[int] = None) -> ExtractedText:
    text = re.sub(r"[ \t\u00a0]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < MIN_TEXT_CHARS:
        if fmt == "pdf":
            raise ContextImportError("В PDF нет текстового слоя (похоже на скан). Вставьте текст вручную")
        raise ContextImportError("В документе почти нет текста: нужно хотя бы несколько предложений")
    visible = [ch for ch in text if not ch.isspace()]
    letters = sum(1 for ch in visible if ch.isalpha())
    if visible and letters / len(visible) < MIN_LETTER_SHARE:
        warnings.append("Текст извлечён с искажениями: проверьте результат или вставьте текст вручную")
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        warnings.append(f"Документ длиннее {MAX_TEXT_CHARS:,} символов, разобрано только начало".replace(",", " "))
        text = text[:MAX_TEXT_CHARS]
    return ExtractedText(text=text, format=fmt, chars=len(text), pages=pages, truncated=truncated, warnings=warnings)


def extract_text(filename: str, data: bytes) -> ExtractedText:
    """Text of an uploaded document; the extension decides the reader, the content must match it."""
    extension = Path(filename or "").suffix.lower()
    fmt = SUPPORTED_EXTENSIONS.get(extension)
    if fmt is None:
        allowed = ", ".join(sorted({ext for ext in SUPPORTED_EXTENSIONS if ext != ".markdown"}))
        raise ContextImportError(f"Формат «{extension or 'без расширения'}» не поддерживается. Допустимо: {allowed}", 415, "unsupported")
    if len(data) > MAX_FILE_BYTES:
        raise ContextImportError(f"Файл больше {MAX_FILE_BYTES // (1024 * 1024)} МБ", 413, "too_large")
    warnings: list[str] = []
    if fmt == "pdf":
        if not data.startswith(b"%PDF"):
            raise ContextImportError("Файл не похож на PDF")
        text, pages = _pdf_text(data, warnings)
        return _finish(text, fmt, warnings, pages)
    if fmt == "docx":
        if not data.startswith(b"PK"):
            raise ContextImportError("Файл не похож на документ Word (.docx). Старый формат .doc не поддерживается")
        return _finish(_docx_text(data), fmt, warnings)
    if b"\x00" in data[:4096]:
        raise ContextImportError("Файл не похож на текстовый")
    return _finish(_plain_text(data), fmt, warnings)


def text_from_paste(text: object) -> ExtractedText:
    if not isinstance(text, str):
        raise ContextImportError("Вставьте текст описания системы")
    return _finish(text, "text", [])


# ---- the model's answer ------------------------------------------------------------------------------


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class _Component(_Item):
    id: str = ""
    name: str = ""
    role: str = ""
    criticality: str = ""
    technologies: list[str] = Field(default_factory=list)


class _Dependency(_Item):
    from_: str = Field(default="", alias="from")
    to: str = ""
    kind: str = ""
    purpose: str = ""


class _DataStore(_Item):
    id: str = ""
    type: str = ""
    purpose: str = ""
    used_by: list[str] = Field(default_factory=list)


class _Entrypoint(_Item):
    id: str = ""
    name: str = ""
    kind: str = ""
    business_priority: str = ""


class _Flow(_Item):
    id: str = ""
    name: str = ""
    steps: list[str] = Field(default_factory=list)
    success_signals: list[str] = Field(default_factory=list)


class _System(_Item):
    name: str = ""
    domain: str = ""
    description: str = ""
    test_goal: str = ""


class _Architecture(_Item):
    style: str = ""
    components: list[_Component] = Field(default_factory=list)
    dependencies: list[_Dependency] = Field(default_factory=list)
    data_stores: list[_DataStore] = Field(default_factory=list)


class _LoadModel(_Item):
    entrypoints: list[_Entrypoint] = Field(default_factory=list)
    critical_user_flows: list[_Flow] = Field(default_factory=list)
    expected_hotspots: list[str] = Field(default_factory=list)


class _Operations(_Item):
    known_constraints: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    normal_degradation_rules: list[str] = Field(default_factory=list)
    analysis_focus: list[str] = Field(default_factory=list)


class SystemContextDraft(_Item):
    """Schema of the structured answer (GigaChat function calling)."""

    system: _System = Field(default_factory=_System)
    architecture: _Architecture = Field(default_factory=_Architecture)
    load_model: _LoadModel = Field(default_factory=_LoadModel)
    operational_context: _Operations = Field(default_factory=_Operations)


def _mapped(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value.strip().lower(), "")


def shape_context(value: object) -> dict[str, Any]:
    """Full context structure from any dict: unknown keys dropped, enums mapped, every field present."""
    source = value if isinstance(value, dict) else {}
    out: dict[str, Any] = {group: {} for group in GROUPS}
    for group, names in STRING_FIELDS.items():
        node = source.get(group) if isinstance(source.get(group), dict) else {}
        for name in names:
            out[group][name] = _clean_text(node.get(name))
    for (group, name), (strings, lists) in OBJECT_LISTS.items():
        node = source.get(group) if isinstance(source.get(group), dict) else {}
        rows = _normalize_object_list(node.get(name), string_fields=strings, list_fields=lists)
        out[group][name] = [
            {key: row[key] for key in (*strings, *lists)}
            for row in rows
            if any(row[key] for key in (*strings, *lists))
        ]
    for group, name in STRING_LISTS:
        node = source.get(group) if isinstance(source.get(group), dict) else {}
        out[group][name] = _clean_string_list(node.get(name))
    out["architecture"]["style"] = _mapped(out["architecture"]["style"], ARCH_STYLES) or (
        out["architecture"]["style"] if out["architecture"]["style"] in ARCH_STYLES.values() else ""
    )
    for row in out["architecture"]["components"]:
        row["criticality"] = _mapped(row["criticality"], LEVELS)
    for row in out["architecture"]["dependencies"]:
        row["kind"] = _mapped(row["kind"], DEPENDENCY_KINDS)
    for row in out["load_model"]["entrypoints"]:
        row["business_priority"] = _mapped(row["business_priority"], LEVELS)
    return out


def parse_answer(raw: object) -> dict[str, Any]:
    """The context in the model's answer: JSON alone, in prose or in a code block, or under "system_context"."""
    if isinstance(raw, BaseModel):
        obj: Any = raw.model_dump(by_alias=True)
    else:
        from AI.scoring import _extract_json_like, _json_loads_lenient

        text = str(raw or "")
        obj = _json_loads_lenient(text.strip()) or _extract_json_like(text)
    if isinstance(obj, dict) and isinstance(obj.get("system_context"), dict):
        obj = obj["system_context"]
    if not isinstance(obj, dict) or not any(isinstance(obj.get(group), dict) for group in GROUPS):
        raise ContextImportError("Модель вернула ответ не в формате контекста системы. Повторите или сократите документ", 502, "bad_answer")
    return shape_context(obj)


def _prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def llm_for_scope(llm_section: object) -> tuple[str, dict[str, Any]]:
    """Provider name and its settings from the ``llm`` section the area uses."""
    llm = llm_section if isinstance(llm_section, dict) else {}
    provider = str(llm.get("provider") or "").strip().lower()
    settings = llm.get(provider) if provider else None
    if not provider or not isinstance(settings, dict):
        raise ContextImportError("Не настроен провайдер LLM: Настройки → LLM", 422, "missing_llm")
    return provider, settings


def ask_model(text: str, llm_section: object, use_structured: bool = False) -> dict[str, Any]:
    """One model call for the document; returns the shaped context."""
    from AI.providers import ask_llm_structured, ask_llm_with_text_data, usage_domain

    provider, settings = llm_for_scope(llm_section)
    document = f"<<<ДОКУМЕНТ\n{text}\nДОКУМЕНТ>>>"
    try:
        with usage_domain("context_import"):
            if use_structured and provider == "gigachat":
                raw: object = ask_llm_structured(_prompt(), document, SystemContextDraft, system_prompt=SYSTEM_PROMPT)
            else:
                raw = ask_llm_with_text_data(
                    _prompt(),
                    document,
                    llm_config={"provider": provider, provider: settings, "force_json": True},
                    system_prompt=SYSTEM_PROMPT,
                )
    except ContextImportError:
        raise
    except Exception as exc:
        logger.warning("context import: the model call failed: %s", exc)
        raise ContextImportError(f"Модель {provider} не ответила: {str(exc)[:200]}", 502, "llm_failed") from exc
    return parse_answer(raw)


# ---- merging ---------------------------------------------------------------------------------------


def _key(text: object) -> str:
    return _clean_text(text).casefold()


def _same_item(name: str, left: dict[str, Any], right: dict[str, Any]) -> bool:
    if name == "dependencies":
        return bool(_key(left.get("from")) and _key(left.get("to"))) and (
            _key(left.get("from")), _key(left.get("to"))) == (_key(right.get("from")), _key(right.get("to")))
    for field_name in ("id", "name"):
        if _key(left.get(field_name)) and _key(left.get(field_name)) == _key(right.get(field_name)):
            return True
    return False


def merge_context(current: object, draft: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Context after the import and the list of changed fields.

    ``fill``: only empty strings are filled; list items and lines that are not there yet are added.
    ``replace``: the draft replaces every field. ``enabled`` and ``schema_version`` stay as they were.
    """
    if mode not in MODES:
        raise ContextImportError(f"mode: допустимо {', '.join(MODES)}")
    base = current if isinstance(current, dict) else {}
    before = shape_context(base)
    draft = shape_context(draft)
    result = copy.deepcopy(before) if mode == "fill" else copy.deepcopy(draft)
    changes: list[dict[str, Any]] = []

    def note(path: str, action: str, count: int = 0) -> None:
        changes.append({"path": path, "title": FIELD_TITLES.get(path, path), "action": action, "count": count})

    for group, names in STRING_FIELDS.items():
        for name in names:
            path, old, new = f"{group}.{name}", before[group][name], draft[group][name]
            if mode == "fill":
                if not old and new:
                    result[group][name] = new
                    note(path, "filled")
            elif old != new:
                note(path, "replaced" if new else "cleared")
    for (group, name) in OBJECT_LISTS:
        path, old, new = f"{group}.{name}", before[group][name], draft[group][name]
        if mode == "fill":
            added = [item for item in new if not any(_same_item(name, item, existing) for existing in old)]
            if added:
                result[group][name] = old + added
                note(path, "added", len(added))
        elif old != new:
            note(path, "replaced" if new else "cleared", len(new))
    for group, name in STRING_LISTS:
        path, old, new = f"{group}.{name}", before[group][name], draft[group][name]
        if mode == "fill":
            known = {_key(line) for line in old}
            added = [line for line in new if _key(line) not in known and not known.add(_key(line))]
            if added:
                result[group][name] = old + added
                note(path, "added", len(added))
        elif old != new:
            note(path, "replaced" if new else "cleared", len(new))

    for key in ("enabled", "schema_version"):
        if key in base:
            result[key] = base[key]
    result.setdefault("enabled", True)
    result.setdefault("schema_version", 1)
    return result, changes


__all__ = [
    "ContextImportError",
    "ExtractedText",
    "MAX_FILE_BYTES",
    "MAX_TEXT_CHARS",
    "MODES",
    "SystemContextDraft",
    "ask_model",
    "extract_text",
    "llm_for_scope",
    "merge_context",
    "parse_answer",
    "shape_context",
    "text_from_paste",
]
