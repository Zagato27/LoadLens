"""Lightweight key validation for configuration sections.

There is no formal schema: the pristine ``settings.py`` defaults, together with
``settings.example.py`` shipped with the code, act as the reference. Keys that
are absent from both are reported as warnings so a typo such as ``max_p95_m`` is
visible instead of being silently ignored.
"""

from __future__ import annotations

import copy
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from loadlens_app.appearance import AppearanceError, parse_accent

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.py"
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "settings.example.py"

# Sections with free-form structure are not validated.
VALIDATED_SECTIONS = frozenset({
    "llm", "logs_source", "confluence",
    "storage", "storage.timescale", "default_params", "sla", "appearance",
})


def _load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(f"loadlens_settings_{path.stem.replace('.', '_')}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load settings from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise RuntimeError(f"{path} does not define a CONFIG dict")
    return config


def _with_missing_keys(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _with_missing_keys(out[key], value)
        elif key not in out:
            out[key] = copy.deepcopy(value)
    return out


@lru_cache(maxsize=1)
def default_config() -> dict:
    """Known keys: ``settings.CONFIG`` from disk plus keys of ``settings.example.py``.

    A deployment keeps its own settings.py, which can lag behind the code; keys the
    code ships in the example are valid there and must not be reported as typos.
    """
    config = _load_config(SETTINGS_PATH)
    if EXAMPLE_PATH.exists():
        config = _with_missing_keys(config, _load_config(EXAMPLE_PATH))
    return config


def _section_defaults(section: str) -> Any:
    node: Any = default_config()
    for part in section.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _collect_unknown(payload: Any, reference: Any, path: str, out: list[str]) -> None:
    if not isinstance(payload, dict) or not isinstance(reference, dict):
        return
    if not reference:
        # Empty default dict (e.g. proxies) accepts arbitrary keys.
        return
    for key, value in payload.items():
        key_path = f"{path}.{key}" if path else str(key)
        if key not in reference:
            out.append(key_path)
            continue
        _collect_unknown(value, reference[key], key_path, out)


def unknown_keys(section: str, payload: dict) -> list[str]:
    """Returns dotted paths of keys in ``payload`` that do not exist in the defaults."""
    if section not in VALIDATED_SECTIONS:
        return []
    reference = _section_defaults(section)
    if not isinstance(reference, dict):
        return []
    found: list[str] = []
    _collect_unknown(payload, reference, section, found)
    return found


def known_keys_hint(section: str, unknown_path: str) -> list[str]:
    """Known sibling keys for an unknown path, to help spot typos."""
    parent_parts = unknown_path.split(".")[:-1]
    node: Any = _section_defaults(section)
    for part in parent_parts[1:]:
        if not isinstance(node, dict):
            return []
        node = node.get(part)
    return sorted(node.keys()) if isinstance(node, dict) else []


class QueryConfigError(ValueError):
    """queries arrays that the report builder cannot line up."""


SLA_CHOICES: dict[str, tuple[str, ...]] = {"latency_unit": ("ms", "s"), "load_model": ("open", "closed")}


class SlaConfigError(ValueError):
    """sla values outside the allowed choices."""


def validate_sla(payload: dict) -> None:
    """Choice fields of ``sla`` must hold one of the allowed values."""
    for key, allowed in SLA_CHOICES.items():
        if key in payload and payload[key] not in allowed:
            raise SlaConfigError(f"sla.{key}: допустимо {', '.join(allowed)}, получено «{payload[key]}»")


def validate_appearance(payload: dict) -> None:
    """``appearance.accent`` must be a ``#rrggbb`` color."""
    if not isinstance(payload, dict):
        raise AppearanceError(payload)
    parse_accent(payload.get("accent"))


def validate_queries(payload: dict) -> None:
    """Each non-empty query list must match ``labels``, with unique non-empty names."""
    if not isinstance(payload, dict):
        raise QueryConfigError("queries должен быть объектом")
    query_keys = ("promql_queries", "flux_queries", "influxql_queries")
    for domain, block in payload.items():
        if not isinstance(block, dict):
            raise QueryConfigError(f"{domain}: ожидается объект")
        labels = block.get("labels") or []
        if not isinstance(labels, list):
            raise QueryConfigError(f"{domain}: labels должен быть списком")
        seen: set[str] = set()
        for index, label in enumerate(labels, start=1):
            text = str(label or "").strip()
            if not text:
                raise QueryConfigError(f"{domain}: пустое название в строке {index}")
            if text in seen:
                raise QueryConfigError(f"{domain}: название «{text}» повторяется")
            seen.add(text)
        for key in query_keys:
            queries = block.get(key) or []
            if not isinstance(queries, list):
                raise QueryConfigError(f"{domain}: {key} должен быть списком")
            if queries and len(queries) != len(labels):
                raise QueryConfigError(
                    f"{domain}: {key} ({len(queries)}) и labels ({len(labels)}) разной длины, строка 1"
                )
        prom_keys = block.get("label_keys_list") or []
        if (block.get("promql_queries") or []) and len(prom_keys) != len(labels):
            raise QueryConfigError(f"{domain}: label_keys_list и labels разной длины")
        uses_tags = bool(block.get("flux_queries") or block.get("influxql_queries"))
        tag_keys = block.get("label_tag_keys_list") or []
        if uses_tags and len(tag_keys) != len(labels):
            raise QueryConfigError(f"{domain}: label_tag_keys_list и labels разной длины")


def warnings_for(section: str, payload: dict) -> list[str]:
    """Human-readable warnings about unknown keys."""
    messages: list[str] = []
    for path in unknown_keys(section, payload):
        hint = known_keys_hint(section, path)
        suffix = f" Известные ключи: {', '.join(hint)}." if hint else ""
        messages.append(f"Неизвестный ключ «{path}» — возможно, опечатка; он будет сохранён, но не используется.{suffix}")
    return messages


def iter_sections() -> Iterable[str]:
    return sorted(VALIDATED_SECTIONS)


__all__ = [
    "SLA_CHOICES",
    "SlaConfigError",
    "VALIDATED_SECTIONS",
    "QueryConfigError",
    "default_config",
    "unknown_keys",
    "warnings_for",
    "known_keys_hint",
    "validate_appearance",
    "validate_queries",
    "validate_sla",
]
