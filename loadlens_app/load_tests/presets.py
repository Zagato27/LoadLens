# SPDX-License-Identifier: Apache-2.0
"""Saved launch forms ("presets") per service in ``load_test_presets``.

A preset keeps the instances and options of a launch form so a team repeats the same test with one click.
Environment values are never stored: only the variable names, the values are typed again at launch.
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

from psycopg2 import sql

from .messages import LtError
from .store import _parse_dt, utcnow
from .tools import parse_env

DEFAULT_TABLE = "load_test_presets"
MAX_NAME_CHARS = 80
MAX_SPEC_BYTES = 64 * 1024
MAX_PRESETS_PER_SERVICE = 50
INSTANCE_KEYS = ("name", "generator", "tool", "script", "params", "start_delay_sec", "ci_ref", "ci_extra")
OPTION_KEYS = ("test_type", "max_duration_min", "auto_report", "use_llm")
MASK = "***"


class PresetError(LtError, ValueError):
    """A preset cannot be saved; ``message`` is an lt.* key with ``params``, a localized error or a ready text."""

    def __init__(self, message: Any, status: int = 400, **params: Any) -> None:
        super().__init__(message, **params)
        self.status = status


@dataclass
class PresetRecord:
    preset_id: str
    name: str
    service: str
    project_area: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    owner: str = ""
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def to_view(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "service": self.service,
            "project_area": self.project_area,
            "spec": copy.deepcopy(self.spec),
            "owner": self.owner,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def _plain(value: Any) -> Any:
    """Values of pipeline parameters as the form sends them: text, booleans and lists of text."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value][:100]
    return "" if value is None else str(value)[:4000]


def clean_spec(form: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a launch form a preset keeps; environment values become empty ``NAME=`` lines.

    Pipeline parameters keep their values except the password ones (``ci_secret`` of the form) and masked ones.
    """
    if not isinstance(form, Mapping):
        raise PresetError("lt.preset.spec_object")
    instances = form.get("instances")
    if not isinstance(instances, list) or not instances:
        raise PresetError("lt.preset.no_instances")
    out: dict[str, Any] = {key: form.get(key) for key in OPTION_KEYS if key in form}
    cleaned = []
    for raw in instances[:50]:
        if not isinstance(raw, Mapping):
            raise PresetError("lt.preset.instance_object")
        item = {key: raw.get(key) for key in INSTANCE_KEYS if raw.get(key) not in (None, "")}
        try:
            names = list(parse_env(raw.get("env")))
        except ValueError as exc:
            raise PresetError(exc) from exc
        item["env"] = "\n".join(f"{name}=" for name in names)
        params = raw.get("ci_params")
        if isinstance(params, Mapping) and params:
            secret = {str(name) for name in raw.get("ci_secret") or [] if name}
            item["ci_params"] = {
                str(name): "" if str(name) in secret or value == MASK else _plain(value)
                for name, value in list(params.items())[:100]
            }
        cleaned.append(item)
    out["instances"] = cleaned
    if len(json.dumps(out, ensure_ascii=False).encode("utf-8")) > MAX_SPEC_BYTES:
        raise PresetError("lt.preset.too_big")
    return out


class PresetStore:
    def list(self, services: Optional[list[str]] = None) -> list[PresetRecord]:
        raise NotImplementedError

    def get(self, preset_id: str) -> Optional[PresetRecord]:
        raise NotImplementedError

    def save(self, record: PresetRecord) -> PresetRecord:
        """Inserts, or replaces the preset with the same service and name."""
        raise NotImplementedError

    def delete(self, preset_id: str) -> bool:
        raise NotImplementedError

    def count(self, service: str) -> int:
        raise NotImplementedError


class MemoryPresetStore(PresetStore):
    def __init__(self) -> None:
        self._items: dict[str, PresetRecord] = {}
        self._lock = threading.Lock()

    def list(self, services: Optional[list[str]] = None) -> list[PresetRecord]:
        with self._lock:
            items = [copy.deepcopy(p) for p in self._items.values() if not services or p.service in services]
        return sorted(items, key=lambda p: (p.service, p.name.lower()))

    def get(self, preset_id: str) -> Optional[PresetRecord]:
        with self._lock:
            item = self._items.get(preset_id)
            return copy.deepcopy(item) if item else None

    def save(self, record: PresetRecord) -> PresetRecord:
        with self._lock:
            existing = next((p for p in self._items.values() if p.service == record.service and p.name == record.name), None)
            if existing:
                record.preset_id = existing.preset_id
                record.created_at = existing.created_at
            record.updated_at = utcnow()
            self._items[record.preset_id] = copy.deepcopy(record)
            return copy.deepcopy(record)

    def delete(self, preset_id: str) -> bool:
        with self._lock:
            return self._items.pop(preset_id, None) is not None

    def count(self, service: str) -> int:
        with self._lock:
            return sum(1 for p in self._items.values() if p.service == service)


def _table_ref() -> sql.Composed:
    from settings import CONFIG

    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return sql.SQL("{}.{}").format(sql.Identifier(str(cfg.get("schema", "public"))),
                                   sql.Identifier(str(cfg.get("load_test_presets_table", DEFAULT_TABLE))))


_COLUMNS = ("preset_id", "name", "service", "project_area", "spec", "owner", "created_at", "updated_at")


def _row(row: tuple) -> PresetRecord:
    values = dict(zip(_COLUMNS, row))
    spec = values.get("spec")
    if isinstance(spec, str):
        spec = json.loads(spec or "{}")
    return PresetRecord(
        preset_id=str(values["preset_id"]), name=str(values["name"]), service=str(values["service"]),
        project_area=str(values.get("project_area") or ""), spec=dict(spec or {}), owner=str(values.get("owner") or ""),
        created_at=_parse_dt(values.get("created_at")) or utcnow(), updated_at=_parse_dt(values.get("updated_at")) or utcnow(),
    )


class PgPresetStore(PresetStore):
    def __init__(self) -> None:
        self._ensured = False
        self._lock = threading.Lock()

    def _conn(self):
        from loadlens_app.core import _ts_conn

        conn = _ts_conn()
        if not self._ensured:
            with self._lock:
                prev = getattr(conn, "autocommit", False)
                conn.autocommit = True
                ok = False
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql.SQL(
                            """
                            CREATE TABLE IF NOT EXISTS {} (
                                preset_id    TEXT PRIMARY KEY,
                                name         TEXT NOT NULL,
                                service      TEXT NOT NULL,
                                project_area TEXT NOT NULL DEFAULT '',
                                spec         JSONB NOT NULL,
                                owner        TEXT NOT NULL DEFAULT '',
                                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                                updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                                UNIQUE (service, name)
                            )
                            """
                        ).format(_table_ref()))
                    ok = True
                finally:
                    if ok:
                        conn.autocommit = prev
                    else:
                        # Closed without touching autocommit: that would raise and hide the real error.
                        conn.close()
                self._ensured = True
        return conn

    def _select(self) -> sql.Composed:
        return sql.SQL("SELECT {} FROM {}").format(sql.SQL(", ").join(sql.Identifier(c) for c in _COLUMNS), _table_ref())

    def list(self, services: Optional[list[str]] = None) -> list[PresetRecord]:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                if services:
                    cur.execute(sql.SQL("{} WHERE service = ANY(%s) ORDER BY service, lower(name)").format(self._select()), (list(services),))
                else:
                    cur.execute(sql.SQL("{} ORDER BY service, lower(name)").format(self._select()))
                return [_row(row) for row in cur.fetchall() or []]
        finally:
            conn.close()

    def get(self, preset_id: str) -> Optional[PresetRecord]:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql.SQL("{} WHERE preset_id = %s").format(self._select()), (preset_id,))
                row = cur.fetchone()
        finally:
            conn.close()
        return _row(row) if row else None

    def save(self, record: PresetRecord) -> PresetRecord:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (preset_id, name, service, project_area, spec, owner) VALUES (%s, %s, %s, %s, %s::jsonb, %s)"
                        " ON CONFLICT (service, name) DO UPDATE SET spec = EXCLUDED.spec, owner = EXCLUDED.owner,"
                        " project_area = EXCLUDED.project_area, updated_at = now() RETURNING {}"
                    ).format(_table_ref(), sql.SQL(", ").join(sql.Identifier(c) for c in _COLUMNS)),
                    (record.preset_id, record.name, record.service, record.project_area, json.dumps(record.spec, ensure_ascii=False), record.owner),
                )
                return _row(cur.fetchone())
        finally:
            conn.close()

    def delete(self, preset_id: str) -> bool:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql.SQL("DELETE FROM {} WHERE preset_id = %s").format(_table_ref()), (preset_id,))
                return cur.rowcount == 1
        finally:
            conn.close()

    def count(self, service: str) -> int:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT count(*) FROM {} WHERE service = %s").format(_table_ref()), (service,))
                return int(cur.fetchone()[0])
        finally:
            conn.close()


_store: PresetStore = PgPresetStore()


def get_presets() -> PresetStore:
    return _store


def set_presets(store: PresetStore) -> PresetStore:
    global _store
    previous, _store = _store, store
    return previous


def save_preset(name: str, service: str, project_area: str, form: Mapping[str, Any], owner: str,
                is_admin: bool = False) -> PresetRecord:
    """Stores the form as a preset of the service; a preset of the same name is replaced by its author or an admin."""
    name = str(name or "").strip()
    if not name or len(name) > MAX_NAME_CHARS:
        raise PresetError("lt.preset.name_length", max=MAX_NAME_CHARS)
    store = get_presets()
    existing = next((p for p in store.list([service]) if p.name == name), None)
    if existing is not None and existing.owner and existing.owner != owner and not is_admin:
        raise PresetError("lt.preset.owned", 403, name=name, owner=existing.owner)
    if existing is None and store.count(service) >= MAX_PRESETS_PER_SERVICE:
        raise PresetError("lt.preset.too_many", 409, max=MAX_PRESETS_PER_SERVICE)
    record = PresetRecord(preset_id=uuid.uuid4().hex, name=name, service=service, project_area=project_area,
                          spec=clean_spec(form), owner=owner)
    return store.save(record)


__all__ = ["MemoryPresetStore", "PgPresetStore", "PresetError", "PresetRecord", "PresetStore", "clean_spec",
           "get_presets", "save_preset", "set_presets"]
