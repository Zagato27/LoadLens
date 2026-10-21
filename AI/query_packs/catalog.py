# SPDX-License-Identifier: Apache-2.0
"""Loads and validates the packs shipped in ``packs/*.toml`` and the bundles in ``bundles.toml``.

TOML keeps queries readable: literal strings ('...' and '''...''') need no escaping of the quotes
PromQL and InfluxQL are full of. A broken file is a programming error, so loading fails loudly with
the file name instead of skipping the pack.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .models import CATEGORY_TITLES, Bundle, BundleFile, QueryPack
from .render import PackRenderError, render_pack

PACKAGE_DIR = Path(__file__).resolve().parent
PACKS_DIR = PACKAGE_DIR / "packs"
BUNDLES_FILE = PACKAGE_DIR / "bundles.toml"
_CATEGORY_ORDER = {name: index for index, name in enumerate(CATEGORY_TITLES)}


class PackCatalogError(RuntimeError):
    """A pack or bundle file that does not load or does not validate."""


@dataclass(frozen=True)
class Catalog:
    packs: dict[str, QueryPack]
    bundles: dict[str, Bundle]

    def pack(self, pack_id: str) -> Optional[QueryPack]:
        return self.packs.get(pack_id)


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PackCatalogError(f"{path.name}: {exc}") from exc


def _load_pack(path: Path) -> QueryPack:
    try:
        pack = QueryPack.model_validate(_read_toml(path))
    except ValidationError as exc:
        raise PackCatalogError(f"{path.name}: {exc}") from exc
    if pack.id != path.stem:
        raise PackCatalogError(f"{path.name}: id «{pack.id}» must match the file name")
    try:
        rendered = render_pack(pack)
    except PackRenderError as exc:
        raise PackCatalogError(f"{path.name}: defaults do not render: {exc}") from exc
    if len({query.label for query in rendered.queries}) != len(rendered.queries):
        raise PackCatalogError(f"{path.name}: labels repeat after rendering the defaults")
    return pack


def load_catalog_from(packs_dir: Path, bundles_file: Path) -> Catalog:
    packs: dict[str, QueryPack] = {}
    for path in sorted(packs_dir.glob("*.toml")):
        pack = _load_pack(path)
        packs[pack.id] = pack
    ordered = dict(sorted(packs.items(), key=lambda item: (_CATEGORY_ORDER[item[1].category], item[1].title.lower())))
    bundles: dict[str, Bundle] = {}
    if bundles_file.exists():
        try:
            parsed = BundleFile.model_validate(_read_toml(bundles_file))
        except ValidationError as exc:
            raise PackCatalogError(f"{bundles_file.name}: {exc}") from exc
        for bundle in parsed.bundles:
            missing = [pack_id for pack_id in bundle.packs if pack_id not in packs]
            if missing:
                raise PackCatalogError(f"{bundles_file.name}: bundle {bundle.id} names unknown packs {missing}")
            if bundle.id in bundles:
                raise PackCatalogError(f"{bundles_file.name}: bundle {bundle.id} is repeated")
            bundles[bundle.id] = bundle
    _check_label_collisions(ordered)
    return Catalog(packs=ordered, bundles=bundles)


def _check_label_collisions(packs: dict[str, QueryPack]) -> None:
    """Two packs of one domain must not render the same label: merging one would overwrite the other."""
    owners: dict[tuple[str, str], str] = {}
    for pack in packs.values():
        for query in render_pack(pack).queries:
            key = (pack.domain, query.label)
            if key in owners:
                raise PackCatalogError(f"{pack.id}: label «{query.label}» is also used by {owners[key]}")
            owners[key] = pack.id


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    return load_catalog_from(PACKS_DIR, BUNDLES_FILE)


__all__ = ["BUNDLES_FILE", "Catalog", "PACKS_DIR", "PackCatalogError", "load_catalog", "load_catalog_from"]
