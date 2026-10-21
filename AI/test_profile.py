# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for load-test types and their analysis profiles.

A run carries a free-form ``test_type`` string (UI sends step/soak/spike/stress, the API
accepts anything). Every layer — SLA evaluation, step derivation, prompts, forecast gating,
web and Confluence rendering — resolves it through :func:`profile_for` so the four types are
handled consistently:

- ``step``   → mode ``capacity``: search for the last stable load step (max performance);
- ``soak``   → mode ``stability``: hold a given load for the whole run, watch for drift;
- ``spike``  → mode ``spike``: reaction to sudden load bursts and recovery afterwards;
- ``stress`` → mode ``stress``: capacity semantics, the result is reported as a saturation point.

Unknown or empty types fall back to the step profile with ``test_type=""`` (legacy behaviour).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from AI.i18n import t
from AI.report_texts import LanguageTable, text_lang

TEST_TYPES = ("step", "soak", "spike", "stress")
MODES = ("capacity", "stability", "spike", "stress")
CAPACITY_LIKE_MODES = frozenset({"capacity", "stress"})

TEST_TYPE_ALIASES: Dict[str, tuple[str, ...]] = {
    "step": (
        "step", "capacity", "max", "max_performance", "load", "ступенчатый",
        "поиск_максимума", "поиск_максимальной_производительности",
    ),
    "soak": (
        "soak", "stability", "stable", "endurance", "long", "long_run", "longevity", "reliability",
        "стабильность", "стаб", "длительный", "длительная_нагрузка", "долговременный",
    ),
    "spike": ("spike", "burst", "всплеск", "всплески"),
    "stress": ("stress", "стресс"),
}
# Substring markers recognise free-form soak names such as "night-stability-run".
SOAK_MARKERS = ("stabil", "soak", "endurance", "longevity", "reliability", "стабил", "длительн")


DEFAULT_LANG = "ru"
# Dictionary keys of the visible texts of each type: the label (common.test_type.*, also LL.TEST_TYPE_LABELS in
# static/js/common.js), the name of its RPS figure (report.peak_label.*) and the focus of the analysis
# (report.focus.*). Readers build them from ``normalized_test_type``; reports saved before kept ``label_ru``.
TEST_TYPE_FIELDS = ("label", "peak_label", "focus")


def _field_text(key: str, field: str, lang: str) -> str:
    if field == "label":
        return t(f"common.test_type.{key}", locale=lang)
    if field == "peak_label":
        return t(f"report.peak_label.{key}", locale=lang)
    return t(f"report.focus.{key}", locale=lang)


# {lang: {type: {"label", "peak_label", "focus"}}}, rendered from the dictionaries on access.
TEST_TYPE_TEXTS = LanguageTable(TEST_TYPES, lambda key, lang: {field: _field_text(key, field, lang) for field in TEST_TYPE_FIELDS})


def test_type_text(key: Optional[str], field: str, lang: Optional[str] = DEFAULT_LANG) -> str:
    """``label`` / ``peak_label`` / ``focus`` of a normalized test type in ``lang``; '' for an unknown type."""
    code = str(key or "").strip().lower()
    if code not in TEST_TYPES or field not in TEST_TYPE_FIELDS:
        return ""
    return _field_text(code, field, text_lang(lang))


def test_type_label(key: Optional[str], lang: Optional[str] = DEFAULT_LANG) -> str:
    """Visible name of a normalized test type (step/soak/spike/stress); '' for an unknown type."""
    return test_type_text(key, "label", lang)


test_type_text.__test__ = False  # type: ignore[attr-defined]  # not pytest tests despite the names
test_type_label.__test__ = False  # type: ignore[attr-defined]


@dataclass(frozen=True)
class TestProfile:
    """How a run of a given test type is analysed and presented."""

    __test__ = False  # not a pytest test class despite the name

    test_type: str
    raw_test_type: str
    mode: str
    peak_performance_applicable: bool
    step_detection_default: bool
    step_detection_configurable: bool
    forecast_allowed: bool

    @property
    def _text_key(self) -> str:
        # Unknown types are analysed with the step profile and named like it.
        return self.test_type or "step"

    def label(self, lang: Optional[str] = DEFAULT_LANG) -> str:
        return test_type_text(self._text_key, "label", lang)

    def peak_label(self, lang: Optional[str] = DEFAULT_LANG) -> str:
        return test_type_text(self._text_key, "peak_label", lang)

    def focus(self, lang: Optional[str] = DEFAULT_LANG) -> str:
        return test_type_text(self._text_key, "focus", lang)

    def to_dict(self, lang: Optional[str] = DEFAULT_LANG) -> Dict[str, Any]:
        """Profile as stored in contexts and the final payload; ``test_type`` keeps the raw value.

        ``label`` and ``focus`` are for the model context in the language of the report; readers name the
        type by ``normalized_test_type``.
        """
        data = asdict(self)
        data["test_type"] = self.raw_test_type
        data["normalized_test_type"] = self.test_type
        data["label"] = self.label(lang)
        data["focus"] = self.focus(lang)
        return data


_PROFILE_FIELDS: Dict[str, Dict[str, Any]] = {
    "step": {
        "mode": "capacity",
        "peak_performance_applicable": True,
        "step_detection_default": True,
        "step_detection_configurable": True,
        "forecast_allowed": True,
    },
    "soak": {
        "mode": "stability",
        "peak_performance_applicable": False,
        "step_detection_default": False,
        "step_detection_configurable": False,
        "forecast_allowed": False,
    },
    "spike": {
        "mode": "spike",
        "peak_performance_applicable": False,
        "step_detection_default": False,
        "step_detection_configurable": False,
        "forecast_allowed": False,
    },
    "stress": {
        "mode": "stress",
        "peak_performance_applicable": True,
        "step_detection_default": True,
        "step_detection_configurable": True,
        "forecast_allowed": True,
    },
}


def stored_type_label(profile: Any, lang: Optional[str] = DEFAULT_LANG) -> str:
    """Name of the type of a stored ``test_profile``: by ``normalized_test_type``, else the saved
    ``label_ru`` / ``label`` of an old report, else ''."""
    if not isinstance(profile, Mapping):
        return ""
    label = test_type_label(profile.get("normalized_test_type"), lang)
    if label:
        return label
    for key in ("label_ru", "label"):
        text = str(profile.get(key) or "").strip()
        if text:
            return text
    return ""


def stored_type_focus(profile: Any, lang: Optional[str] = DEFAULT_LANG, fallback_key: str = "") -> str:
    """Focus of a stored ``test_profile`` (or of ``fallback_key`` when it has no known type); '' if unknown."""
    profile = profile if isinstance(profile, Mapping) else {}
    key = str(profile.get("normalized_test_type") or "").strip().lower() or fallback_key
    return test_type_text(key, "focus", lang)


def _canonical(raw: Optional[str]) -> str:
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_test_type(raw: Optional[str]) -> str:
    """Maps user-facing test type names (RU/EN) to step/soak/spike/stress, '' when unknown."""
    value = _canonical(raw)
    if not value:
        return ""
    for key, aliases in TEST_TYPE_ALIASES.items():
        if value in aliases:
            return key
    if any(marker in value for marker in SOAK_MARKERS):
        return "soak"
    return ""


def profile_for(raw: Optional[str]) -> TestProfile:
    """Analysis profile of a run; unknown types get the step profile with an empty ``test_type``."""
    normalized = normalize_test_type(raw)
    fields = dict(_PROFILE_FIELDS[normalized or "step"])
    if not normalized:
        # Legacy runs without a type: the step detector runs only when the config enables it.
        fields["step_detection_default"] = False
    return TestProfile(test_type=normalized, raw_test_type=str(raw or "").strip(), **fields)


def is_capacity_like(mode: Optional[str]) -> bool:
    """Modes that look for the last stable load step (step and stress tests)."""
    return str(mode or "").strip().lower() in CAPACITY_LIKE_MODES


def mode_of(test_profile: Any) -> str:
    """Mode of a stored ``test_profile`` dict (old reports have only capacity/stability)."""
    if isinstance(test_profile, TestProfile):
        return test_profile.mode
    if not isinstance(test_profile, Mapping):
        return "capacity"
    mode = str(test_profile.get("mode") or "").strip().lower()
    if mode in MODES:
        return mode
    if mode in {"soak", "endurance"}:
        return "stability"
    normalized = normalize_test_type(str(test_profile.get("test_type") or ""))
    return _PROFILE_FIELDS[normalized]["mode"] if normalized else "capacity"


def _flag(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_step_detection(profile: TestProfile, sla_cfg: Optional[Mapping[str, Any]]) -> bool:
    """Whether to run the step detector; ``sla.step_detection_enabled`` applies to step/stress only."""
    if not profile.step_detection_configurable:
        return profile.step_detection_default
    raw = (sla_cfg or {}).get("step_detection_enabled") if isinstance(sla_cfg, Mapping) else None
    return _flag(raw, profile.step_detection_default)


__all__ = [
    "CAPACITY_LIKE_MODES",
    "MODES",
    "SOAK_MARKERS",
    "TEST_TYPES",
    "TEST_TYPE_ALIASES",
    "TEST_TYPE_FIELDS",
    "TEST_TYPE_TEXTS",
    "TestProfile",
    "is_capacity_like",
    "mode_of",
    "normalize_test_type",
    "profile_for",
    "resolve_step_detection",
    "stored_type_focus",
    "stored_type_label",
    "test_type_label",
    "test_type_text",
]
