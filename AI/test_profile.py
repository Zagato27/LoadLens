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


@dataclass(frozen=True)
class TestProfile:
    """How a run of a given test type is analysed and presented."""

    __test__ = False  # not a pytest test class despite the name

    test_type: str
    raw_test_type: str
    mode: str
    label_ru: str
    peak_performance_applicable: bool
    peak_label_ru: str
    step_detection_default: bool
    step_detection_configurable: bool
    forecast_allowed: bool
    focus: str

    def to_dict(self) -> Dict[str, Any]:
        """Profile as stored in contexts and the final payload; ``test_type`` keeps the raw value."""
        data = asdict(self)
        data["test_type"] = self.raw_test_type
        data["normalized_test_type"] = self.test_type
        return data


_PROFILE_FIELDS: Dict[str, Dict[str, Any]] = {
    "step": {
        "mode": "capacity",
        "label_ru": "Ступенчатый (max perf)",
        "peak_performance_applicable": True,
        "peak_label_ru": "Максимальный RPS",
        "step_detection_default": True,
        "step_detection_configurable": True,
        "forecast_allowed": True,
        "focus": "Определить максимальную устойчивую производительность и момент деградации.",
    },
    "soak": {
        "mode": "stability",
        "label_ru": "Долговременный (soak)",
        "peak_performance_applicable": False,
        "peak_label_ru": "Удерживаемый RPS",
        "step_detection_default": False,
        "step_detection_configurable": False,
        "forecast_allowed": False,
        "focus": "Оценить удержание заданной нагрузки на всем интервале без накопления деградации.",
    },
    "spike": {
        "mode": "spike",
        "label_ru": "Всплески (spike)",
        "peak_performance_applicable": False,
        "peak_label_ru": "Уровень всплеска",
        "step_detection_default": False,
        "step_detection_configurable": False,
        "forecast_allowed": False,
        "focus": "Оценить реакцию на резкий рост нагрузки: рост задержек и ошибок в окне всплеска и время восстановления.",
    },
    "stress": {
        "mode": "stress",
        "label_ru": "Стресс",
        "peak_performance_applicable": True,
        "peak_label_ru": "Точка насыщения",
        "step_detection_default": True,
        "step_detection_configurable": True,
        "forecast_allowed": True,
        "focus": "Найти точку насыщения, характер деградации и ограничивающий ресурс.",
    },
}


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
    "TestProfile",
    "is_capacity_like",
    "mode_of",
    "normalize_test_type",
    "profile_for",
    "resolve_step_detection",
]
