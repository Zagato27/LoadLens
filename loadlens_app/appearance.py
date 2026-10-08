"""Install-wide accent color: shades for both themes, CSS variables and a recolored logo."""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image

from settings import CONFIG

DEFAULT_ACCENT = "#6200ee"
_HEX = 16
_WHITE = (1.0, 1.0, 1.0)
_INK = (0x1A / 255, 0x1F / 255, 0x24 / 255)
_DARK_PAGE = (0x12 / 255, 0x12 / 255, 0x12 / 255)
_LIGHT_PAGE = (0xF4 / 255, 0xF6 / 255, 0xF8 / 255)
_MIN_TEXT = 4.5
_MIN_CHROME = 3.0
_SOFT_ALPHA = {"dark": 0.16, "light": 0.12}
_STATUS_GREEN = 142.0
_STATUS_RED = 0.0
_STATUS_SPAN = 18.0
_LOGO = Path(__file__).resolve().parent.parent / "templates" / "logo.png"
_COLORED_SATURATION = 0.08


class AppearanceError(ValueError):
    """Accent value is not a ``#rrggbb`` color."""

    def __init__(self, raw: object) -> None:
        super().__init__(f"Цвет акцента «{raw}» должен быть в формате #rrggbb")


@dataclass(frozen=True)
class AccentShades:
    accent: str
    hover: str
    soft: str
    focus: str
    on_accent: str


@dataclass(frozen=True)
class AccentPalette:
    accent: str
    dark: AccentShades
    light: AccentShades
    warnings: tuple[str, ...]


def parse_accent(raw: object) -> str:
    """``#rrggbb`` or ``rrggbb``, returned in lower case with a leading hash."""
    text = str(raw or "").strip().lower()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6 or any(char not in "0123456789abcdef" for char in text):
        raise AppearanceError(raw)
    return f"#{text}"


def accent_palette(accent: str) -> AccentPalette:
    """Shades that stay readable on the dark page and on the light page."""
    color = parse_accent(accent)
    return AccentPalette(
        accent=color,
        dark=_shades(color, "dark"),
        light=_shades(color, "light"),
        warnings=_warnings(color),
    )


def palette_css(palette: AccentPalette) -> str:
    """Overrides for a custom accent; empty for the built-in purple."""
    if palette.accent == DEFAULT_ACCENT:
        return ""
    return f":root[data-theme=dark]{{{_vars(palette.dark)}}}:root[data-theme=light]{{{_vars(palette.light)}}}"


def current_accent() -> str:
    """Stored accent, or the built-in purple when the stored value is unusable."""
    raw = (CONFIG.get("appearance") or {}).get("accent", DEFAULT_ACCENT)
    try:
        return parse_accent(raw)
    except AppearanceError:
        return DEFAULT_ACCENT


def template_appearance() -> dict[str, str]:
    """Values ``base.html`` prints: a bad stored color falls back to purple and is reported."""
    raw = (CONFIG.get("appearance") or {}).get("accent", DEFAULT_ACCENT)
    try:
        accent = parse_accent(raw)
        error = ""
    except AppearanceError as exc:
        accent = DEFAULT_ACCENT
        error = str(exc)
    palette = accent_palette(accent)
    return {"accent": accent, "key": accent[1:], "css": palette_css(palette), "error": error}


def palette_payload(accent: str) -> dict[str, object]:
    """Preview payload: both themes, the CSS block, the logo URL and status-color notes."""
    palette = accent_palette(accent)
    return {
        "accent": palette.accent,
        "css": palette_css(palette),
        "logo_url": f"/assets/logo.png?c={palette.accent[1:]}",
        "warnings": list(palette.warnings),
        "dark": _shade_dict(palette.dark),
        "light": _shade_dict(palette.light),
    }


@lru_cache(maxsize=32)
def recolored_logo(accent: str) -> bytes:
    """Logo PNG shifted to ``accent``; the built-in purple is the original file."""
    color = parse_accent(accent)
    if color == DEFAULT_ACCENT:
        return _LOGO.read_bytes()
    return _shift_logo(color)


def _shade_dict(shades: AccentShades) -> dict[str, str]:
    return {
        "accent": shades.accent,
        "hover": shades.hover,
        "soft": shades.soft,
        "focus": shades.focus,
        "on_accent": shades.on_accent,
    }


def _vars(shades: AccentShades) -> str:
    return (
        f"--accent:{shades.accent};--accent-hover:{shades.hover};--accent-soft:{shades.soft};"
        f"--focus:{shades.focus};--on-accent:{shades.on_accent};"
    )


def _shades(accent: str, theme: str) -> AccentShades:
    page = _DARK_PAGE if theme == "dark" else _LIGHT_PAGE
    rgb = _fit(_channels(accent), page)
    hover = _with_value(rgb, max(colorsys.rgb_to_hsv(*rgb)[2] * 0.78, 0.12))
    focus = _with_value(rgb, min(colorsys.rgb_to_hsv(*rgb)[2] + 0.18, 1.0)) if theme == "dark" else rgb
    return AccentShades(
        accent=_hex(rgb),
        hover=_hex(hover),
        soft=_rgba(rgb, _SOFT_ALPHA[theme]),
        focus=_hex(focus),
        on_accent=_on_accent(rgb),
    )


def _fit(rgb: tuple[float, float, float], page: tuple[float, float, float]) -> tuple[float, float, float]:
    """Keeps the color when text and page contrast hold; otherwise the nearest value that works."""
    if _serves(rgb, page):
        return rgb
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    best = rgb
    distance = 2.0
    for step in range(8, 97):
        candidate = colorsys.hsv_to_rgb(hue, saturation, step / 100)
        if not _serves(candidate, page):
            continue
        gap = abs(step / 100 - value)
        if gap < distance:
            best, distance = candidate, gap
    return best


def _serves(rgb: tuple[float, float, float], page: tuple[float, float, float]) -> bool:
    text = _WHITE if _on_accent(rgb) == "#ffffff" else _INK
    return _contrast(rgb, text) >= _MIN_TEXT and _contrast(rgb, page) >= _MIN_CHROME


def _on_accent(rgb: tuple[float, float, float]) -> str:
    white = _contrast(rgb, _WHITE)
    ink = _contrast(rgb, _INK)
    return "#ffffff" if white >= ink else "#1a1f24"


def _warnings(accent: str) -> tuple[str, ...]:
    hue, saturation, _value = colorsys.rgb_to_hsv(*_channels(accent))
    if saturation < 0.2:
        return ()
    degrees = hue * 360
    notes: list[str] = []
    if _hue_distance(degrees, _STATUS_GREEN) <= _STATUS_SPAN:
        notes.append("Цвет близок к зелёному статусу «ок» — статусы на экране будет сложнее отличить")
    if _hue_distance(degrees, _STATUS_RED) <= _STATUS_SPAN:
        notes.append("Цвет близок к красному статусу «провалы» — статусы на экране будет сложнее отличить")
    return tuple(notes)


def _hue_distance(left: float, right: float) -> float:
    gap = abs(left - right) % 360
    return min(gap, 360 - gap)


def _with_value(rgb: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    hue, saturation, _value = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _channels(color: str) -> tuple[float, float, float]:
    value = int(color[1:], _HEX)
    return ((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255


def _hex(rgb: tuple[float, float, float]) -> str:
    channels = [max(0, min(255, round(channel * 255))) for channel in rgb]
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _rgba(rgb: tuple[float, float, float], alpha: float) -> str:
    red, green, blue = [max(0, min(255, round(channel * 255))) for channel in rgb]
    return f"rgba({red},{green},{blue},{alpha})"


def _contrast(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    lighter = max(_luminance(left), _luminance(right))
    darker = min(_luminance(left), _luminance(right))
    return (lighter + 0.05) / (darker + 0.05)


def _luminance(rgb: tuple[float, float, float]) -> float:
    red, green, blue = (_linear(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _shift_logo(accent: str) -> bytes:
    image = Image.open(_LOGO).convert("RGBA")
    pixels = np.asarray(image).astype(np.float64)
    hsv = rgb_to_hsv(pixels[..., :3] / 255.0)
    target = rgb_to_hsv(np.array([_channels(accent)], dtype=np.float64))[0]
    source = rgb_to_hsv(np.array([_channels(DEFAULT_ACCENT)], dtype=np.float64))[0]
    colored = hsv[..., 1] >= _COLORED_SATURATION
    hsv[..., 0] = np.where(colored, target[0], hsv[..., 0])
    hsv[..., 1] = np.where(colored, _scaled_saturation(hsv[..., 1], source[1], target[1]), hsv[..., 1])
    pixels[..., :3] = np.round(np.clip(hsv_to_rgb(hsv), 0, 1) * 255)
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _scaled_saturation(saturation: np.ndarray, source: float, target: float) -> np.ndarray:
    if target < _COLORED_SATURATION or source <= 0:
        return np.zeros_like(saturation)
    return np.clip(saturation * (target / source), 0, 1)


__all__ = [
    "DEFAULT_ACCENT",
    "AccentPalette",
    "AccentShades",
    "AppearanceError",
    "accent_palette",
    "current_accent",
    "palette_css",
    "palette_payload",
    "parse_accent",
    "recolored_logo",
    "template_appearance",
]
