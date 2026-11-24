# SPDX-License-Identifier: Apache-2.0
"""Saved LLM connections and their effective configuration for a project."""

from copy import deepcopy
from uuid import uuid4

from AI.i18n import t
from AI.providers import SUPPORTED_PROVIDERS, llm_proxy_url
from loadlens_app import core
from loadlens_app.config_secrets import destination_changed, mask_secrets, restore_secrets

SECTION = "llm_profiles"
PROVIDER_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic", "perplexity": "Perplexity", "gigachat": "GigaChat"}


class ProfileError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _legacy_id(provider: str, project: bool = False) -> str:
    return f"legacy-project-{provider}" if project else f"legacy-{provider}"


def _legacy_provider(section: dict) -> str:
    return str(section.get("provider") or "").strip().lower()


class Connections:
    def __init__(self, runtime: dict):
        self.runtime = runtime
        self.areas = _mapping(runtime.get("per_area"))
        self.global_llm = core._deep_merge_dicts(_mapping(core.CONFIG.get("llm")), _mapping(runtime.get("llm")))

    def _area_llm(self, area: str) -> dict:
        base = _mapping(_mapping(core.CONFIG.get("per_area")).get(area))
        own = _mapping(self.areas.get(area))
        return _mapping(own.get("llm", base.get("llm"))) if area else {}

    def _stored(self, area: str) -> dict:
        return _mapping((_mapping(self.areas.get(area)) if area else self.runtime).get(SECTION))

    def own_profiles(self, area: str) -> dict:
        legacy = self._area_llm(area) if area else self.global_llm
        out = {}
        for provider in SUPPORTED_PROVIDERS:
            if not isinstance(legacy.get(provider), dict):
                continue
            base = _mapping(self.global_llm.get(provider))
            settings = core._deep_merge_dicts(base, legacy[provider]) if area else base
            moved = bool(area and destination_changed(settings, base))
            if moved:
                for key in ("api_key", "credentials", "cert_file", "key_file"):
                    if key not in legacy[provider]:
                        settings.pop(key, None)
            own_key = area and "api_key" in legacy[provider] and legacy[provider]["api_key"] != base.get("api_key")
            use_env = not (moved or own_key)
            out[_legacy_id(provider, bool(area))] = {
                "name": PROVIDER_NAMES[provider], "provider": provider,
                "settings": settings, "use_env": use_env,
            }
        for profile_id, entry in _mapping(self._stored(area).get("profiles")).items():
            if isinstance(entry, dict):
                out[profile_id] = entry
            else:
                out.pop(profile_id, None)
        return out

    def preferred_id(self, area: str) -> str:
        stored = self._stored(area)
        if "active" in stored:
            return str(stored["active"] or "")
        provider = _legacy_provider(self._area_llm(area) if area else self.global_llm)
        if area and not provider:
            provider = _legacy_provider(self.global_llm)
            own_id = _legacy_id(provider, True)
            return own_id if own_id in self.own_profiles(area) else ""
        own_id = _legacy_id(provider, bool(area))
        return _legacy_id(provider) if area and own_id not in self.own_profiles(area) else own_id

    def active(self, area: str) -> tuple[str, dict]:
        own = self.own_profiles(area)
        selected = self.preferred_id(area)
        if area:
            available = {**self.own_profiles(""), **own}
            if selected in available:
                return selected, available[selected]
            return self.active("")
        return selected, own.get(selected, {})

    def analysis_settings(self, area: str) -> dict:
        def legacy_options(section: dict) -> dict:
            return {key: value for key, value in section.items() if key not in (*SUPPORTED_PROVIDERS, "provider")}

        global_options = self._stored("").get("analysis", legacy_options(self.global_llm))
        if not area:
            return deepcopy(global_options)
        own = self._stored(area)
        if "analysis" in own:
            return deepcopy(own["analysis"])
        return deepcopy(core._deep_merge_dicts(global_options, legacy_options(self._area_llm(area))))

    def effective(self, area: str) -> dict:
        profile_id, profile = self.active(area)
        llm = self.analysis_settings(area)
        if profile:
            provider = profile["provider"]
            llm.update(provider=provider, _profile_id=profile_id, _use_env_keys=profile.get("use_env", False))
            llm[provider] = deepcopy(profile["settings"])
        elif "active" not in self._stored(""):
            # An older installation can specify a provider and use its built-in connection defaults.
            legacy = core._deep_merge_dicts(self.global_llm, self._area_llm(area))
            llm.update({key: deepcopy(legacy[key]) for key in ("provider", *SUPPORTED_PROVIDERS) if key in legacy})
        return llm

    def view(self, area: str) -> dict:
        active_id, active = self.active(area)
        scopes = ("", area) if area else ("",)
        return {
            "profiles": [
                {
                    "id": profile_id, "area": profile.get("origin_area", scope), "shared": not scope,
                    **mask_secrets({k: v for k, v in profile.items() if k not in ("use_env", "origin_area")}),
                }
                for scope in scopes for profile_id, profile in self.own_profiles(scope).items()
            ],
            "active_id": active_id if active else "",
            "inherits": bool(area and not self.preferred_id(area)),
            "analysis": mask_secrets(self.analysis_settings(area)),
            "llm": mask_secrets({k: v for k, v in self.effective(area).items() if not k.startswith("_")}),
        }

    def _writable(self, area: str) -> dict:
        node = self.runtime
        if area:
            self.areas = self.runtime.setdefault("per_area", {})
            node = self.areas.setdefault(area, {})
        return node.setdefault(SECTION, {})

    def _editable(self, area: str, profile_id: str) -> tuple[str, dict | None]:
        """Resolve an editor's project to the scope holding the saved credentials."""
        if not profile_id:
            return area, None
        stored = self.own_profiles(area).get(profile_id)
        if stored and stored.get("origin_area", area) == area:
            return area, stored
        if area:
            shared = _mapping(self._stored("").get("profiles")).get(profile_id)
            if isinstance(shared, dict) and shared.get("origin_area") == area:
                return "", shared
        raise ProfileError(t("server.llm_profiles.missing"), 404)

    def prepare(self, area: str, profile_id: str, data: dict) -> dict:
        _, stored = self._editable(area, profile_id)
        return self._prepare_settings(data, stored)

    @staticmethod
    def _prepare_settings(data: dict, stored: dict | None) -> dict:
        provider = data.get("provider")
        settings = data.get("settings")
        if provider not in SUPPORTED_PROVIDERS or not isinstance(settings, dict):
            raise ProfileError(t("server.llm_profiles.invalid_provider"))
        if stored and provider != stored["provider"]:
            raise ProfileError(t("server.llm_profiles.invalid_provider"))
        if "proxy_url" in settings:
            try:
                proxy_url = llm_proxy_url(settings["proxy_url"])
            except ValueError as exc:
                raise ProfileError(str(exc)) from exc
            if proxy_url and provider == "gigachat":
                raise ProfileError(t("llm.error.gigachat_proxy_url"))
            settings = {**settings, "proxy_url": proxy_url}
        previous = stored["settings"] if stored else {}
        restored = restore_secrets(settings, previous, complete=True)
        use_env = bool(stored and stored.get("use_env") and restored.get("api_key") == previous.get("api_key")
                       and not destination_changed(restored, previous, complete=True))
        return {"provider": provider, "settings": restored, "use_env": use_env}

    def save(self, area: str, profile_id: str, data: dict) -> str:
        name = data.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            raise ProfileError(t("server.llm_profiles.invalid_name"))
        scope, stored = self._editable(area, profile_id)
        shared = data.get("shared", not scope)
        if not isinstance(shared, bool) or (not area and not shared):
            raise ProfileError(t("server.llm_profiles.invalid_sharing"))
        target = "" if shared else area
        profile = self._prepare_settings(data, stored)
        if stored and target != scope:
            if not scope:
                self._require_unused_global(profile_id, except_area=area)
            selected = self.preferred_id(area) == profile_id
            self._remove(scope, profile_id)
            if shared:
                # Project copies and legacy providers can have the same local ID.
                profile_id = uuid4().hex
            if selected:
                self._writable(area)["active"] = profile_id
        profile_id = profile_id or uuid4().hex
        profile["name"] = name.strip()
        if shared and area:
            profile["origin_area"] = area
        node = self._writable(target)
        node.setdefault("profiles", {})[profile_id] = profile
        if not area and not self.active("")[1]:
            node["active"] = profile_id
        return profile_id

    def activate(self, area: str, profile_id: str) -> None:
        if not (area and not profile_id) and profile_id not in {**self.own_profiles(""), **self.own_profiles(area)}:
            raise ProfileError(t("server.llm_profiles.missing"), 404)
        self._writable(area)["active"] = profile_id

    def save_analysis(self, area: str, data: dict) -> None:
        if any(key in (*SUPPORTED_PROVIDERS, "provider") or key.startswith("_") for key in data):
            raise ProfileError(t("server.llm_profiles.invalid_analysis"))
        self._writable(area)["analysis"] = data

    def reset_project(self, area: str) -> bool:
        stored = self._stored(area)
        if not self._area_llm(area) and not stored.get("active") and "analysis" not in stored:
            return False
        node = self._writable(area)
        node.setdefault("profiles", {}).update(self.own_profiles(area))
        node["active"] = ""
        node.pop("analysis", None)
        self.runtime["per_area"][area]["llm"] = {}
        return True

    def delete(self, area: str, profile_id: str) -> None:
        scope, stored = self._editable(area, profile_id)
        if stored is None:
            raise ProfileError(t("server.llm_profiles.missing"), 404)
        if not scope:
            self._require_unused_global(profile_id)
        elif self.active(area)[0] == profile_id:
            raise ProfileError(t("server.llm_profiles.in_use"), 409)
        self._remove(scope, profile_id)

    def _require_unused_global(self, profile_id: str, except_area: str = "") -> None:
        areas = self.areas.keys() | _mapping(core.CONFIG.get("per_area")).keys()
        projects = sorted(name for name in areas if name != except_area and self.preferred_id(name) == profile_id)
        if projects:
            raise ProfileError(t("server.llm_profiles.in_use_projects", projects=", ".join(projects)), 409)
        if self.active("")[0] == profile_id:
            raise ProfileError(t("server.llm_profiles.in_use_global"), 409)

    def _remove(self, area: str, profile_id: str) -> None:
        profiles = self._writable(area).setdefault("profiles", {})
        if profile_id.startswith("legacy-"):
            profiles[profile_id] = None
        else:
            del profiles[profile_id]

    def release_project_profiles(self, area: str) -> None:
        """Keep shared connections available after their originating project is deleted."""
        for profile in _mapping(self._stored("").get("profiles")).values():
            if isinstance(profile, dict) and profile.get("origin_area") == area:
                del profile["origin_area"]


def connections() -> Connections:
    return Connections(core._settings_runtime_view())


def effective_llm(area: str | None = None) -> dict:
    return connections().effective(area or "")
