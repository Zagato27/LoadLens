# SPDX-License-Identifier: Apache-2.0
"""Links from LoadLens to Jira: the issues of a release and the ticket of the release.

LoadLens never calls Jira. The global settings section ``jira`` (``enabled``, ``base_url``, ``project_key``) only
gives the address and the project the links are built from: the issues of a release are the issues of the project
whose ``fixVersion`` is the release name, the ticket is ``<base_url>/browse/<KEY-123>``. Without an enabled section
there are no links (``links`` returns None). A connection check and writing to Jira are left for later: the Jira
Server of the owner (8.13) has no personal access tokens.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional
from urllib.parse import quote, urlencode

from AI.i18n import LocalizedError

# http(s)://host[:port][/path] without a query, a fragment, spaces or a trailing slash.
BASE_URL_PATTERN = re.compile(r"^https?://[^\s/?#]+(?:/[^\s?#]*)?$")
PROJECT_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
TICKET_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*-\d+$")
JIRA_KEYS = ("enabled", "base_url", "project_key")


class JiraConfigError(LocalizedError, ValueError):
    """A value of the ``jira`` settings section is not valid."""


class TicketError(LocalizedError, ValueError):
    """A release ticket is not a Jira issue key."""


def jira_config(config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The ``jira`` section of the settings (``settings.CONFIG`` when none is given), {} when it is missing."""
    if config is None:
        from settings import CONFIG

        config = CONFIG
    section = config.get("jira") if isinstance(config, Mapping) else None
    return dict(section) if isinstance(section, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def validate_jira(payload: Mapping[str, Any]) -> None:
    """Checks the keys sent: ``enabled`` a boolean, ``base_url`` an http(s) address without a trailing «/»,
    ``project_key`` a Jira project key. An enabled section needs both the address and the key."""
    if "enabled" in payload and not isinstance(payload["enabled"], bool):
        raise JiraConfigError("server.jira.enabled_bool")
    base_url = payload.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise JiraConfigError("server.jira.base_url_invalid", value=str(base_url))
    base_url = _text(base_url)
    if base_url and (not BASE_URL_PATTERN.match(base_url) or base_url.endswith("/")):
        raise JiraConfigError("server.jira.base_url_invalid", value=base_url)
    project_key = payload.get("project_key")
    if project_key is not None and not isinstance(project_key, str):
        raise JiraConfigError("server.jira.project_key_invalid", value=str(project_key))
    project_key = _text(project_key)
    if project_key and not PROJECT_PATTERN.match(project_key):
        raise JiraConfigError("server.jira.project_key_invalid", value=project_key)
    if payload.get("enabled") is True and (not base_url or not project_key):
        raise JiraConfigError("server.jira.incomplete")


def normalize_ticket(raw: Any) -> str:
    """The ticket key of a release with surrounding spaces removed; '' clears it. Raises :class:`TicketError`."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise TicketError("server.release.ticket_invalid")
    value = raw.strip()
    if value and not TICKET_PATTERN.match(value):
        raise TicketError("server.release.ticket_invalid")
    return value


def enabled(cfg: Mapping[str, Any]) -> bool:
    """An enabled section with a valid address and project key: only then are links built."""
    base_url = _text(cfg.get("base_url"))
    project_key = _text(cfg.get("project_key"))
    return (
        cfg.get("enabled") is True
        and bool(BASE_URL_PATTERN.match(base_url))
        and not base_url.endswith("/")
        and bool(PROJECT_PATTERN.match(project_key))
    )


def _jql_string(value: str) -> str:
    """A JQL string literal: double quotes with the quotes and backslashes inside escaped."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def issues_url(cfg: Mapping[str, Any], release: str) -> Optional[str]:
    """Issue search of the project whose fixVersion is the release; None when Jira links are off."""
    if not enabled(cfg) or not str(release or "").strip():
        return None
    jql = f"project = {_text(cfg.get('project_key'))} AND fixVersion = {_jql_string(str(release).strip())}"
    return f"{_text(cfg.get('base_url'))}/issues/?" + urlencode({"jql": jql}, quote_via=quote)


def ticket_url(cfg: Mapping[str, Any], key: str) -> Optional[str]:
    """Page of the issue ``key``; None when Jira links are off or the key is not an issue key."""
    value = str(key or "").strip()
    if not enabled(cfg) or not TICKET_PATTERN.match(value):
        return None
    return f"{_text(cfg.get('base_url'))}/browse/{value}"


def links(cfg: Mapping[str, Any], release: str, ticket: str = "") -> Optional[dict[str, Optional[str]]]:
    """``{issues_url, ticket_url}`` of a release, None when Jira links are off."""
    if not enabled(cfg):
        return None
    return {"issues_url": issues_url(cfg, release), "ticket_url": ticket_url(cfg, ticket)}


__all__ = [
    "BASE_URL_PATTERN",
    "JIRA_KEYS",
    "JiraConfigError",
    "PROJECT_PATTERN",
    "TICKET_PATTERN",
    "TicketError",
    "enabled",
    "issues_url",
    "jira_config",
    "links",
    "normalize_ticket",
    "ticket_url",
    "validate_jira",
]
