#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Server side of tools/deploy.py: updates a LoadLens source checkout that systemd units run, and restarts them.
# tools/deploy.py sends this file over SSH and starts it as
#   bash -c "<this file>" loadlens-deploy MODE APP_DIR BRANCH REF UNITS HEALTH_URL HEALTH_WAIT PYTHON RUN_AS RESTART
#
#   check     read-only: the checkout, the units, the venv Python, the health URL and sudo; with an empty APP_DIR
#             or UNITS it lists the services that look like LoadLens
#   deploy    fetch origin/BRANCH, move the checkout to REF (a commit of origin/BRANCH), pip install when
#             requirements.txt changed, restart UNITS, wait until they stay active and HEALTH_URL answers
#   rollback  the same with the commit the last deploy replaced (kept in refs/loadlens/previous), without a fetch
#
# Nothing changes when the checkout has modified tracked files, is on another branch or has commits that
# origin/BRANCH has not. A failed pip install puts the previous commit back before anything restarts. Untracked
# and ignored files (settings.py, the runtime JSON files, the venv) are never touched.
# LOADLENS_DEPLOY_POLL_SEC (2) is the pause between readiness checks.
set -euo pipefail

MODE=${1:?mode}
APP_DIR=${2:-}
BRANCH=${3:-main}
REF=${4:-}
UNITS_TEXT=${5:-}
HEALTH_URL=${6:-}
HEALTH_WAIT=${7:-90}
PYTHON_BIN=${8:-}
RUN_AS=${9:-}
RESTART=${10:-0}
POLL=${LOADLENS_DEPLOY_POLL_SEC:-2}

read -r -a UNITS <<< "$UNITS_TEXT" || true

say() { printf '==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

SUDO=()
[ "$(id -u)" -eq 0 ] || SUDO=(sudo)
AS=()
if [ -n "$RUN_AS" ] && [ "$RUN_AS" != "$(id -un)" ]; then AS=(sudo -u "$RUN_AS" -H); fi

# git and pip run as the owner of the checkout, systemctl and journalctl with root rights.
as_owner() { if [ ${#AS[@]} -gt 0 ]; then "${AS[@]}" "$@"; else "$@"; fi; }
privileged() { if [ ${#SUDO[@]} -gt 0 ]; then "${SUDO[@]}" "$@"; else "$@"; fi; }
git_() { as_owner git -C "$APP_DIR" "$@"; }
describe() { git_ log -1 --format='%h %s' "$1"; }

http_ok() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 5 --noproxy '*' -o /dev/null "$1" 2>/dev/null
    else
        python3 -c 'import sys, urllib.request as u; u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=5)' \
            "$1" >/dev/null 2>&1
    fi
}

units_active() {
    local unit
    for unit in "${UNITS[@]}"; do
        systemctl is-active --quiet "$unit" || return 1
    done
}

ready() { units_active && { [ -z "$HEALTH_URL" ] || http_ok "$HEALTH_URL"; }; }

# Two passing checks in a row: a unit that dies right after the start must not count as up.
wait_ready() {
    local deadline=$((SECONDS + HEALTH_WAIT)) passed=0
    while :; do
        if ready; then
            passed=$((passed + 1))
            [ "$passed" -lt 2 ] || return 0
        else
            passed=0
            [ "$SECONDS" -lt "$deadline" ] || return 1
        fi
        sleep "$POLL"
    done
}

show_logs() {
    local unit
    for unit in "${UNITS[@]}"; do
        printf '\n--- %s ---\n' "$unit" >&2
        systemctl --no-pager --lines=0 status "$unit" >&2 || true
        privileged journalctl -u "$unit" -n 40 --no-pager >&2 || true
    done
}

# The venv Python: "python" of the config, else the interpreter of ExecStart or the python next to it
# (gunicorn, celery) in the same bin directory.
resolve_python() {
    if [ -n "$PYTHON_BIN" ]; then
        printf '%s\n' "$PYTHON_BIN"
        return 0
    fi
    local unit exe
    for unit in "${UNITS[@]}"; do
        exe=$(systemctl show -p ExecStart --value "$unit" 2>/dev/null | sed -n 's/^{ path=\([^ ;]*\).*/\1/p;q') || true
        case "$exe" in
            */python | */python3 | */python3.*)
                printf '%s\n' "$exe"
                return 0
                ;;
            ?*)
                if [ -x "${exe%/*}/python" ]; then
                    printf '%s\n' "${exe%/*}/python"
                    return 0
                fi
                ;;
        esac
    done
    return 1
}

# Units whose command line or working directory mentions LoadLens; passwords in URLs are masked.
discover_units() {
    local unit info found=0
    while read -r unit _; do
        case "$unit" in *@.service | "") continue ;; esac
        info=$(systemctl show -p ExecStart -p WorkingDirectory "$unit" 2>/dev/null) || continue
        if printf '%s\n' "$info" | grep -Eq 'loadlens|app\.py|celery|gunicorn'; then
            found=1
            printf '    %s (%s)\n' "$unit" "$(systemctl is-active "$unit" 2>/dev/null || true)"
            printf '%s\n' "$info" | sed -n \
                -e 's/^WorkingDirectory=\(..*\)/        WorkingDirectory=\1/p' \
                -e 's/^ExecStart=.*argv\[\]=\([^;]*\);.*/        ExecStart=\1/p' |
                sed -E 's#://[^@/ ]*@#://***@#g'
        fi
    done < <(systemctl list-unit-files --type=service --no-legend 2>/dev/null || true)
    [ "$found" = 1 ] || printf '    (none found)\n'
}

do_check() {
    say "connected as $(id -un) on $(hostname)"
    if [ ${#UNITS[@]} -eq 0 ] || [ -z "$APP_DIR" ]; then
        say "services that look like LoadLens: put them into \"services\" and their WorkingDirectory into \"app_dir\""
        discover_units
    fi
    if [ -n "$APP_DIR" ]; then
        say "checkout $APP_DIR (owner $(stat -c %U "$APP_DIR" 2>/dev/null || echo '?'))"
        if git_ rev-parse --git-dir >/dev/null 2>&1; then
            printf '    branch  %s\n' "$(git_ symbolic-ref --short -q HEAD || echo '(detached)')"
            printf '    commit  %s\n' "$(git_ log -1 --format='%h %ad %s' --date=short)"
            printf '    origin  %s\n' "$(git_ remote get-url origin 2>/dev/null || echo '(none)')"
            local dirty previous
            dirty=$(git_ status --porcelain --untracked-files=no)
            if [ -n "$dirty" ]; then
                warn "modified tracked files, a deploy will refuse to start:"
                printf '%s\n' "$dirty" >&2
            fi
            previous=$(git_ rev-parse -q --verify 'refs/loadlens/previous^{commit}' || true)
            [ -z "$previous" ] || printf '    rollback target  %s\n' "$(describe "$previous")"
        else
            warn "git cannot use $APP_DIR (not a clone, or owned by another user: set \"run_as\"):"
            git_ rev-parse --git-dir 2>&1 | sed 's/^/    /' >&2 || true
        fi
    fi
    if [ ${#UNITS[@]} -gt 0 ]; then
        local unit py
        say "services"
        for unit in "${UNITS[@]}"; do
            printf '    %-32s %s\n' "$unit" "$(systemctl is-active "$unit" 2>/dev/null || true)"
        done
        if py=$(resolve_python); then
            printf '    python  %s (%s)\n' "$py" "$(as_owner "$py" --version 2>&1 || echo 'does not start')"
        else
            warn "the venv Python is not found in ExecStart: set \"python\" for pip installs"
        fi
    fi
    if [ -n "$HEALTH_URL" ]; then
        if http_ok "$HEALTH_URL"; then say "health $HEALTH_URL: ok"; else warn "health $HEALTH_URL: no answer"; fi
    fi
    if [ ${#SUDO[@]} -gt 0 ]; then
        if sudo -n true 2>/dev/null; then
            say "sudo works without a password"
        else
            say "sudo will ask for a password (tools/deploy.py must run in a terminal)"
        fi
    fi
}

do_deploy() {
    [ -n "$APP_DIR" ] && [ ${#UNITS[@]} -gt 0 ] || die "app_dir and services are required"
    git_ rev-parse --git-dir >/dev/null || die "git cannot use $APP_DIR (owned by another user? set \"run_as\")"
    local branch dirty current target
    branch=$(git_ symbolic-ref --short -q HEAD || true)
    [ "$branch" = "$BRANCH" ] || die "the checkout is on '${branch:-detached HEAD}', expected '$BRANCH'"
    dirty=$(git_ status --porcelain --untracked-files=no)
    [ -z "$dirty" ] || die "modified tracked files on the server, commit or revert them first:
$dirty"
    current=$(git_ rev-parse HEAD)
    if [ "$MODE" = rollback ]; then
        target=$(git_ rev-parse -q --verify 'refs/loadlens/previous^{commit}') || die "no previous deploy is recorded"
    else
        say "fetching origin/$BRANCH"
        git_ fetch --quiet origin "$BRANCH"
        git_ merge-base --is-ancestor "$current" "origin/$BRANCH" ||
            die "the checkout has commits that origin/$BRANCH has not"
        target=$(git_ rev-parse -q --verify "${REF:-origin/$BRANCH}^{commit}") ||
            die "commit $REF is not on the server after the fetch: push it first"
        git_ merge-base --is-ancestor "$target" "origin/$BRANCH" || die "commit $REF is not on origin/$BRANCH"
    fi

    if [ "$current" = "$target" ] && [ "$RESTART" != 1 ]; then
        say "already at $(describe "$current"): nothing to do (--restart restarts the services anyway)"
        return 0
    fi

    local need_pip=0 py=""
    git_ diff --quiet "$current" "$target" -- requirements.txt || need_pip=1
    if [ "$need_pip" = 1 ]; then
        py=$(resolve_python) || die "requirements.txt changed and the venv Python is unknown: set \"python\""
    fi

    if [ "$current" != "$target" ]; then
        say "$(describe "$current")  ->  $(describe "$target")"
        git_ log --oneline --no-decorate -n 30 "$current..$target" | sed 's/^/    /'
        git_ reset --quiet --keep "$target"
    fi
    if [ "$need_pip" = 1 ]; then
        say "requirements.txt changed: pip install"
        if ! as_owner "$py" -m pip install --quiet --disable-pip-version-check -r "$APP_DIR/requirements.txt"; then
            git_ reset --quiet --keep "$current"
            die "pip install failed: the checkout is back at $(describe "$current"), nothing was restarted"
        fi
    fi
    [ "$current" = "$target" ] || git_ update-ref refs/loadlens/previous "$current"

    say "restarting ${UNITS[*]}"
    privileged systemctl restart "${UNITS[@]}"
    say "waiting up to ${HEALTH_WAIT}s for the services${HEALTH_URL:+ and $HEALTH_URL}"
    if ! wait_ready; then
        show_logs
        if [ "$current" != "$target" ]; then
            die "the services did not come up: python tools/deploy.py rollback returns to $(describe "$current")"
        fi
        die "the services did not come up"
    fi
    say "done: $(describe HEAD)"
}

case "$MODE" in
    check) do_check ;;
    deploy | rollback) do_deploy ;;
    *) die "unknown mode $MODE" ;;
esac
