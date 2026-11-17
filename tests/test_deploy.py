# SPDX-License-Identifier: Apache-2.0
"""tools/deploy.py and tools/deploy_remote.sh: update a source checkout run by systemd units, over SSH.

No host is contacted. The remote command line built by tools/deploy.py runs under a local bash against throwaway git
repositories (an "origin", a developer clone and the "server" checkout); systemctl, journalctl, sudo, curl and the
venv Python are stubs that record their calls.
"""

import base64
import gzip
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("deploy_tool", ROOT / "tools" / "deploy.py")
deploy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = deploy
_spec.loader.exec_module(deploy)

UNITS = ("loadlens", "loadlens-worker")


def _bash() -> str | None:
    if os.name == "nt":
        # System32\bash.exe is the WSL launcher, which sees other paths: only Git Bash runs these tests.
        for base in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"), r"C:\Program Files"):
            candidate = Path(base or "") / "Git" / "bin" / "bash.exe"
            if base and candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None or shutil.which("git") is None, reason="needs bash and git")


def _bash_path(path: Path) -> str:
    """A directory for bash's PATH: Git Bash wants /c/... instead of C:/... (the colon separates PATH entries)."""
    text = path.as_posix()
    if os.name == "nt" and re.match(r"^[A-Za-z]:/", text):
        return "/" + text[0].lower() + text[2:]
    return text


# --- configuration and the command line ---------------------------------------------------------------------------


def test_config_defaults_and_comments():
    cfg = deploy.parse_config({"_comment": "x", "host": " 192.0.2.5 ", "services": ["loadlens"]}, Path("c.json"))
    assert cfg.host == "192.0.2.5" and cfg.target == "192.0.2.5"
    assert cfg.branch == "main" and cfg.health_url == "http://127.0.0.1:5000/healthz" and cfg.health_wait_sec == 90
    assert deploy.parse_config({"host": "h", "user": "ops"}, Path("c.json")).target == "ops@h"


@pytest.mark.parametrize(
    "raw, message",
    [
        ({}, '"host" is required'),
        ({"host": "h", "hots": 1}, "unknown keys hots"),
        ({"host": "h", "services": "loadlens"}, "list of non-empty strings"),
        ({"host": "h", "services": ["loadlens; reboot"]}, "not a systemd unit name"),
        ({"host": "h", "branch": "-x"}, "not a branch name"),
        ({"host": "h", "port": "22"}, '"port"'),
        ({"host": "h", "health_wait_sec": 0}, '"health_wait_sec"'),
        ({"host": "h", "app_dir": 5}, '"app_dir" must be a string'),
    ],
)
def test_config_errors(raw, message):
    with pytest.raises(deploy.DeployError, match=re.escape(message)):
        deploy.parse_config(raw, Path("c.json"))


def test_missing_config_names_the_example(tmp_path):
    with pytest.raises(deploy.DeployError, match="deploy.example.json"):
        deploy.load_config(tmp_path / "deploy.local.json")


def test_example_config_is_valid():
    cfg = deploy.load_config(ROOT / "tools" / "deploy.example.json")
    assert cfg.app_dir and cfg.services


def test_remote_command_carries_the_script_with_lf_and_quoted_args():
    cfg = deploy.Config(host="h", app_dir="/opt/load lens's", services=UNITS)
    command = deploy.remote_command(cfg, "deploy", "abc", restart=True, script="echo 1\r\necho 2\r\n")
    encoded = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d \| gunzip", command).group(1)
    assert gzip.decompress(base64.b64decode(encoded)) == b"echo 1\necho 2\n"
    assert """'/opt/load lens'"'"'s'""" in command and "'loadlens loadlens-worker'" in command
    assert command.endswith(" 1")


def test_remote_command_stays_short():
    # Windows passes an MSYS program about 8K characters of arguments; keep the real command well below.
    cfg = deploy.Config(host="h", app_dir="/opt/loadlens", services=UNITS)
    assert len(deploy.remote_command(cfg, "deploy", "a" * 40)) < 7000


def test_ssh_argv():
    cfg = deploy.Config(host="h", user="ops", port=2222, ssh_options=("-J", "jump"))
    assert deploy.ssh_argv(cfg, "CMD", tty=True) == ["ssh", "-t", "-p", "2222", "-J", "jump", "ops@h", "CMD"]
    assert deploy.ssh_argv(deploy.Config(host="h"), "CMD", tty=False) == ["ssh", "h", "CMD"]


def test_remote_script_is_lf_only():
    assert b"\r" not in (ROOT / "tools" / "deploy_remote.sh").read_bytes()


# --- the remote script against throwaway repositories ----------------------------------------------------------

STUBS = {
    "systemctl": """#!/usr/bin/env bash
echo "systemctl $*" >> "$STUB_LOG"
case "$1" in
  show)
    if [[ " $* " == *" --value "* ]]; then
      printf '{ path=%s ; argv[]=%s app.py ; ignore_errors=no }\\n' "$FAKE_PY" "$FAKE_PY"
    else
      printf 'ExecStart={ path=%s ; argv[]=%s app.py --redis redis://:secret@r:6379 ; ignore_errors=no }\\n' "$FAKE_PY" "$FAKE_PY"
      printf 'WorkingDirectory=/opt/loadlens\\n'
    fi ;;
  is-active) [ "${UNITS_INACTIVE:-0}" = 1 ] && { echo failed; exit 3; }; [[ " $* " == *" --quiet "* ]] || echo active ;;
  list-unit-files) printf 'loadlens.service enabled enabled\\ngetty@.service static -\\n' ;;
esac
exit 0
""",
    "journalctl": '#!/usr/bin/env bash\necho "journalctl $*" >> "$STUB_LOG"\necho "journal line"\n',
    "sudo": """#!/usr/bin/env bash
echo "sudo $*" >> "$STUB_LOG"
while [[ "${1:-}" == -* ]]; do case "$1" in -u) shift 2 ;; *) shift ;; esac; done
exec "$@"
""",
    "curl": '#!/usr/bin/env bash\necho "curl $*" >> "$STUB_LOG"\nexit "${CURL_EXIT:-0}"\n',
}


class Stand:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.origin, self.work, self.app = tmp / "origin.git", tmp / "work", tmp / "app"
        self.log = tmp / "calls.log"
        bin_dir, venv_bin = tmp / "bin", tmp / "venv" / "bin"
        self.bin_path = _bash_path(bin_dir)
        bin_dir.mkdir()
        venv_bin.mkdir(parents=True)
        for name, text in STUBS.items():
            (bin_dir / name).write_bytes(text.encode())
        self.python = venv_bin / "python"
        self.python.write_bytes(b'#!/usr/bin/env bash\necho "python $*" >> "$STUB_LOG"\nexit "${PIP_EXIT:-0}"\n')
        for path in [*bin_dir.iterdir(), self.python]:
            path.chmod(0o755)
        self.env = {
            **os.environ,
            "HOME": str(tmp),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "STUB_LOG": self.log.as_posix(),
            "FAKE_PY": self.python.as_posix(),
            "LOADLENS_DEPLOY_POLL_SEC": "0",
        }
        for key in ("GIT_DIR", "GIT_WORK_TREE", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            self.env.pop(key, None)
        self.git(tmp, "init", "--quiet", "--bare", str(self.origin))
        self.git(self.origin, "symbolic-ref", "HEAD", "refs/heads/main")
        self.git(tmp, "clone", "--quiet", str(self.origin), str(self.work))
        self.git(self.work, "symbolic-ref", "HEAD", "refs/heads/main")
        self.first = self.commit({"app.py": "v1\n", "requirements.txt": "flask\n"}, push=True)
        self.git(tmp, "clone", "--quiet", str(self.origin), str(self.app))

    def git(self, repo: Path, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", "-C", str(repo), *args],
            capture_output=True, text=True, env=self.env,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def commit(self, files: dict[str, str], push: bool = True) -> str:
        for name, text in files.items():
            (self.work / name).write_bytes(text.encode())
        self.git(self.work, "add", "-A")
        self.git(self.work, "commit", "--quiet", "-m", "change " + ", ".join(files))
        if push:
            self.git(self.work, "push", "--quiet", "origin", "main")
        return self.git(self.work, "rev-parse", "HEAD")

    def head(self) -> str:
        return self.git(self.app, "rev-parse", "HEAD")

    def run(self, mode: str, ref: str = "", restart: bool = False, units=UNITS, app_dir=None, **env):
        cfg = deploy.Config(
            host="unused", app_dir=self.app.as_posix() if app_dir is None else app_dir, services=tuple(units),
            health_wait_sec=1,
        )
        if self.log.exists():
            self.log.unlink()
        # The stubs go first explicitly: Git Bash puts its own directories (with a real curl) before the inherited PATH.
        command = f'export PATH="{self.bin_path}:$PATH"; ' + deploy.remote_command(cfg, mode, ref, restart)
        proc = subprocess.run(
            [BASH, "-c", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env={**self.env, **env}, timeout=120,
        )
        calls = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return proc.returncode, proc.stdout + proc.stderr, calls


@pytest.fixture
def stand(tmp_path):
    return Stand(tmp_path)


def _restarted(calls):
    return any(call == "systemctl restart loadlens loadlens-worker" for call in calls)


@needs_bash
def test_deploy_moves_the_checkout_and_restarts(stand):
    new = stand.commit({"app.py": "v2\n"})
    code, out, calls = stand.run("deploy", new)
    assert code == 0, out
    assert stand.head() == new
    assert stand.git(stand.app, "rev-parse", "refs/loadlens/previous") == stand.first
    assert _restarted(calls) and any(call.startswith("curl ") for call in calls)
    assert not any(call.startswith("python ") for call in calls), "requirements.txt did not change"
    assert "change app.py" in out and "done:" in out


@needs_bash
def test_changed_requirements_run_pip_of_the_venv(stand):
    new = stand.commit({"requirements.txt": "flask\nrequests\n"})
    code, out, calls = stand.run("deploy", new)
    assert code == 0, out
    pip = [call for call in calls if call.startswith("python ")]
    assert len(pip) == 1 and "-m pip install" in pip[0] and pip[0].endswith("/requirements.txt")


@needs_bash
def test_failed_pip_puts_the_old_commit_back(stand):
    new = stand.commit({"requirements.txt": "broken\n"})
    code, out, calls = stand.run("deploy", new, PIP_EXIT="1")
    assert code != 0 and "pip install failed" in out
    assert stand.head() == stand.first and not _restarted(calls)


@needs_bash
def test_modified_tracked_file_blocks_the_deploy(stand):
    new = stand.commit({"app.py": "v2\n"})
    (stand.app / "app.py").write_bytes(b"edited on the server\n")
    (stand.app / "settings.py").write_bytes(b"untracked config\n")
    code, out, calls = stand.run("deploy", new)
    assert code != 0 and "modified tracked files" in out and "app.py" in out
    assert stand.head() == stand.first and not _restarted(calls)


@needs_bash
def test_untracked_config_files_survive(stand):
    (stand.app / "settings.py").write_bytes(b"secret = 1\n")
    new = stand.commit({"app.py": "v2\n"})
    code, out, _calls = stand.run("deploy", new)
    assert code == 0, out
    assert (stand.app / "settings.py").read_bytes() == b"secret = 1\n"


@needs_bash
def test_unpushed_commit_is_refused(stand):
    local_only = stand.commit({"app.py": "v2\n"}, push=False)
    code, out, calls = stand.run("deploy", local_only)
    assert code != 0 and "push it first" in out
    assert stand.head() == stand.first and not _restarted(calls)


@needs_bash
def test_commits_of_the_server_block_the_deploy(stand):
    (stand.app / "local.txt").write_bytes(b"x\n")
    stand.git(stand.app, "add", "local.txt")
    stand.git(stand.app, "commit", "--quiet", "-m", "server-only")
    new = stand.commit({"app.py": "v2\n"})
    code, out, calls = stand.run("deploy", new)
    assert code != 0 and "has commits that origin/main has not" in out and not _restarted(calls)


@needs_bash
def test_failed_health_check_prints_logs_and_the_rollback_hint(stand):
    new = stand.commit({"app.py": "v2\n"})
    code, out, calls = stand.run("deploy", new, CURL_EXIT="7")
    assert code != 0 and "tools/deploy.py rollback" in out and "journal line" in out
    assert _restarted(calls) and any(call.startswith("journalctl -u loadlens ") for call in calls)


@needs_bash
def test_inactive_unit_fails_the_deploy(stand):
    new = stand.commit({"app.py": "v2\n"})
    code, out, _calls = stand.run("deploy", new, UNITS_INACTIVE="1")
    assert code != 0 and "did not come up" in out


@needs_bash
def test_rollback_returns_to_the_previous_commit(stand):
    new = stand.commit({"app.py": "v2\n"})
    assert stand.run("deploy", new)[0] == 0
    code, out, calls = stand.run("rollback")
    assert code == 0, out
    assert stand.head() == stand.first and _restarted(calls)
    assert "fetching" not in out, "a rollback works without the repository"
    # The commit left behind becomes the rollback target: a second rollback goes forward again.
    assert stand.git(stand.app, "rev-parse", "refs/loadlens/previous") == new


@needs_bash
def test_rollback_without_a_previous_deploy_fails(stand):
    code, out, calls = stand.run("rollback")
    assert code != 0 and "no previous deploy" in out and not _restarted(calls)


@needs_bash
def test_same_commit_restarts_only_on_request(stand):
    code, out, calls = stand.run("deploy", stand.first)
    assert code == 0 and "nothing to do" in out and not _restarted(calls)
    code, out, calls = stand.run("deploy", stand.first, restart=True)
    assert code == 0 and _restarted(calls)


@needs_bash
def test_check_lists_services_and_changes_nothing(stand):
    new = stand.commit({"app.py": "v2\n"})
    code, out, calls = stand.run("check", units=(), app_dir="")
    assert code == 0, out
    assert "loadlens.service" in out and "WorkingDirectory=/opt/loadlens" in out and "getty@" not in out
    assert "secret" not in out and "redis://***@r:6379" in out
    code, out, calls = stand.run("check")
    assert code == 0, out
    assert "branch  main" in out and "health http://127.0.0.1:5000/healthz: ok" in out
    assert f"python  {stand.python.as_posix()}" in out
    assert not _restarted(calls) and not any(call.startswith("systemctl restart") for call in calls)
    assert stand.head() == stand.first != new


@needs_bash
def test_resolve_target_needs_a_pushed_commit(stand, monkeypatch, capsys):
    for key in ("HOME", "GIT_CONFIG_NOSYSTEM", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, stand.env[key])
    cfg = deploy.Config(host="h", app_dir="/opt/x", services=UNITS)
    sha, description = deploy.resolve_target(cfg, "", push=False, repo=stand.work)
    assert sha == stand.first and description.endswith("change app.py, requirements.txt")
    stand.commit({"app.py": "v2\n"}, push=False)
    with pytest.raises(deploy.DeployError, match="not on origin/main"):
        deploy.resolve_target(cfg, "", push=False, repo=stand.work)
    (stand.work / "app.py").write_bytes(b"dirty\n")
    sha, _ = deploy.resolve_target(cfg, "", push=True, repo=stand.work)
    assert stand.git(stand.origin, "rev-parse", "main") == sha
    assert "uncommitted changes are not deployed" in capsys.readouterr().err
