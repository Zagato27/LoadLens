# SPDX-License-Identifier: Apache-2.0
"""Update a LoadLens source checkout that systemd units run on another host: one command, one SSH session.

    python tools/deploy.py check             # read-only: checkout, services, venv Python, health URL, sudo
    python tools/deploy.py                   # deploy the local HEAD (it must be on origin/<branch>)
    python tools/deploy.py --push            # git push the local <branch> first, then deploy its HEAD
    python tools/deploy.py --ref 1a2b3c4     # deploy another commit of origin/<branch>
    python tools/deploy.py --restart         # restart the services even when the server is at that commit
    python tools/deploy.py rollback          # back to the commit the last deploy replaced

The server fetches from its own ``origin``, so it needs access to the repository; this tool checks that the commit
is pushed and runs tools/deploy_remote.sh on the server. Password prompts of ssh and sudo appear in this terminal.
The remote script refuses a checkout with modified tracked files, another branch or commits of its own, installs
requirements.txt only when it changed, restarts the units and waits until they stay active and the health URL
answers; otherwise it prints their logs. settings.py, the runtime JSON files and the venv are not tracked by git
and stay as they are.

Configuration: tools/deploy.local.json (git-ignored; start from tools/deploy.example.json) or --config PATH.
    host             SSH host name or address (required)
    user, port       SSH login and port (optional: ssh defaults and ~/.ssh/config apply)
    ssh_options      more ssh arguments, e.g. ["-J", "jump.example.com"] or ["-i", "~/.ssh/deploy_key"]
    app_dir          the checkout on the server (required except for check)
    services         systemd units to restart, e.g. ["loadlens", "loadlens-worker"] (required except for check)
    branch           the deployed branch ("main")
    health_url       requested on the server after the restart ("http://127.0.0.1:5000/healthz"; "" skips it)
    health_wait_sec  how long to wait for the units and the health URL (90)
    python           the venv Python for pip (default: found from ExecStart of the units)
    run_as           owner of the checkout when it is not the SSH user: git and pip run as that user via sudo
Keys that start with "_" are comments.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_SCRIPT = Path(__file__).with_name("deploy_remote.sh")
DEFAULT_CONFIG = Path(__file__).with_name("deploy.local.json")
MODES = ("deploy", "check", "rollback")

_UNIT = re.compile(r"^[A-Za-z0-9@._:\\-]+$")
_BRANCH = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/-]*$")


class DeployError(Exception):
    """A configuration or preflight problem, printed without a traceback."""


@dataclass(frozen=True)
class Config:
    host: str
    user: str = ""
    port: int | None = None
    ssh_options: tuple[str, ...] = ()
    app_dir: str = ""
    services: tuple[str, ...] = ()
    branch: str = "main"
    health_url: str = "http://127.0.0.1:5000/healthz"
    health_wait_sec: int = 90
    python: str = ""
    run_as: str = ""

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def _str_list(raw: dict, key: str, path: Path) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise DeployError(f"{path}: \"{key}\" must be a list of non-empty strings")
    return tuple(value)


def parse_config(raw: object, path: Path) -> Config:
    """Validated configuration from the parsed JSON of ``path`` (the path only names the file in errors)."""
    if not isinstance(raw, dict):
        raise DeployError(f"{path}: expected a JSON object")
    raw = {key: value for key, value in raw.items() if not key.startswith("_")}
    unknown = sorted(set(raw) - {f.name for f in fields(Config)})
    if unknown:
        raise DeployError(f"{path}: unknown keys {', '.join(unknown)}")
    for key in ("host", "user", "app_dir", "branch", "health_url", "python", "run_as"):
        if key in raw and not isinstance(raw[key], str):
            raise DeployError(f"{path}: \"{key}\" must be a string")
    host = raw.get("host", "").strip()
    if not host:
        raise DeployError(f"{path}: \"host\" is required")
    services = _str_list(raw, "services", path)
    bad = [unit for unit in services if not _UNIT.match(unit)]
    if bad:
        raise DeployError(f"{path}: not a systemd unit name: {', '.join(bad)}")
    branch = raw.get("branch", "main")
    if not _BRANCH.match(branch):
        raise DeployError(f"{path}: not a branch name: {branch!r}")
    port = raw.get("port")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536):
        raise DeployError(f"{path}: \"port\" must be a number from 1 to 65535")
    wait = raw.get("health_wait_sec", 90)
    if isinstance(wait, bool) or not isinstance(wait, int) or wait < 1:
        raise DeployError(f"{path}: \"health_wait_sec\" must be a positive number")
    return Config(
        host=host,
        user=raw.get("user", "").strip(),
        port=port,
        ssh_options=_str_list(raw, "ssh_options", path),
        app_dir=raw.get("app_dir", "").strip(),
        services=services,
        branch=branch,
        health_url=raw.get("health_url", Config.health_url).strip(),
        health_wait_sec=wait,
        python=raw.get("python", "").strip(),
        run_as=raw.get("run_as", "").strip(),
    )


def load_config(path: Path) -> Config:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise DeployError(
            f"no configuration {path}: copy tools/deploy.example.json to tools/deploy.local.json and fill it in"
        ) from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeployError(f"{path}: invalid JSON: {exc}") from None
    return parse_config(raw, path)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and proc.returncode != 0:
        raise DeployError(f"git {' '.join(args)}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def resolve_target(cfg: Config, ref: str, push: bool, repo: Path = ROOT) -> tuple[str, str]:
    """Full hash and one-line description of the commit to deploy; it must be on origin/<branch>.

    ``push`` pushes the local <branch> first when its HEAD is not there yet (only without ``ref`` and only from a
    checkout of that branch). Uncommitted changes are reported: they are not deployed.
    """
    proc = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref or 'HEAD'}^{{commit}}", check=False)
    if proc.returncode != 0:
        raise DeployError(f"unknown commit {ref or 'HEAD'}")
    sha = proc.stdout.strip()
    if not ref:
        changed = _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.splitlines()
        if changed:
            print(f"WARNING: {len(changed)} files with uncommitted changes are not deployed", file=sys.stderr)
    upstream = f"origin/{cfg.branch}"
    _git(repo, "fetch", "--quiet", "origin", cfg.branch)
    if _git(repo, "merge-base", "--is-ancestor", sha, upstream, check=False).returncode != 0:
        if not push:
            raise DeployError(f"{sha[:7]} is not on {upstream}: push it first or add --push")
        current = _git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.strip()
        if ref or current != cfg.branch:
            raise DeployError(f"--push pushes the local branch {cfg.branch} only: check it out and run again")
        print(f"==> git push origin {cfg.branch}")
        if subprocess.run(["git", "-C", str(repo), "push", "origin", cfg.branch]).returncode != 0:
            raise DeployError("git push failed")
    return sha, _git(repo, "log", "-1", "--format=%h %s", sha).stdout.strip()


def remote_command(cfg: Config, mode: str, ref: str = "", restart: bool = False, script: str | None = None) -> str:
    """The command line the server's login shell runs: the remote script travels inside it, gzip + base64.

    The script goes in the command, not on stdin, so that ssh -t leaves the terminal to sudo's password prompt.
    gzip keeps the line short: Windows hands an MSYS program (Git Bash, used by the tests) at most about 8K
    characters of arguments.
    """
    text = REMOTE_SCRIPT.read_text(encoding="utf-8") if script is None else script
    packed = gzip.compress(text.replace("\r\n", "\n").encode("utf-8"), mtime=0)
    encoded = base64.b64encode(packed).decode("ascii")
    args = [
        mode, cfg.app_dir, cfg.branch, ref, " ".join(cfg.services), cfg.health_url, str(cfg.health_wait_sec),
        cfg.python, cfg.run_as, "1" if restart else "0",
    ]
    return (
        f'bash -c "$(echo {encoded} | base64 -d | gunzip)" loadlens-deploy '
        + " ".join(shlex.quote(arg) for arg in args)
    )


def ssh_argv(cfg: Config, command: str, tty: bool) -> list[str]:
    argv = ["ssh"]
    if tty:
        argv.append("-t")  # sudo asks for its password on the terminal
    if cfg.port:
        argv += ["-p", str(cfg.port)]
    argv += list(cfg.ssh_options)
    return argv + [cfg.target, command]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/deploy.py",
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring or docs/deployment.md for the configuration keys.",
    )
    parser.add_argument("mode", nargs="?", choices=MODES, default="deploy")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="default: tools/deploy.local.json")
    parser.add_argument("--ref", default="", help="deploy this commit of origin/<branch> instead of HEAD")
    parser.add_argument("--push", action="store_true", help="git push the local branch first")
    parser.add_argument("--restart", action="store_true", help="restart even when the server is at that commit")
    args = parser.parse_args(argv)
    if args.mode != "deploy" and (args.ref or args.push):
        parser.error("--ref and --push apply to deploy only")

    try:
        cfg = load_config(args.config)
        if args.mode != "check" and (not cfg.app_dir or not cfg.services):
            raise DeployError("\"app_dir\" and \"services\" are required: python tools/deploy.py check helps to find them")
        ref = ""
        if args.mode == "deploy":
            ref, description = resolve_target(cfg, args.ref, args.push)
            print(f"==> deploying {description} to {cfg.target}:{cfg.app_dir} ({', '.join(cfg.services)})")
        elif args.mode == "rollback":
            print(f"==> rolling back {cfg.target}:{cfg.app_dir} to the commit before the last deploy")
        command = remote_command(cfg, args.mode, ref, args.restart)
    except DeployError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        code = subprocess.run(ssh_argv(cfg, command, tty=sys.stdin.isatty())).returncode
    except FileNotFoundError:
        print("ERROR: ssh is not installed or not on PATH", file=sys.stderr)
        return 2
    if code == 255:
        print(f"ERROR: ssh could not connect to {cfg.target} (host, user, port, key or password)", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
