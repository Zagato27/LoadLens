# SPDX-License-Identifier: Apache-2.0
"""Minimal SSH server of the dev-stack load generator. Dev stack only.

Networks with a filtering proxy block the Debian mirror, so the image cannot install OpenSSH. This server
does what LoadLens needs from sshd and nothing more: public key login of one user, ``exec`` requests run
as that user with ``/bin/sh -c``, stdin/stdout/stderr and the exit status passed through. No shells, no
port forwarding, no SFTP. The client side of LoadLens (paramiko, host key pinning, the runner scripts)
is the same as against OpenSSH.
"""

from __future__ import annotations

import base64
import logging
import os
import pwd
import socket
import subprocess
import threading

import paramiko

PORT = int(os.environ.get("SSH_PORT", "2222"))
USER = os.environ.get("SSH_USER", "loadlens")
HOST_KEY_PATH = os.environ.get("SSH_HOST_KEY", "/etc/devsshd/host_ecdsa")
AUTHORIZED_KEYS = os.environ.get("SSH_AUTHORIZED_KEYS", "/keys/authorized_keys")
CHUNK = 32768

logging.basicConfig(level=logging.INFO, format="devsshd %(levelname)s %(message)s")
log = logging.getLogger("devsshd")


def host_key() -> paramiko.PKey:
    if not os.path.exists(HOST_KEY_PATH):
        os.makedirs(os.path.dirname(HOST_KEY_PATH), exist_ok=True)
        paramiko.ECDSAKey.generate().write_private_key_file(HOST_KEY_PATH)
    return paramiko.ECDSAKey(filename=HOST_KEY_PATH)


def authorized() -> set[bytes]:
    keys: set[bytes] = set()
    with open(AUTHORIZED_KEYS, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("#"):
                keys.add(paramiko.PKey.from_type_string(parts[0], base64.b64decode(parts[1])).asbytes())
    return keys


def login_env(account: pwd.struct_passwd) -> dict[str, str]:
    return {
        "HOME": account.pw_dir,
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
        "SHELL": "/bin/sh",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }


def run_command(channel: paramiko.Channel, command: str) -> None:
    account = pwd.getpwnam(USER)
    proc = subprocess.Popen(
        ["/bin/sh", "-c", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=account.pw_dir,
        env=login_env(account),
        user=account.pw_uid,
        group=account.pw_gid,
        extra_groups=[],
    )

    def feed() -> None:
        try:
            while True:
                data = channel.recv(CHUNK)
                if not data:
                    break
                proc.stdin.write(data)
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def drain(stream, send) -> None:
        fd = stream.fileno()
        while True:
            data = os.read(fd, CHUNK)
            if not data:
                break
            send(data)

    threads = [
        threading.Thread(target=feed, daemon=True),
        threading.Thread(target=drain, args=(proc.stdout, channel.sendall), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, channel.sendall_stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    code = proc.wait()
    threads[1].join()
    threads[2].join()
    channel.send_exit_status(code)
    channel.shutdown_write()
    channel.close()


class Handler(paramiko.ServerInterface):
    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def get_allowed_auths(self, username):
        return "publickey"

    def check_auth_publickey(self, username, key):
        if username == USER and key.asbytes() in authorized():
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_exec_request(self, channel, command):
        def run() -> None:
            try:
                run_command(channel, command.decode("utf-8", "replace"))
            except Exception:
                log.exception("exec failed")
                channel.close()

        threading.Thread(target=run, daemon=True).start()
        return True


def serve(client: socket.socket, key: paramiko.PKey) -> None:
    transport = paramiko.Transport(client)
    transport.add_server_key(key)
    # A Channel closes itself when garbage collected: keep accepted ones until they are done.
    open_channels: list[paramiko.Channel] = []
    try:
        transport.start_server(server=Handler())
        while transport.is_active():
            channel = transport.accept(1)
            open_channels = [ch for ch in open_channels if not ch.closed]
            if channel is not None:
                open_channels.append(channel)
    except Exception as exc:  # a broken client must not stop the server
        log.info("connection closed: %s", exc)
    finally:
        transport.close()


def main() -> None:
    key = host_key()
    log.info("host key %s %s", key.get_name(), key.fingerprint)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(32)
    log.info("listening on %s for user %s", PORT, USER)
    while True:
        client, _ = server.accept()
        threading.Thread(target=serve, args=(client, key), daemon=True).start()


if __name__ == "__main__":
    main()
