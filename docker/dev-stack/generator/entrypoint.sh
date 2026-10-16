#!/bin/sh
# Checks the public key LoadLens logs in with and starts the dev SSH server.
set -e
if [ ! -s /keys/authorized_keys ]; then
  echo "generator: /keys/authorized_keys is missing, see docker/dev-stack/README.md" >&2
  exit 1
fi
# The Gatling bundle compiles simulations and writes reports inside its own directory.
if [ -d /opt/gatling ]; then chown -R loadlens:loadlens /opt/gatling 2>/dev/null || true; fi
exec /opt/venv/bin/python /opt/devsshd/devsshd.py
