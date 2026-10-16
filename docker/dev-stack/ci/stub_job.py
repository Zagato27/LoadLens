"""The "pipeline" of the GitLab and Jenkins stubs: one k6 container of the dev stack with the run's values.

Like a real load test pipeline it prepares for a while, runs k6 with the values it was given, may report
``started`` and ``finished`` to LoadLens and may tag the metrics with LOADLENS_RUN_TAG, then cleans up.

Values the job understands (pipeline variables, inputs or Jenkins parameters, case-insensitive):
VUS / USERS, DURATION, PROFILE (smoke 2, load 5, stress 10 RPS) or RPS, TAG_METRICS (true: --tag
testid=$LOADLENS_RUN_TAG), SEND_EVENTS (true: started/finished to LOADLENS_EVENTS_URL), SETUP_SEC,
TEARDOWN_SEC. Scripts come from the dev-stack volume ``loadlens-dev-scripts``.
"""

import subprocess

IMAGE = "grafana/k6:0.54.0"
NETWORK = "loadlens-dev-stack_default"
REMOTE_WRITE = "http://prometheus:9090/api/v1/write"
RPS_BY_PROFILE = {"smoke": "2", "load": "5", "stress": "10"}

SCRIPT = r"""
echo "LoadLens stub job: preparing for ${SETUP_SEC}s"
sleep "${SETUP_SEC}"
post() {
  if [ -n "$LOADLENS_EVENTS_URL" ] && [ "$SEND_EVENTS" = "true" ]; then
    wget -q -T 10 -O /dev/null --header="Content-Type: application/json" --header="X-LoadLens-Token: $LOADLENS_TOKEN" \
      --post-data="$1" "$LOADLENS_EVENTS_URL" || echo "event was not delivered"
  fi
}
post '{"event":"started"}'
TAG=""
if [ "$TAG_METRICS" = "true" ] && [ -n "$LOADLENS_RUN_TAG" ]; then TAG="--tag testid=$LOADLENS_RUN_TAG"; fi
echo "k6: vus=$VUS duration=$DURATION rps=$RPS tag=${TAG:-none}"
k6 run --vus "$VUS" --duration "$DURATION" --rps "$RPS" --out experimental-prometheus-rw $TAG /scripts/k6/load.js
code=$?
post "{\"event\":\"finished\",\"exit_code\":$code}"
echo "LoadLens stub job: cleaning up for ${TEARDOWN_SEC}s"
sleep "${TEARDOWN_SEC}"
exit $code
"""


def job_env(values: dict) -> dict:
    """Environment of the k6 container from the values of the run (later keys win)."""
    upper = {str(key).upper(): "" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
             for key, value in values.items()}
    profile = upper.get("PROFILE", "load")
    return {
        "VUS": upper.get("VUS") or upper.get("USERS") or "2",
        "DURATION": upper.get("DURATION") or "60s",
        "RPS": upper.get("RPS") or RPS_BY_PROFILE.get(profile, "5"),
        "TAG_METRICS": upper.get("TAG_METRICS", "false"),
        "SEND_EVENTS": upper.get("SEND_EVENTS", "false"),
        "SETUP_SEC": upper.get("SETUP_SEC") or "20",
        "TEARDOWN_SEC": upper.get("TEARDOWN_SEC") or "15",
        "LOADLENS_RUN_TAG": upper.get("LOADLENS_RUN_TAG", ""),
        "LOADLENS_EVENTS_URL": upper.get("LOADLENS_EVENTS_URL", ""),
        "LOADLENS_TOKEN": upper.get("LOADLENS_TOKEN", ""),
        "K6_PROMETHEUS_RW_SERVER_URL": REMOTE_WRITE,
        "K6_PROMETHEUS_RW_TREND_STATS": "p(95),p(99),avg",
    }


def start(name: str, values: dict) -> subprocess.Popen:
    env_args = []
    for key, value in job_env(values).items():
        env_args += ["-e", f"{key}={value}"]
    cmd = ["docker", "run", "--name", name, "--network", NETWORK, *env_args, "-v", "loadlens-dev-scripts:/scripts:ro",
           "--entrypoint", "sh", IMAGE, "-c", SCRIPT]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def logs(name: str) -> str:
    out = subprocess.run(["docker", "logs", name], capture_output=True)
    return (out.stdout + out.stderr).decode("utf-8", "replace")


def kill(name: str) -> None:
    subprocess.run(["docker", "kill", name], capture_output=True)


def remove(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
