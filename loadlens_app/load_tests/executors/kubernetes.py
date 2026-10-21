# SPDX-License-Identifier: Apache-2.0
"""Kubernetes generators: one Job per instance through the Kubernetes API.

The Job runs the tool image with the rendered command, ``backoffLimit: 0`` and ``activeDeadlineSeconds``
equal to the run limit, so the cluster stops the tool even when LoadLens is down; ``ttlSecondsAfterFinished``
removes it afterwards. Environment variables of the instance go to a Secret owned by the Job (created right
after it), not into the Job spec. Scripts come from a ConfigMap or a PersistentVolumeClaim mounted at the scripts directory, or are
baked into the image.

Kubernetes cannot deliver an arbitrary signal: stopping deletes the pod with the tool's grace period
(SIGTERM, then SIGKILL), killing deletes the Job at once.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from datetime import datetime
from typing import Any, Mapping, Optional

from ..generators import Generator
from ..messages import msg, t
from ..tools import ToolTemplate
from .base import LOG_TAIL_BYTES, CheckReport, GeneratorError, LaunchPlan, LaunchResult, Probe
from .http import ApiClient, ApiError

IN_CLUSTER_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
IN_CLUSTER_URL = "https://kubernetes.default.svc"
MANAGED_BY = "loadlens"
CONTAINER = "tool"
# Waiting reasons that will not resolve without a change (image name, pull secret, missing Secret).
STUCK_REASONS = frozenset({"ErrImagePull", "ImagePullBackOff", "InvalidImageName", "CreateContainerConfigError", "CreateContainerError"})
DNS_SANITIZE_RE = re.compile(r"[^a-z0-9-]+")
REQUIRED_VERBS = (
    ("create", "batch", "jobs"),
    ("get", "batch", "jobs"),
    ("delete", "batch", "jobs"),
    ("create", "", "secrets"),
    ("delete", "", "secrets"),
    ("list", "", "pods"),
    ("deletecollection", "", "pods"),
    ("get", "", "pods/log"),
)


def job_name(run_id: str, index: int, tag: str) -> str:
    clean = DNS_SANITIZE_RE.sub("-", tag.lower()).strip("-")
    name = f"ll-{run_id[:10]}-{index + 1:02d}-{clean}" if clean else f"ll-{run_id[:10]}-{index + 1:02d}"
    return name[:52].rstrip("-")


def _k8s_time(value: Any) -> Optional[float]:
    text = str(value or "")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def kubernetes_api(generator: Generator) -> ApiClient:
    if generator.kubernetes is None:
        raise GeneratorError("lt.k8s.not_k8s", generator=generator.title)
    cfg = generator.kubernetes
    if cfg.in_cluster:
        try:
            with open(os.path.join(IN_CLUSTER_DIR, "token"), encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError as exc:
            raise GeneratorError("lt.k8s.not_in_cluster") from exc
        return ApiClient(IN_CLUSTER_URL, name="Kubernetes API", token=token, ca_file=os.path.join(IN_CLUSTER_DIR, "ca.crt"))
    return ApiClient(cfg.api_url, name="Kubernetes API", verify=cfg.verify, ca_cert=cfg.ca_cert, token=cfg.token)


class KubernetesExecutor:
    enforces_deadline = True
    supports_signals = False
    supports_control = False

    def __init__(self, generator: Generator, api: ApiClient) -> None:
        assert generator.kubernetes is not None
        self.generator = generator
        self.cfg = generator.kubernetes
        self.api = api

    @property
    def ns(self) -> str:
        return f"/api/v1/namespaces/{self.cfg.namespace}"

    @property
    def batch(self) -> str:
        return f"/apis/batch/v1/namespaces/{self.cfg.namespace}"

    def _labels(self, plan: LaunchPlan) -> dict[str, str]:
        return {
            "app.kubernetes.io/managed-by": MANAGED_BY,
            "loadlens.io/run-id": plan.run_id[:63],
            "loadlens.io/instance": str(plan.instance_index),
            "loadlens.io/tool": re.sub(r"[^A-Za-z0-9_.-]", "-", plan.tool_id)[:63],
        }

    def job_manifest(self, plan: LaunchPlan, secret_name: str = "") -> dict[str, Any]:
        container: dict[str, Any] = {
            "name": CONTAINER,
            "image": plan.image,
            # Kubernetes expands $(NAME) in the command from the container environment; $$ keeps a literal $,
            # so parameters cannot pull values of the generator's secret variables into the visible spec.
            "command": [arg.replace("$", "$$") for arg in plan.command.argv],
            "workingDir": plan.command.cwd,
            "securityContext": {"allowPrivilegeEscalation": False},
        }
        if secret_name:
            container["envFrom"] = [{"secretRef": {"name": secret_name}}]
        resources: dict[str, dict[str, str]] = {}
        if self.cfg.cpu:
            resources.setdefault("requests", {})["cpu"] = self.cfg.cpu
            resources.setdefault("limits", {})["cpu"] = self.cfg.cpu
        if self.cfg.memory:
            resources.setdefault("requests", {})["memory"] = self.cfg.memory
            resources.setdefault("limits", {})["memory"] = self.cfg.memory
        if resources:
            container["resources"] = resources
        pod: dict[str, Any] = {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": int(plan.grace_seconds),
            "automountServiceAccountToken": False,
            "containers": [container],
        }
        volume: Optional[dict[str, Any]] = None
        if self.cfg.scripts_config_map:
            volume = {"name": "scripts", "configMap": {"name": self.cfg.scripts_config_map}}
        elif self.cfg.scripts_pvc:
            volume = {"name": "scripts", "persistentVolumeClaim": {"claimName": self.cfg.scripts_pvc, "readOnly": True}}
        if volume and self.generator.scripts_dir:
            pod["volumes"] = [volume]
            container["volumeMounts"] = [{"name": "scripts", "mountPath": self.generator.scripts_dir, "readOnly": True}]
        if self.cfg.node_selector:
            pod["nodeSelector"] = dict(self.cfg.node_selector)
        if self.cfg.service_account:
            pod["serviceAccountName"] = self.cfg.service_account
        if self.cfg.image_pull_secret:
            pod["imagePullSecrets"] = [{"name": self.cfg.image_pull_secret}]
        labels = self._labels(plan)
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": plan.ref, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": int(plan.max_seconds),
                "ttlSecondsAfterFinished": int(self.cfg.ttl_seconds),
                "template": {"metadata": {"labels": labels}, "spec": pod},
            },
        }

    def launch(self, plan: LaunchPlan) -> LaunchResult:
        """Creates the Job, then its Secret owned by it, so no Secret is left behind without a Job.

        The pod waits for the Secret a moment (kubelet retries a missing Secret); if the Secret cannot be
        created, the Job is deleted and the launch fails.
        """
        if not plan.image:
            raise GeneratorError("lt.executor.no_image", tool=plan.tool_id)
        secret_name = f"{plan.ref}-env" if plan.command.env else ""
        try:
            job = self.api.json("POST", f"{self.batch}/jobs", json=self.job_manifest(plan, secret_name), ok=(201,))
        except ApiError as exc:
            if exc.status == 409:
                return LaunchResult(ref=plan.ref, already_running=True)
            raise
        if secret_name:
            try:
                self._create_secret(plan, secret_name, str((job.get("metadata") or {}).get("uid") or ""))
            except Exception:
                try:
                    self.kill(plan.ref)
                except Exception:
                    pass
                raise
        return LaunchResult(ref=plan.ref)

    def _create_secret(self, plan: LaunchPlan, secret_name: str, job_uid: str) -> None:
        metadata: dict[str, Any] = {"name": secret_name, "labels": self._labels(plan)}
        if job_uid:
            # Garbage collection removes the Secret together with the Job.
            metadata["ownerReferences"] = [{
                "apiVersion": "batch/v1", "kind": "Job", "name": plan.ref, "uid": job_uid, "blockOwnerDeletion": False,
            }]
        secret = {"apiVersion": "v1", "kind": "Secret", "metadata": metadata, "type": "Opaque", "stringData": dict(plan.command.env)}
        try:
            self.api.request("POST", f"{self.ns}/secrets", json=secret, ok=(201,))
        except ApiError as exc:
            if exc.status != 409:
                raise
            # A Secret of an earlier Job with this name: it belongs to that Job, replace it.
            self.api.request("DELETE", f"{self.ns}/secrets/{secret_name}", ok=(200, 202, 404))
            self.api.request("POST", f"{self.ns}/secrets", json=secret, ok=(201,))

    def _pod(self, ref: str) -> Optional[dict[str, Any]]:
        pods = self.api.json("GET", f"{self.ns}/pods", params={"labelSelector": f"job-name={ref}"}).get("items") or []
        if not pods:
            return None
        return max(pods, key=lambda pod: str((pod.get("metadata") or {}).get("creationTimestamp") or ""))

    def probe(self, ref: str) -> Probe:
        try:
            job = self.api.json("GET", f"{self.batch}/jobs/{ref}")
        except ApiError as exc:
            if exc.status == 404:
                return Probe(state="missing", remote_now=time.time())
            raise
        status = job.get("status") or {}
        conditions = {c.get("type"): c for c in status.get("conditions") or [] if isinstance(c, Mapping)}
        pod = self._pod(ref)
        statuses = ((pod or {}).get("status") or {}).get("containerStatuses") or []
        state = (statuses[0].get("state") if statuses else None) or {}
        if "terminated" in state:
            term = state["terminated"]
            code = term.get("exitCode")
            message = msg("lt.executor.oom") if term.get("reason") == "OOMKilled" else ""
            return Probe(state="exited", started_at=_k8s_time(term.get("startedAt")), finished_at=_k8s_time(term.get("finishedAt")),
                         exit_code=int(code) if isinstance(code, int) else None, message=message)
        failed = conditions.get("Failed")
        if failed and failed.get("status") == "True":
            reason = str(failed.get("reason") or "")
            code = 124 if reason == "DeadlineExceeded" else None
            return Probe(state="exited", exit_code=code, finished_at=_k8s_time(failed.get("lastTransitionTime")),
                         message=str(failed.get("message") or reason))
        complete = conditions.get("Complete")
        if complete and complete.get("status") == "True":
            return Probe(state="exited", exit_code=0, finished_at=_k8s_time(complete.get("lastTransitionTime")))
        if "running" in state:
            return Probe(state="running", started_at=_k8s_time(state["running"].get("startedAt")))
        waiting = state.get("waiting") or {}
        reason = str(waiting.get("reason") or "")
        if reason:
            return Probe(state="starting", message=f"{reason}: {waiting.get('message') or ''}".strip(": "), stuck=reason in STUCK_REASONS)
        phase = str(((pod or {}).get("status") or {}).get("phase") or "")
        if pod is None:
            return Probe(state="starting", message=msg("lt.k8s.pod_not_created"))
        return Probe(state="starting", message=msg("lt.k8s.pod_phase", phase=phase or "Pending"))

    def signal(self, ref: str, signal: str, grace_seconds: int = 30) -> None:
        """Deletes the pod with the grace period: Kubernetes sends SIGTERM, then SIGKILL."""
        try:
            self.api.request("DELETE", f"{self.ns}/pods", params={"labelSelector": f"job-name={ref}", "gracePeriodSeconds": int(grace_seconds)})
        except ApiError as exc:
            if exc.status != 404:
                raise

    def kill(self, ref: str) -> None:
        try:
            self.api.request("DELETE", f"{self.batch}/jobs/{ref}",
                             json={"kind": "DeleteOptions", "apiVersion": "v1", "propagationPolicy": "Background", "gracePeriodSeconds": 0})
        except ApiError as exc:
            if exc.status != 404:
                raise

    def logs(self, ref: str, max_bytes: int = LOG_TAIL_BYTES) -> str:
        pod = self._pod(ref)
        if pod is None:
            return ""
        name = (pod.get("metadata") or {}).get("name")
        try:
            response = self.api.request("GET", f"{self.ns}/pods/{name}/log",
                                        params={"container": CONTAINER, "limitBytes": int(max_bytes) * 4, "tailLines": 3000})
        except ApiError as exc:
            if exc.status in (400, 404):  # the container has not started yet or the pod is gone
                return ""
            raise
        return response.text[-max_bytes:]

    def cleanup(self, ref: str) -> None:
        """ttlSecondsAfterFinished removes the Job; the log is already in the run."""
        return None

    def control(self, ref: str, argv: tuple[str, ...], cwd: str = "") -> tuple[int, str]:
        return 1, t("lt.k8s.no_control")

    def list_scripts(self, scripts_dir: str, tool: ToolTemplate) -> Optional[list[str]]:
        """Keys of the scripts ConfigMap that match the tool's masks; nothing for PVCs and images."""
        if not self.cfg.scripts_config_map:
            return None
        data = self.api.json("GET", f"{self.ns}/configmaps/{self.cfg.scripts_config_map}")
        keys = sorted(set((data.get("data") or {}).keys()) | set((data.get("binaryData") or {}).keys()))
        globs = list(tool.scripts_glob) or ["*"]
        return [key for key in keys if any(fnmatch.fnmatch(key, pattern) for pattern in globs)]


def _allowed(api: ApiClient, namespace: str, verb: str, group: str, resource: str) -> bool:
    sub = ""
    if "/" in resource:
        resource, sub = resource.split("/", 1)
    review = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectAccessReview",
        "spec": {"resourceAttributes": {"namespace": namespace, "verb": verb, "group": group, "resource": resource, "subresource": sub}},
    }
    result = api.json("POST", "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews", json=review, ok=(201,))
    return bool((result.get("status") or {}).get("allowed"))


def check_kubernetes_generator(generator: Generator, tools: Mapping[str, ToolTemplate]) -> CheckReport:
    """API version, the namespace, the permissions LoadLens needs and the scripts ConfigMap (texts in the request language)."""
    assert generator.kubernetes is not None
    cfg = generator.kubernetes
    try:
        api = kubernetes_api(generator)
        version = api.json("GET", "/version")
    except GeneratorError as exc:
        return CheckReport(ok=False, message=str(exc))
    system = f"Kubernetes {version.get('gitVersion', '?')}"
    try:
        api.json("GET", f"/api/v1/namespaces/{cfg.namespace}")
    except ApiError as exc:
        if exc.status == 404:
            problem = t("lt.k8s.no_namespace", namespace=cfg.namespace)
            return CheckReport(ok=False, message=problem, details={"system": system}, connected=True, problems=[problem])
        # A Role inside the namespace cannot read the namespace object itself: that is fine.
        if exc.status != 403:
            return CheckReport(ok=False, message=str(exc), details={"system": system}, connected=True, problems=[str(exc)])
    except GeneratorError as exc:
        return CheckReport(ok=False, message=str(exc))
    missing: list[str] = []
    try:
        for verb, group, resource in REQUIRED_VERBS:
            if not _allowed(api, cfg.namespace, verb, group, resource):
                missing.append(f"{verb} {resource}")
    except GeneratorError as exc:
        missing.append(t("lt.k8s.rbac_failed", error=exc))
    problems = [t("lt.k8s.no_rights", missing=", ".join(missing))] if missing else []
    if cfg.scripts_config_map:
        try:
            api.json("GET", f"/api/v1/namespaces/{cfg.namespace}/configmaps/{cfg.scripts_config_map}")
        except GeneratorError as exc:
            problems.append(f"ConfigMap {cfg.scripts_config_map}: {exc}")
    images = {tool_id: generator.image_for(tool) for tool_id, tool in tools.items()
              if generator.offers(tool_id, tool) and generator.image_for(tool)}
    parts = [t("lt.k8s.summary", version=version.get("gitVersion", "?"), namespace=cfg.namespace)]
    if images:
        parts.append(t("lt.k8s.images", images=", ".join(f"{tools[tool_id].display_title} — {image}" for tool_id, image in images.items())))
    else:
        parts.append(t("lt.check.no_images"))
    if problems:
        parts.append(t("lt.check.problems", problems="; ".join(problems)))
    return CheckReport(ok=not missing, message="; ".join(parts),
                       details={"system": system, "tools_found": list(images), "images": images,
                                "missing_permissions": missing},
                       connected=True, problems=problems)


__all__ = ["KubernetesExecutor", "check_kubernetes_generator", "job_name", "kubernetes_api"]
