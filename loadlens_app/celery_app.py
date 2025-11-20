"""Minimal Celery application used for background tasks (image downloads, etc.)."""

from __future__ import annotations

import os

try:
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover - fallback for environments без celery
    class _SyncAsyncResult:
        def __init__(self, value):
            self._value = value

        def get(self, timeout=None):
            return self._value

    class Celery:
        def __init__(self, *args, **kwargs):
            self.conf = {}

        def task(self, *task_args, **task_kwargs):
            def decorator(func):
                def delay_wrapper(*args, **kwargs):
                    return _SyncAsyncResult(func(*args, **kwargs))

                func.delay = delay_wrapper
                return func

            return decorator


def _make_celery() -> Celery:
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend_url = os.getenv("CELERY_RESULT_BACKEND", broker_url)
    app = Celery("loadlens_app", broker=broker_url, backend=backend_url)
    always_eager = os.getenv("CELERY_TASK_ALWAYS_EAGER", "1") == "1"
    app.conf.update(
        task_track_started=True,
        task_always_eager=always_eager,
        task_eager_propagates=True,
        result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
    )
    return app


celery_app = _make_celery()

__all__ = ["celery_app"]


