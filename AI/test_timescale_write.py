import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

from settings import CONFIG
from AI.db_store import save_domain_labeled


def main():
    # Готовим простую тестовую серию на 5 точек
    now = datetime.now(timezone.utc)
    idx = pd.date_range(start=now - timedelta(minutes=4), end=now, freq="1min", tz="UTC")
    df = pd.DataFrame({"test_series=constant": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

    labeled = [{"label": "Test: constant", "df": df}]
    domain_conf = {"labels": ["Test: constant"], "promql_queries": ["test://synthetic"]}

    run_meta = {
        "run_id": f"ts_test_{int(time.time())}",
        "run_name": "timescale_selftest",
        "service": "selftest",
        "start_ms": int((now - timedelta(minutes=4)).timestamp() * 1000),
        "end_ms": int(now.timestamp() * 1000),
    }

    storage_cfg = ((CONFIG.get("storage", {}) or {}).get("timescale") or {})
    if not storage_cfg:
        raise SystemExit("Timescale config (CONFIG['storage']['timescale']) не задан.")

    save_domain_labeled(
        domain_key="selftest",
        domain_conf=domain_conf,
        labeled_dfs=labeled,
        run_meta=run_meta,
        storage_cfg=storage_cfg,
    )
    print("OK: тестовые строки записаны. Проверьте public.metrics.")


if __name__ == "__main__":
    main()


