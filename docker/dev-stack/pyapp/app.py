# SPDX-License-Identifier: Apache-2.0
"""Small FastAPI app of the dev stack: Python runtime and HTTP metrics for the query packs."""

import random
import time

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)


@app.get("/items/{item_id}")
def read_item(item_id: int) -> dict:
    # A little CPU and latency that grows with the id, so steps of load show up in the series.
    total = sum(i * i for i in range(2000 + item_id))
    time.sleep(random.uniform(0.002, 0.02))
    return {"id": item_id, "checksum": total % 997}


@app.get("/fail")
def fail() -> dict:
    raise HTTPException(status_code=503, detail="intentional failure for the error-rate queries")
