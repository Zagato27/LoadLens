# SPDX-License-Identifier: Apache-2.0
"""Locust scenario of the dev stack: requests to the FastAPI app, a few of them failing."""

import random

from locust import HttpUser, between, task


class ShopUser(HttpUser):
    wait_time = between(0.2, 0.6)

    @task(10)
    def item(self):
        self.client.get(f"/items/{random.randint(1, 999)}", name="GET /items/{id}")

    @task(1)
    def fail(self):
        self.client.get("/fail", name="GET /fail")
