# SPDX-License-Identifier: Apache-2.0
"""Running load tests from the UI: tool command templates, load generators, runs and their watcher.

A run is one or more instances started together (for example a REST script in k6 and a Kafka script in
Gatling). Each instance runs a command built from the template of its tool on a load generator; LoadLens
only starts, watches and stops the processes. Metrics come from the load tool output through the query
packs of the ``lt_framework`` domain, and the report is built for the window of the whole run.
"""
