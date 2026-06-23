# metrics_config.example.py
#
# Copy this file to metrics_config.py and replace placeholder values with
# project-specific Confluence page IDs, Grafana render URLs and Loki queries.
# Do not commit the real metrics_config.py: it may contain internal URLs,
# datasource IDs, service names and Confluence page IDs.

METRICS_CONFIG = {
    "demo": {
        "title": "Demo project area",
        "services": {
            "demo-service": {
                "page_sample_id": "CONFLUENCE_TEMPLATE_PAGE_ID",
                "page_parent_id": "CONFLUENCE_PARENT_PAGE_ID",
                "metrics": [
                    {
                        "name": "RPS",
                        "grafana_url": (
                            "/render/d-solo/GRAFANA_DASHBOARD_UID/load-test?"
                            "orgId=1&panelId=1&width=1000&height=500&tz=Europe%2FMoscow"
                        ),
                    },
                    {
                        "name": "ResponseTime",
                        "grafana_url": (
                            "/render/d-solo/GRAFANA_DASHBOARD_UID/load-test?"
                            "orgId=1&panelId=2&width=1000&height=500&tz=Europe%2FMoscow"
                        ),
                    },
                    {
                        "name": "Errors",
                        "grafana_url": (
                            "/render/d-solo/GRAFANA_DASHBOARD_UID/load-test?"
                            "orgId=1&panelId=3&width=1000&height=500&tz=Europe%2FMoscow"
                        ),
                    },
                    {
                        "name": "service_cpu",
                        "grafana_url": (
                            "/render/d-solo/K8S_DASHBOARD_UID/kubernetes-workload?"
                            "orgId=1&var-datasource=PROMETHEUS_DATASOURCE_UID"
                            "&var-namespace=apps&var-type=deployment&var-workload=demo-service"
                            "&panelId=1&width=1000&height=500&tz=Europe%2FMoscow"
                        ),
                    },
                    {
                        "name": "service_mem",
                        "grafana_url": (
                            "/render/d-solo/K8S_DASHBOARD_UID/kubernetes-workload?"
                            "orgId=1&var-datasource=PROMETHEUS_DATASOURCE_UID"
                            "&var-namespace=apps&var-type=deployment&var-workload=demo-service"
                            "&panelId=3&width=1000&height=500&tz=Europe%2FMoscow"
                        ),
                    },
                ],
                "logs": [
                    {
                        "placeholder": "demo-service",
                        "filter_query": (
                            '{namespace=~"apps", service_name=~"demo-service"} |= "ERROR"'
                        ),
                    }
                ],
            }
        },
    },

    # Legacy flat format is also supported by the app:
    #
    # "demo-service": {
    #     "area": "demo",
    #     "page_sample_id": "CONFLUENCE_TEMPLATE_PAGE_ID",
    #     "page_parent_id": "CONFLUENCE_PARENT_PAGE_ID",
    #     "metrics": [],
    #     "logs": [],
    # },
}
