# Load generator of the dev stack for the "Run test" page of LoadLens: an SSH endpoint with k6, Locust and
# Java for the Gatling bundle. LoadLens logs in as "loadlens" with the key from keys/ (see ../README.md).
# The SSH server is devsshd.py (paramiko): it installs from wheels downloaded on the host, so the image
# builds on networks where the Debian mirror and PyPI are blocked.
FROM locustio/locust:2.31.8

USER root
COPY wheels /tmp/wheels
RUN /opt/venv/bin/pip install --no-cache-dir --no-index --find-links /tmp/wheels paramiko \
 && rm -rf /tmp/wheels \
 && useradd --create-home --shell /bin/sh loadlens \
 && mkdir -p /opt/loadlens/runs /opt/loadlens/tests \
 && chown loadlens:loadlens /opt/loadlens/runs \
 && ln -s /opt/venv/bin/locust /usr/local/bin/locust

COPY --from=grafana/k6:0.54.0 /usr/bin/k6 /usr/local/bin/k6
# A JDK, not a JRE: the Gatling bundle compiles the simulations with javac before the run.
COPY --from=eclipse-temurin:21-jdk /opt/java/openjdk /opt/java/openjdk
RUN ln -s /opt/java/openjdk/bin/java /usr/local/bin/java && ln -s /opt/java/openjdk/bin/javac /usr/local/bin/javac

COPY devsshd.py /opt/devsshd/devsshd.py
COPY entrypoint.sh /entrypoint.sh
EXPOSE 2222
ENTRYPOINT ["sh", "/entrypoint.sh"]
