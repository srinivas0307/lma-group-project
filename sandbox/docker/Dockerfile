# Docker sandbox image (extension #5, id `docker`; docs/extensions.md). Same pinned pandas / numpy as the subprocess
# sandbox and the frozen pool, so results pickled inside the container load on the host.
#   docker build -t igt-sandbox:docker1 sandbox/docker
FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.3.3 numpy==2.1.3 \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
USER 10001
WORKDIR /tmp
ENTRYPOINT []
