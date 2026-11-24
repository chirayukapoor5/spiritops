FROM ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa8217f92b80223bc83aac7db36ec80a84437dbc0d04bbc659cae030d8c9 AS uv
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

ARG GIT_SHA=unknown
ENV SPIRITOPS_GIT_SHA=$GIT_SHA SPIRITOPS_HOME=/var/lib/spiritops HOME=/home/spiritops
LABEL org.opencontainers.image.title="SpiritOps" org.opencontainers.image.revision=$GIT_SHA
RUN groupadd --gid 10001 spiritops && useradd --uid 10001 --gid spiritops --create-home spiritops \
    && install -d -o spiritops -g spiritops /var/lib/spiritops
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY configs ./configs
COPY scripts ./scripts
COPY tests/fixtures ./tests/fixtures
USER 10001:10001
CMD ["python", "scripts/container.py", "api"]
