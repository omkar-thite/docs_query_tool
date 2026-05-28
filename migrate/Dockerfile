FROM python:3.12 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
# ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    # --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project 

    # Install project
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    # --mount=type=cache,target=/root/.cache/uv  \
    uv sync --locked --no-dev  

# COPY ../shared/ /shared/
# RUN uv pip install /shared/

# Production
FROM python:3.12-slim 

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

RUN mkdir -p /app/data/local_model_cache /app/migrate && \
    chown -R nonroot:nonroot /app/data /app/migrate 

COPY --from=builder --chown=nonroot:nonroot /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=5001


CMD ["python", "migrate/migrate.py"]