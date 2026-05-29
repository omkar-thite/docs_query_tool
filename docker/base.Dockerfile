# Stage 1: Extract only configuration files
FROM alpine AS config-extractor
WORKDIR /app
# Copy the entire monorepo
COPY pyproject.toml uv.lock .
COPY services/ services/
# Recursively delete all files that are not uv configurations
RUN find . -type f ! -name 'pyproject.toml' ! -name 'uv.lock' -delete


# Stage 2: Install uv dependancies
FROM python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \ 
    UV_PYTHON_DOWNLOADS=0

COPY --from=config-extractor /app /app

