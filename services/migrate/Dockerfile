# Stage 3: Production build
FROM uv-base:latest 

WORKDIR /app

# Install dependancies
RUN uv sync --locked --no-dev --no-install-project --package migrate
# Install project
RUN uv sync --locked --no-dev  --package migrate

COPY services/migrate services/migrate

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot

RUN mkdir -p /app/data/local_model_cache /app/services/migrate && \
    chown -R nonroot:nonroot /app/data /app/services/migrate 

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

USER nonroot

CMD ["python", "services/migrate/migrate.py"]



