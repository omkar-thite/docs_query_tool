# Stage 3: Production build
FROM uv-base:latest 

WORKDIR /app

# Install dependancies
RUN uv sync --locked --no-dev --no-install-project --package embed
# Install project
RUN uv sync --locked --no-dev  --package embed

COPY services/embed services/embed

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

RUN mkdir -p /app/data/local_model_cache /app/services/embed && \
    chown -R nonroot:nonroot /app/data /app/services/embed 

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

COPY services/embed/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "embed.py"]



