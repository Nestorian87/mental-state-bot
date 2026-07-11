FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY mental_state_bot ./mental_state_bot
COPY migrations ./migrations
COPY alembic.ini ./

RUN pip install .

RUN useradd --create-home --uid 10001 appuser
RUN mkdir -p /app/data/media && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 \
    CMD mental-state-bot healthcheck || exit 1

CMD ["sh", "-c", "mental-state-bot migrate && mental-state-bot run"]
