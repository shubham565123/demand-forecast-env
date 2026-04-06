FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir \
    openenv-core>=0.2.1 \
    pydantic>=2.0.0 \
    uvicorn>=0.20.0 \
    fastapi>=0.100.0 \
    openai>=1.0.0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
