# Multi-Agent RAG Research Platform — service image.
#
# The retrieval index is built at IMAGE BUILD TIME, not on container start:
# `docs/` is static and already committed, so baking `data/chroma_db` and
# `data/chunks.json` into the image means `docker compose up` serves a working
# UI immediately, with no manual indexing step (Phase 10's exit criterion).
# Phase 2 needs no API key, so this build step works with zero secrets present.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so this layer is cached and only rebuilds when
# requirements actually change, not on every source edit.
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

# Application code and the static corpus the index is built from.
COPY app/ app/
COPY docs/ docs/
COPY pytest.ini .

# Bakes data/chroma_db and data/chunks.json into the image. Always rebuilds
# from scratch (app/retrieval/index.py's own behavior), so there is never a
# stale chunk left over from a previous build.
RUN python -m app.retrieval.index

EXPOSE 8000

# No --reload: this is a service image, not a dev loop. AsyncSqliteSaver's
# checkpoints and the semantic cache's Redis connection are both external to
# this process either way, so a container restart behaves the same as the
# `--reload` case Phase 4/5 already proved durable.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
