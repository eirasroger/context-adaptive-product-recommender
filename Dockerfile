# Serving image for container hosts such as a Hugging Face Space. Vercel does not use it.

FROM node:24-slim AS frontend

WORKDIR /build
COPY contracts/ contracts/
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci
COPY frontend/ frontend/
RUN cd frontend && npm run build

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Spaces run as UID 1000; copying with --chown avoids a duplicate layer from a later chown.
RUN useradd --create-home --uid 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=app requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=app core/ core/
COPY --chown=app serve/ serve/
COPY --chown=app ingest/generators/ ingest/generators/
COPY --chown=app ingest/__init__.py ingest/
COPY --chown=app app.py ./
COPY --chown=app --from=frontend /build/frontend/dist frontend/dist/

# 7860 is the Hugging Face Spaces default. Override with PORT elsewhere.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health').read()"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
