# Serving image.
#
# The portable deployment path: any container host, a VPS, or a Hugging Face
# Docker Space. Vercel does not use this file, it builds from requirements.txt
# and the top-level app.py.
#
# Only the serving dependencies are installed. Training and data building are
# offline work and have no place in a request-serving image.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Hugging Face Spaces runs containers as UID 1000. Creating the user up front
# and copying with --chown avoids a recursive chown later, which would duplicate
# every file into a new layer.
RUN useradd --create-home --uid 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=app requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=app core/ core/
COPY --chown=app model/ model/
COPY --chown=app serve/ serve/
COPY --chown=app ingest/generators/ ingest/generators/
COPY --chown=app ingest/__init__.py ingest/

# 7860 is the Hugging Face Spaces default. Override with PORT elsewhere.
ENV PORT=7860 \
    RECOMMENDER_DEVICE=cpu
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health').read()"

CMD ["sh", "-c", "uvicorn serve.api:app --host 0.0.0.0 --port ${PORT}"]
