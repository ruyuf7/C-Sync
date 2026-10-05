# C-Sync, the AI Trend Agent interface, on the committed recorded run.
#
#   docker compose up --build        then open http://localhost:8501
#
# The image holds code and the recorded run only. Secrets (.env), the course
# index (vectorstore/, built from course material that is not redistributed)
# and instructor decisions stay outside it: see docker-compose.yml.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chown app /data

COPY --chown=app . .

USER app

# Decisions made on the Decision page go to the /data volume.
ENV REVIEWS_PATH=/data/reviews.json

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# Run from c_sync/ so Streamlit picks up c_sync/.streamlit/config.toml (theme).
WORKDIR /app/c_sync
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
