FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies if needed (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/server/requirements.txt

# Copy project files
COPY . /app

# Expose default port
EXPOSE 8000

# Start FastAPI with Uvicorn; use Vercel-provided PORT or default 8000
CMD ["bash", "-lc", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]