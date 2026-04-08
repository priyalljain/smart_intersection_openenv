FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy server requirements first (better caching)
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/server/requirements.txt

# Copy the rest of the project
COPY . .

# Install the environment package (so that my_env is importable)
RUN pip install -e .

# Create non-root user (optional, for security)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Run the FastAPI server directly with uvicorn (not openenv serve)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]