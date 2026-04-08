FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy server requirements first
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/server/requirements.txt

# Copy the rest of the project
COPY . .

# Install root dependencies (if any)
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]