FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal system deps (libgomp for PyTorch CPU wheels) + bash for interactive shells
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 bash \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better build caching
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt

# Bring in the application code
COPY . /app

# Default command launches the CLI menu
CMD ["python", "-m", "src"]
