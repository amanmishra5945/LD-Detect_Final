# Smart Learning Disability Screening System (Ages 5–12)
# Production Container Image

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal system dependencies required by opencv-python-headless and scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching)
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code, models, and frontend assets
COPY backend /app/backend
COPY frontend /app/frontend

# Set working directory to backend so relative paths to models/data/frontend work identically
WORKDIR /app/backend

# Expose default application port
EXPOSE 8000

# Start production ASGI server with dynamic port support
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
