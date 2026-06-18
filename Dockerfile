# Passport Photo Validator API - Dockerfile
# Bangladesh Passport Photo Validation Service v2.0

FROM python:3.12.10-slim

# Install system dependencies for OpenCV
# Note: libgl1-mesa-glx is deprecated in Debian 12 (Bookworm), use libgl1 instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY validator.py .
COPY index.html .
COPY validator_v22.py .
COPY validator_mediapipe_fixed.py .
COPY validator_mediapipe_v1.py .

# Set default port (override at runtime with -e PORT=5001)
ENV PORT=5001

# Expose port
EXPOSE 5001

# Health check - uses the PORT env var dynamically
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')\"" || exit 1

# Run the application - binds to 0.0.0.0 on the PORT env var
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]