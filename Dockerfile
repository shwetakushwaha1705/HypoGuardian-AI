# HypoGuardian AI: Production Docker Container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase and artifacts
COPY app.py config.py ./
COPY src/ ./src/
COPY utils/ ./utils/
COPY models/ ./models/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/
COPY docs/ ./docs/

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck to verify the web server is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
