FROM python:3.11-slim

WORKDIR /app

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so they're cached on rebuild
COPY pyproject.toml ./
RUN pip install --no-cache-dir "fastapi>=0.111" "uvicorn[standard]>=0.29" pyyaml "groq>=0.4.0"

# Copy application code
COPY app_web.py ./
COPY configs/ ./configs/
COPY src/ ./src/

EXPOSE 8000

# GROQ_API_KEY must be passed in at runtime:
#   docker run -e GROQ_API_KEY=... -p 8000:8000 spotify-support-agent
CMD ["uvicorn", "app_web:app", "--host", "0.0.0.0", "--port", "8000"]
