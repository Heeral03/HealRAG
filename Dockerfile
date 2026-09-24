FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

ENV PYTHONPATH=/app/src
ENV PORT=8000

# Pre-seed corpus and pre-build Qdrant index during image build stage
RUN python3 src/seeder.py && python3 src/embedder.py

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with Uvicorn on 0.0.0.0:8000
CMD ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
