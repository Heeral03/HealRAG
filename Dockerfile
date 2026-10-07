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
ENV DISABLE_BG_PRELOAD=1
ENV ENABLE_CROSS_ENCODER=0

# Pre-seed corpus, build Qdrant index, and pre-download CrossEncoder model weights during image build stage
RUN python3 src/seeder.py && python3 src/embedder.py && python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Expose FastAPI port
EXPOSE 8000

# Run HealRAG Uvicorn server dynamically binding to $PORT
CMD ["python3", "src/start.py"]
