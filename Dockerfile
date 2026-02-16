FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py ./
COPY src/ ./src/

# Copy frontend assets
COPY templates/ ./templates/
COPY static/ ./static/

# Copy data files (models, PDFs, embeddings)
COPY data/ ./data/
COPY embeddings.pkl ./

# Expose Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
