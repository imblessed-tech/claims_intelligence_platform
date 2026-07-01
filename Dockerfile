FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first
# (Docker caches this layer if requirements.txt hasn't changed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/
COPY data/claims_with_notes.csv ./data/

# Expose the API port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.claims.main:app", "--host", "0.0.0.0", "--port", "8000"]
