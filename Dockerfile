# Use official Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_DIR=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create cache directory and set permissions
RUN mkdir -p $CACHE_DIR && chmod -R 777 $CACHE_DIR

# Expose port (7860 for Hugging Face Spaces compatibility)
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]