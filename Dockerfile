# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy application code
COPY ingest.py .
COPY ./app /app/app
COPY ./data /app/data

# Expose port
EXPOSE 8000
