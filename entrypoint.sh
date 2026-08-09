#!/bin/sh
set -e

echo "Fetching model artifacts from S3..."
python download_model.py

echo "Starting API..."
exec uvicorn src.app:app --host 0.0.0.0 --port 8000
