FROM python:3.12-slim

# System deps: tesseract for the scanned-PDF adapter; poppler for rasterization.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m src.cli synth || true   # build sample pairs + ground truth at image time

EXPOSE 8000
# ANTHROPIC_API_KEY comes from the environment (docker-compose / -e). Never baked in.
CMD ["python", "-m", "uvicorn", "src.app:api", "--host", "0.0.0.0", "--port", "8000"]
