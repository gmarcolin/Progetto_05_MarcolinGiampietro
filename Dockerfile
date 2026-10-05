FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installazione curl per l'healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Installiamo PyTorch in versione CPU-ONLY (pesa solo ~180MB invece di 2.5GB!)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Installiamo solo le dipendenze essenziali di serving
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# 3. Copiamo la configurazione e solo i file sorgente necessari per API e Dashboard
COPY config.yaml .
COPY src/api.py ./src/api.py
COPY src/app.py ./src/app.py

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
