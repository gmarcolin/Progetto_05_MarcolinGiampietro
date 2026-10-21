FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installazione PyTorch CPU-Only
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Installazione dipendenze di produzione
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia sorgenti necessari per il serving
COPY config.yaml .
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
