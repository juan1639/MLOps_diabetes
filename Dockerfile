FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY mlruns/ ./mlruns/

EXPOSE 8000

# Render (and most cloud services) tell the app which port to use through the
# PORT environment variable. Shell form is needed so ${PORT} gets expanded.
# Locally it falls back to 8000.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
