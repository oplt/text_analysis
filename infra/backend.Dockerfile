FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-tur \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend /app/backend
RUN cd backend && uv sync --frozen --extra pdf-ocr --no-dev
ENV PATH="/app/backend/.venv/bin:$PATH" PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
