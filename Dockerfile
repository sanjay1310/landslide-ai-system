FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY artifacts ./artifacts
COPY scripts ./scripts
COPY main.py train.py train_forecaster.py train_lstm_forecaster.py train_gnn.py run_api.py ./

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

USER appuser

EXPOSE 8000 8501

CMD ["python", "run_api.py"]
