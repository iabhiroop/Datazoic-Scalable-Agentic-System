FROM python:3.11-slim
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY app ./app
COPY resources ./resources
COPY alembic.ini ./
COPY migrations ./migrations
RUN pip install --no-cache-dir .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
