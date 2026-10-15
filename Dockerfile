FROM python:3.12-slim

WORKDIR /app

# Dependências primeiro (cache de camada).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "dealership_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
