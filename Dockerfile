FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY server/ ./server/
COPY openenv.yaml .

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["python", "-c", "from server.app import main; main()"]
