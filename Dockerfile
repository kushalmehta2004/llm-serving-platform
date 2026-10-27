FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ server/
COPY .env.example .env.example

ENV HOST=0.0.0.0
ENV PORT=8000
ENV SCHEDULER=paged

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
