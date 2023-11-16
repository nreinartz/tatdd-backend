
FROM python:3.10-bookworm

LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.version="1.0.0-dev.5"
LABEL org.opencontainers.image.authors="Nico Reinartz <nico.reinartz@rwth-aachen.de>"
LABEL org.opencontainers.image.vendor="Nico Reinartz"
LABEL org.opencontainers.image.title="Trend Detection API and processing service"
LABEL org.opencontainers.image.description="WebAPI for trend detection and processing service for trend detection"
LABEL org.opencontainers.image.source = "https://github.com/nreinartz/tatdd-backend"

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --no-deps

COPY ./src .

ENV WEAVIATE_HOST="weaviate"
ENV WEAVIATE_REST_PORT="8080"
ENV WEAVIATE_GRPC_PORT="50051"

ENV POSTGRES_HOST="postgres"
ENV POSTGRES_USER="postgres"
ENV POSTGRES_PASSWORD="postgres"
ENV POSTGRES_DB="trend_api"

ENV TREND_DESCRIPTOR="rule_based"

ENV OPENAI_MODEL="gpt-4"
ENV OPENAI_API_BASE=
ENV OPENAI_API_KEY=

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]

