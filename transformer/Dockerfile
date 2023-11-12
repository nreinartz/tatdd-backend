FROM semitechnologies/transformers-inference:custom

LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Nico Reinartz <nico.reinartz@rwth-aachen.de>"
LABEL org.opencontainers.image.vendor="Nico Reinartz"
LABEL org.opencontainers.image.title="WeaviteTransformer"
LABEL org.opencontainers.image.description="Transformer for Weaviate, using allenai/specter2"
LABEL org.opencontainers.image.source = "https://github.com/nreinartz/tatdd-backend"

RUN MODEL_NAME=allenai/specter2_base ./download.py