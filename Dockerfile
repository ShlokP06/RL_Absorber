# ── Adsorber — FastAPI Backend ────────────────────────────────────────────────
#
# Build (CPU, default):
#   docker build -t adsorber-backend .
#
# Build (CUDA 12.1 — for GPU inference):
#   docker build --build-arg USE_CUDA=1 -t adsorber-backend .
#
# Run standalone:
#   docker run -p 8000:8000 adsorber-backend

ARG USE_CUDA=0

# ── Base image: CUDA or slim ──────────────────────────────────────────────────
FROM python:3.11-slim AS base-cpu
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-cuda

# Select base based on build arg
FROM base-${USE_CUDA:+cuda}${USE_CUDA:-cpu} AS final

# For CUDA base, python isn't pre-installed
ARG USE_CUDA=0
RUN if [ "$USE_CUDA" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            python3.11 python3.11-dev python3-pip curl gcc g++ \
        && ln -sf python3.11 /usr/bin/python \
        && ln -sf pip3 /usr/bin/pip \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
            gcc g++ curl \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# ── Torch: CPU or CUDA wheel ──────────────────────────────────────────────────
# CPU wheel is ~300 MB; CUDA wheel is ~2 GB — only pull what you need.
RUN if [ "$USE_CUDA" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── Python dependencies ───────────────────────────────────────────────────────
COPY demo/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/          ./src/
COPY demo/backend/ ./demo/backend/
COPY download_models.py .

# ── Startup: download models then serve ───────────────────────────────────────
# Models are downloaded at runtime (not bake-time) so the image stays lean
# and model updates don't require a full rebuild.
CMD ["sh", "-c", \
    "python download_models.py && \
     cd demo/backend && \
     uvicorn main:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000
