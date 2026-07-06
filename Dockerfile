FROM ml-base:v20260703
# Use base image

RUN switch-python 3.10

# =====================================================
# Proxy (optional)
# =====================================================
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    NO_PROXY=${no_proxy}

# =====================================================
# Environment
# =====================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tokyo

# =====================================================
# Working directory
# =====================================================
WORKDIR /app

# =====================================================
# Python virtual environments (baked in at build time)
# =====================================================
# segenv  -> mmsegmentation + its CUDA-matched deps (heavy, rarely changes)
# flowenv -> Prefect/orchestration deps (light, stable)
RUN python3 -m venv /app/segenv && \
    python3 -m venv /app/flowenv

# =====================================================
# Cache directories
# =====================================================
ENV XDG_CACHE_HOME=/app/cache \
    TORCH_HOME=/app/cache/torch \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    MPLCONFIGDIR=/app/cache/matplotlib
RUN mkdir -p \
    /app/cache \
    /app/cache/torch \
    /app/cache/huggingface \
    /app/cache/matplotlib

# =====================================================
# Copy installer files first (better layer caching) —
# app code changes below won't invalidate these expensive layers.
# =====================================================
COPY setup_env/ /app/setup_env/

# =====================================================
# Install segmentation environment (mmsegmentation itself lives HERE,
# baked into the image — NOT cloned at trigger time)
# =====================================================
RUN chmod +x /app/setup_env/install_mmseg_cu118.sh && \
    VIRTUAL_ENV="/app/segenv" \
    PATH="/app/segenv/bin:$PATH" \
    /app/setup_env/install_mmseg_cu118.sh

# =====================================================
# Install orchestration environment
# =====================================================
RUN /app/flowenv/bin/python -m pip install --upgrade pip && \
    /app/flowenv/bin/pip install \
        --no-cache-dir \
        -r /app/setup_env/requirements.txt

# =====================================================
# Copy application source (orchestration flows, api, prefect.yaml)
# =====================================================
COPY . /app

# =====================================================
# Python paths
# =====================================================
ENV PYTHONPATH="/app:/app/mmsegmentation"

# Default runtime environment
ENV PATH="/app/flowenv/bin:$PATH"

# =====================================================
# Default command — starts the Prefect worker, which then picks up
# triggered flow runs.
# =====================================================
CMD ["/app/flowenv/bin/prefect", "worker", "start", "--pool", "local-process-pool"]
