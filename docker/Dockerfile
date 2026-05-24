FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "numpy<2" \
        torch==2.0.0+cpu \
        torchvision==0.15.1+cpu \
        torchaudio==2.0.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install mmcv==2.0.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html \
    && python -m pip install -r /tmp/requirements-docker.txt

COPY . /workspace

CMD ["python", "docker/verify_environment.py"]
