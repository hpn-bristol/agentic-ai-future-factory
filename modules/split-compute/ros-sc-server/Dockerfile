FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    python3-opencv python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0 torchvision==0.19.0 && \
    pip install -r requirements.txt

COPY y5SLayers.py yolov5s.pt ./

EXPOSE 12350

CMD ["python", "y5SLayers.py"]