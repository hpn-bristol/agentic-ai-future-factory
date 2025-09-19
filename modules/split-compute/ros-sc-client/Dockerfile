FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv python3-pip python-is-python3 \
    ros-humble-cv-bridge ros-humble-image-transport ros-humble-sensor-msgs \
    git wget unzip iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir psutil && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.4.0 torchvision==0.19.0 && \
    pip install -r requirements.txt

COPY y5CLayersROS.py y5CLayersROS_vis.py monitoring.py yolov5s.pt ./

ARG CLIENT_SCRIPT=y5CLayersROS.py
ENV CLIENT_SCRIPT=${CLIENT_SCRIPT}

CMD ["/bin/bash","-lc","source /opt/ros/humble/setup.bash && python ${CLIENT_SCRIPT}"]