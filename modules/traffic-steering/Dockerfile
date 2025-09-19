FROM ros:humble-ros-base AS builder
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-pip \
    python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/netem_ws

COPY src ./src

RUN rosdep update \
 && rosdep install --rosdistro humble --from-paths src --ignore-src -y \
    --skip-keys="ament_python"

RUN . /opt/ros/humble/setup.sh && colcon build

FROM ros:humble-ros-core
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /root/netem_ws

COPY --from=builder /root/netem_ws/install ./install

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash", "-lc", "source install/setup.bash && ros2 run traffic_steering traffic_steering"]
