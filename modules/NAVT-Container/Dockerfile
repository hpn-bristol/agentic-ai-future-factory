FROM osrf/ros:humble-simulation

RUN apt-get update && \
    apt-get install -y python3-colcon-common-extensions python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir pandas==2.1.0
RUN pip install --no-cache-dir scikit-learn==1.6.1
RUN pip install --no-cache-dir scipy==1.11.1
RUN pip install --no-cache-dir joblib==1.3.2
RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

WORKDIR /ros2_ws

COPY . /ros2_ws

RUN rm -rf build log || true

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 run NAVT NAVT"]
