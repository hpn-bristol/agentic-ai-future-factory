FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    python3-pip \
    python3-numpy \
    git \
    wget \
    libasio-dev \
    libtinyxml2-dev \
    libssl-dev \
    libp11-dev \
    softhsm2 \
    libengine-pkcs11-openssl \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir jsondiff==1.2.0 xmltodict==0.12.0 pandas vcstool colcon-common-extensions psutil

RUN mkdir -p ~/Fast-DDS && \
    cd ~/Fast-DDS && \
    wget https://raw.githubusercontent.com/eProsima/Fast-DDS/master/fastdds.repos && \
    mkdir src && \
    vcs import src < fastdds.repos && \
    colcon build --packages-up-to fastdds

WORKDIR /app

RUN mkdir -p discovery-server-ws/src && cd discovery-server-ws && \
    wget https://raw.githubusercontent.com/eProsima/Discovery-Server/master/discovery-server.repos && \
    vcs import src < discovery-server.repos && \
    colcon build --base-paths src \
        --packages-up-to discovery-server \
        --cmake-args -DTHIRDPARTY=ON -DLOG_LEVEL_INFO=ON -DCOMPILE_EXAMPLES=ON -DINTERNALDEBUG=ON -DCMAKE_BUILD_TYPE=Debug

EXPOSE 11811/udp

# Entrypoint: start the Fast DDS Discovery Server
ENV SERVER_ID=0 \
    DISCOVERY_PORT=11811 \
    DISCOVERY_ADDR=0.0.0.0

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]