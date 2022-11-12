FROM openvino/ubuntu20_dev

USER root

WORKDIR /tmp/dependencies

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc make cmake cmake-gui cmake-curses-gui \
        libssl-dev \
        libgflags-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

RUN git clone https://github.com/eclipse/paho.mqtt.c.git && \
    cd /tmp/dependencies/paho.mqtt.c && \
    git checkout v1.3.8 && \
    cmake -Bbuild -H. -DPAHO_ENABLE_TESTING=OFF -DPAHO_BUILD_STATIC=ON \
    -DPAHO_WITH_SSL=ON -DPAHO_HIGH_PERFORMANCE=ON && \
    cmake --build build/ --target install && \
    ldconfig

WORKDIR /tmp/dependencies

RUN git clone https://github.com/eclipse/paho.mqtt.cpp && \
    cd /tmp/dependencies/paho.mqtt.cpp && \
    cmake -Bbuild -H. -DPAHO_BUILD_STATIC=ON && \
    cmake --build build/ --target install && \
    ldconfig

WORKDIR /tmp/dependencies

RUN git clone https://github.com/nadjieb/cpp-mjpeg-streamer.git && \
    cd /tmp/dependencies/cpp-mjpeg-streamer && \
    cmake . && \
    make install

WORKDIR /tmp/dependencies

RUN omz_downloader --name yolo-v3-tiny-tf && \
    omz_converter --name yolo-v3-tiny-tf

WORKDIR /tmp/dependencies

RUN git clone https://github.com/AndBobsYourUncle/mqtt_neural_system.git && \
    cd /tmp/dependencies/mqtt_neural_system && \
    cmake . && \
    make

WORKDIR /app

RUN mv /tmp/dependencies/mqtt_neural_system/mqtt_neural_system ./ && \
    mv /tmp/dependencies/public/yolo-v3-tiny-tf/FP16/yolo-v3-tiny-tf.* ./ && \
    rm -rf /tmp/dependencies

USER openvino
