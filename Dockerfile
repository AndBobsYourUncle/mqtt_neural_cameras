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

RUN git clone https://github.com/eclipse/paho.mqtt.c.git

WORKDIR /tmp/dependencies/paho.mqtt.c

RUN git checkout v1.3.8

RUN cmake -Bbuild -H. -DPAHO_ENABLE_TESTING=OFF -DPAHO_BUILD_STATIC=ON \
    -DPAHO_WITH_SSL=ON -DPAHO_HIGH_PERFORMANCE=ON

RUN cmake --build build/ --target install

RUN ldconfig

WORKDIR /tmp/dependencies

RUN git clone https://github.com/eclipse/paho.mqtt.cpp

WORKDIR /tmp/dependencies/paho.mqtt.cpp

RUN cmake -Bbuild -H. -DPAHO_BUILD_STATIC=ON

RUN cmake --build build/ --target install

RUN ldconfig

WORKDIR /tmp/dependencies

RUN git clone https://github.com/nadjieb/cpp-mjpeg-streamer.git

WORKDIR /tmp/dependencies/cpp-mjpeg-streamer

RUN cmake .

RUN make install

WORKDIR /tmp/dependencies

RUN omz_downloader --name yolo-v3-tiny-tf

RUN omz_converter --name yolo-v3-tiny-tf

WORKDIR /tmp/dependencies

RUN git clone https://github.com/AndBobsYourUncle/mqtt_neural_system.git

WORKDIR /tmp/dependencies/mqtt_neural_system

RUN cmake .

RUN make

WORKDIR /app

RUN mv /tmp/dependencies/mqtt_neural_system/mqtt_neural_system ./

RUN mv /tmp/dependencies/public/yolo-v3-tiny-tf/FP16/yolo-v3-tiny-tf.* ./

RUN rm -rf /tmp/dependencies

USER openvino
