FROM openvino/ubuntu20_dev:2022.1.0

USER root

RUN apt-get update && \
    apt-get install pkg-config software-properties-common -y --no-install-recommends && \
    add-apt-repository ppa:rmescandon/yq && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential gcc make cmake cmake-gui cmake-curses-gui \
        libssl-dev \
        libgflags-dev \
        ffmpeg \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libdc1394-22-dev \
        libgtk2.0-dev libgtk-3-dev gnome-devel \
        libcanberra-gtk-module libcanberra-gtk3-module \
        yq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV NO_AT_BRIDGE=1

WORKDIR /tmp/dependencies

RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.x && \
    mkdir opencv_build && \
    cd opencv_build && \
    cmake ../ && \
    make -j4 && \
    make install && \
    cd ../../ && rm -rf opencv

WORKDIR /tmp/dependencies

RUN git clone https://github.com/eclipse/paho.mqtt.c.git && \
    cd /tmp/dependencies/paho.mqtt.c && \
    git checkout v1.3.8 && \
    cmake -Bbuild -H. -DPAHO_ENABLE_TESTING=OFF -DPAHO_BUILD_STATIC=ON \
    -DPAHO_WITH_SSL=ON -DPAHO_HIGH_PERFORMANCE=ON && \
    cmake --build build/ --target install && \
    ldconfig && \
    cd ../ && rm -rf paho.mqtt.c

WORKDIR /tmp/dependencies

RUN git clone https://github.com/eclipse/paho.mqtt.cpp && \
    cd /tmp/dependencies/paho.mqtt.cpp && \
    cmake -Bbuild -H. -DPAHO_BUILD_STATIC=ON && \
    cmake --build build/ --target install && \
    ldconfig && \
    cd ../ && rm -rf paho.mqtt.cpp

WORKDIR /tmp/dependencies

RUN git clone https://github.com/nadjieb/cpp-mjpeg-streamer.git && \
    cd /tmp/dependencies/cpp-mjpeg-streamer && \
    cmake . && \
    make install && \
    cd ../ && rm -rf cpp-mjpeg-streamer

WORKDIR /tmp/dependencies

RUN git clone https://github.com/jbeder/yaml-cpp.git && \
    cd /tmp/dependencies/yaml-cpp && \
    cmake DYAML_BUILD_SHARED_LIBS=ON . && \
    make install && \
    cd ../ && rm -rf yaml-cpp

USER openvino

WORKDIR /home/openvino

RUN omz_downloader --name yolo-v3-tiny-tf && \
    omz_converter --name yolo-v3-tiny-tf

RUN git clone https://github.com/AndBobsYourUncle/mqtt_neural_cameras.git && \
    cd mqtt_neural_cameras && \
    git checkout ab8e4ca && cmake . && make

WORKDIR /home/openvino/mqtt_neural_system

USER root

RUN chmod +x /home/openvino/mqtt_neural_system/start_mqtt_neural_cameras.sh

CMD [ "/bin/bash", "-c", "/home/openvino/mqtt_neural_cameras/start_mqtt_neural_cameras.sh" ]
