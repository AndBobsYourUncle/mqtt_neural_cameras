# MQTT Neural Cameras

## Introduction

This application builds off of the Open Model Zoo YoloV3 multiple camera detection demo found here:
https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/multi_channel_object_detection_demo_yolov3/cpp

Some of the common includes were cloned over, and a couple slight tweaks were made. The main.cpp is also largely the same, with a bunch of annotated additions for parsing a YAML file as input (instead of command line arguments), creating a MJPEG stream for the detection output, and connecting to the MQTT broker to publish detections.

In the end, the application that runs is largely the same as the demo, except that you can see the output through an MJPEG stream, configure it purely via a YAML file, and get the live detection results fed into an MQTT broker.

## Getting Started

To run this, first create a `config.json` file that matches the format of the one found here:
https://github.com/AndBobsYourUncle/mqtt_neural_cameras/blob/master/options.json

You'll need to replace the MQTT host connection string with your MQTT broker's IP, as well as your camera connection strings.

The only YOLO model provided in the Docker image is Tiny YOLOv3, which the sample `options.json` already has as its value. The full YOLOv3 model would likely work as well, but with a lower FPS.

Any input that can be loaded in through OpenCV and GStreamer should work, although I've only tried it with an RTSP stream and an MJPEG camera feed so far.

Also, any of the standard COCO classes are trackable, as those are the ones trained into the provided YOLOv3 model. The sample configuration has "person" as the only tracked class, but you can add others as well.

After creating your `options.yaml` file, you can then run this application using the prebuilt Docker image like this:

```shell
docker run -u root:root --rm --device /dev/dri --device-cgroup-rule='c 189:* rmw' \
  -v /dev/bus/usb:/dev/bus/usb -v /path/to/config.yaml:/data -p 8085:8080 andbobsyouruncle/amd64-mqtt-neural-cameras:0.0.1
```

This gives the Docker container access to the GPU or a connected Neural Compute Stick 2, if one is connected to the host machine. Only certain Intel GPUs are compatible (as this is using Intel's OpenVINO SDK), as well as Intel CPUs.

Running this with the "CPU" device at least once will output which other devices have been detected as available. For example, you might see "MYRIAD" if a Neural Compute Stick 2 was detected as plugged into the host's USB port, or "GPU", if a compatible Intel GPU is present.

