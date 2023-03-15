# MQTT Neural Cameras

## Introduction

This application builds off of the Open Model Zoo YoloV3 multiple camera detection demo found here:
https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/multi_channel_object_detection_demo_yolov3/cpp

Some of the common includes were cloned over, and a couple slight tweaks were made. The main.cpp is also largely the same, with a bunch of annotated additions for parsing a YAML file as input (instead of command line arguments), creating a MJPEG stream for the detection output, and connecting to the MQTT broker to publish detections.

In the end, the application that runs is largely the same as the demo, except that you can see the output through an MJPEG stream, configure it purely via a YAML file, and get the live detection results fed into an MQTT broker.

The philosophy behind this project is simple: it shouldn't do anything above and beyond simply informing MQTT of detected classes, and their confidence/areas. This is to maximize the usefulness and modularity of the app, and let it be used in many different setup scenarios. For example, a user could be running Motioneye, and have Motioneye record videos for them, but use this app's MQTT values to trigger Motioneye to record and stop recording.

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
  -v /dev/bus/usb:/dev/bus/usb -v /path/containing/config_json_file:/data \
  -p 8085:8080 andbobsyouruncle/amd64-mqtt-neural-cameras:0.0.1
```

This gives the Docker container access to the GPU or a connected Neural Compute Stick 2, if one is connected to the host machine. Only certain Intel GPUs are compatible (as this is using Intel's OpenVINO SDK), as well as Intel CPUs.

Running this with the "CPU" device at least once will output which other devices have been detected as available. For example, you might see "MYRIAD" if a Neural Compute Stick 2 was detected as plugged into the host's USB port, or "GPU", if a compatible Intel GPU is present.

For example, here is the start of the output when running with "CPU" as the device:

```shell
[ INFO ] OpenVINO
[ INFO ]  version: 2022.1.0
[ INFO ]  build: 2022.1.0-7019-cdb9bec7210-releases/2022/1
[E:] [BSL] found 0 ioexpander device
Available devices: CPU GNA GPU MYRIAD
```

This is letting you know that "CPU", "GNA", "GPU", and "MYRIAD" are the available devices that were detected. You can then use any of these in your `options.json` file.

## Usage

Once the application is running, it does a couple things. Firstly, it starts streaming an MJPEG view of all cameras and what it is detecting live, including bounding boxes. This is being broadcast on `http://IP_OF_APP:8080/detection_output`. Opening that URL in a web browser will give you a live peek into what the neural model is processing.

Another thing that is going on is that it is publishing MQTT topics containing the highest confidence percentage, and pixel area of detections from each camera.

For example, if you have a "Front Door" camera configured, it would publish this, if there was a person detected in the front door camera, with a confidence of 90% and a pixel area of 10k pixels squared:

```
mqtt_neural_cameras/front-door/person/confidence => 90
mqtt_neural_cameras/front-door/person/area => 10000
```

Using these MQTT topcis, one can create automations that do anything from turning on lights, to trigging an NVR to start/end recording.

## Installation

The best and easiest way to install this on a system is through Home Assistant as an addon.

Add this repository to your addon store in Home Assistant:
https://github.com/AndBobsYourUncle/hassio-addons

You can then install and configure the addon through Home Assistant's UI. All values from the config.json are exposed via the configuration of the addon itself.

Once running, various MQTT sensors can be created for each camera within Home Assistant, and even the addition of an MJPEG camera for viewing the live detections.

## Integration with Motioneye

A simple automation using the Motioneye addon's "Motion webcontrol" can be used to automatically stop/start recording based on detections. To start, you want to configure Motioneye to record movies based on motion. Then create a motion mask that blocks out the entire frame within Motioneye's UI. This enables Motioneye to record based on motion events, but because you are masking out the entire frame, then the only way in which it will start recording is if we artificially trigger motion on the camera.

Your automation can then make an API GET request to
`http://IP_OF_MOTIONEYE_HOST:7999/1/config/set?emulate_motion=on`
to start recording on camera 1 in Motioneye.

Another call to 
`http://IP_OF_MOTIONEYE_HOST:7999/1/config/set?emulate_motion=off`
would then stop the recording.

I've typically gone with a bit of NodeRED to automate this, as it makes it easier to setup the logic around expected area and confidence, time intervals, and timers to automatically stop recording, but it should also be doable with plain Home Assistant automations as well.

## Restrictions

I've only built a Docker image for this based on the amd64 architecture, so running it as an addon in Home Assistant using a Raspberry Pi is not possible at the moment. However, based on how OpenVINO supports running within Docker on a Raspberry Pi, it should be possible if an image was built for that architecture.

Also, running this on a Raspberry PI would limit you to only being able to use a Neural Compute Stick 2 as the inference device, since the CPU in a RPI is not an Intel CPU. This is partly why I have only really built it for amd64 at the moment.
