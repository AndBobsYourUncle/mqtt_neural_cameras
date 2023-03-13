project(MqttNeuralSystem)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME mqtt_neural_cameras)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(gflags REQUIRED)
find_package(PahoMqttCpp REQUIRED)
find_package(nadjieb_mjpeg_streamer REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
  openvino::runtime
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  gflags paho-mqttpp3
  paho-mqtt3as
  yaml-cpp
  ssl
  crypto
  ${OpenCV_LIBS})
