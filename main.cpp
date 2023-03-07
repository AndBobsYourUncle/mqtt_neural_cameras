// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
* \brief The entry point for the OpenVINIO multichannel_yolo_detection demo application
* \file multichannel_yolo_detection/main.cpp
* \example multichannel_yolo_detection/main.cpp
*/
#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
#include <mutex>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <opencv2/opencv.hpp>
#include <openvino/op/region_yolo.hpp>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "input.hpp"
#include "multichannel_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

// ADDED STUFF START

const std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

#include "mqtt/async_client.h"

static const char stream_message[] = "Stream the live detection output as an MJPEG.";

DEFINE_bool(stream, false, stream_message);

static const char display_width_message[] = "Width of the output display. Default value is 800.";

DEFINE_uint32(dw, 800, display_width_message);

static const char display_height_message[] = "Height of the output display. Default value is 600.";

DEFINE_uint32(dh, 600, display_height_message);

static const char mqtt_host_message[] = "MQTT host url to publish detections.";

DEFINE_string(mh, "", mqtt_host_message);

static const char mqtt_username_message[] = "MQTT host username.";

DEFINE_string(mu, "", mqtt_username_message);

static const char mqtt_password_message[] = "MQTT host password.";

DEFINE_string(mp, "", mqtt_password_message);

const int    QOS = 1;

const auto PERIOD = std::chrono::seconds(5);

const int MAX_BUFFERED_MSGS = 120;  // 120 * 5sec => 10min off-line buffering

const std::string MQTT_CLIENT_ID { "mqtt_neural_system" };

#include <nadjieb/mjpeg_streamer.hpp>

// for convenience
using MJPEGStreamer = nadjieb::MJPEGStreamer;

std::vector<int> stream_params = {cv::IMWRITE_JPEG_QUALITY, 90};

MJPEGStreamer streamer;

#include "yaml-cpp/yaml.h"

// ADDED STUFF END

namespace {
constexpr char threshold_message[] = "Probability threshold for detections";
DEFINE_double(t, 0.5, threshold_message);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    slog::info << ov::get_openvino_version() << slog::endl;
    if (FLAGS_h || argc == 1) {
        std::cout << "\n    [-h]              " << help_message
                  << "\n     -i               " << input_message
                  << "\n    [-loop]           " << loop_message
                  << "\n    [-duplicate_num]  " << duplication_channel_number_message
                  << "\n     -m <path>        " << model_path_message
                  << "\n    [-d <device>]     " << target_device_message
                  << "\n    [-n_iqs]          " << input_queue_size
                  << "\n    [-fps_sp]         " << fps_sampling_period
                  << "\n    [-n_sp]           " << num_sampling_periods
                  << "\n    [-t]              " << threshold_message
                  << "\n    [-no_show]        " << no_show_message
                  << "\n    [-show_stats]     " << show_statistics
                  << "\n    [-real_input_fps] " << real_input_fps
                  << "\n    [-u]              " << utilization_monitors_message
// ADDED STUFF START
                  << "\n    [-stream]         " << stream_message
                  << "\n    [-dw]             " << display_width_message
                  << "\n    [-dh]             " << display_height_message
                  << "\n    [-mh]             " << mqtt_host_message
                  << "\n    [-mu]             " << mqtt_username_message
                  << "\n    [-mp]             " << mqtt_password_message << '\n';
// ADDED STUFF END
        showAvailableDevices();
        std::exit(0);
    } if (FLAGS_m.empty()) {
        throw std::runtime_error("Parameter -m is not set");
    } if (FLAGS_i.empty()) {
        throw std::runtime_error("Parameter -i is not set");
    } if (FLAGS_duplicate_num == 0) {
        throw std::runtime_error("Parameter -duplicate_num must be positive");
    } if (FLAGS_bs != 1) {
        throw std::runtime_error("Parameter -bs must be 1");
    } if (!FLAGS_mh.empty() && FLAGS_mu.empty()) {
        throw std::runtime_error("Parameter -mu is not set");
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<float> & initialAnchors, const std::vector<T> & mask) {
        anchors.resize(num * 2);
        for (int i = 0; i < num; ++i) {
            anchors[i * 2] = initialAnchors[mask[i] * 2];
            anchors[i * 2 + 1] = initialAnchors[mask[i] * 2 + 1];
        }
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors;

    YoloParams() {}

    YoloParams(const ov::op::v0::RegionYolo& regionYolo) {
        coords = regionYolo.get_num_coords();
        classes = regionYolo.get_num_classes();
        const std::vector<float>& initialAnchors = regionYolo.get_anchors();
        const std::vector<int64_t>& mask = regionYolo.get_mask();
        num = mask.size();

        computeAnchors(initialAnchors, mask);
    }
};

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) :
        xmin{static_cast<int>((x - w / 2) * w_scale)},
        ymin{static_cast<int>((y - h / 2) * h_scale)},
        xmax{static_cast<int>(this->xmin + w * w_scale)},
        ymax{static_cast<int>(this->ymin + h * h_scale)},
        class_id{class_id},
        confidence{confidence} {}

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void parseYOLOOutput(ov::Tensor tensor,
                    const YoloParams &yoloParams, const unsigned long resized_im_h,
                    const unsigned long resized_im_w, const unsigned long original_im_h,
                    const unsigned long original_im_w,
                    const double threshold, std::vector<DetectionObject> &objects) {

    const int height = static_cast<int>(tensor.get_shape()[2]);
    const int width = static_cast<int>(tensor.get_shape()[3]);
    if (height != width)
        throw std::runtime_error("Invalid size of output. It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(height) +
        ", current W = " + std::to_string(height));

    auto num = yoloParams.num;
    auto coords = yoloParams.coords;
    auto classes = yoloParams.classes;

    auto anchors = yoloParams.anchors;

    auto side = height;
    auto side_square = side * side;
    const float* data = tensor.data<float>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = data[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + data[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + data[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(data[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(data[box_index + 2 * side_square]) * anchors[2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * data[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void drawDetections(cv::Mat& img, const std::vector<DetectionObject>& detections, const std::vector<cv::Scalar>& colors) {
    for (const DetectionObject& f : detections) {
        std::string label = cv::format("%.2f", f.confidence);
        if (!class_names.empty())
        {
            CV_Assert(f.class_id < (int)class_names.size());
            label = class_names[f.class_id] + ": " + label;
        }

        cv::rectangle(img,
                      cv::Rect2f(static_cast<float>(f.xmin),
                                 static_cast<float>(f.ymin),
                                 static_cast<float>((f.xmax-f.xmin)),
                                 static_cast<float>((f.ymax-f.ymin))),
                      colors[static_cast<int>(f.class_id)],
                      2);

// ADDED STUFF START
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);

        cv::putText(img, label, cv::Point(f.xmin, f.ymin - labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[static_cast<int>(f.class_id)], 1.5);
// ADDED STUFF END
    }
}

const size_t DISP_WIDTH  = 1920;
const size_t DISP_HEIGHT = 1080;
const size_t MAX_INPUTS  = 25;

struct DisplayParams {
    std::string name;
    cv::Size windowSize;
    cv::Size frameSize;
    size_t count;
    cv::Point points[MAX_INPUTS];
};

DisplayParams prepareDisplayParams(size_t count) {
    DisplayParams params;
    params.count = count;
    params.windowSize = cv::Size(FLAGS_dw, FLAGS_dh);

    size_t gridCount = static_cast<size_t>(ceil(sqrt(count)));
    size_t gridStepX = static_cast<size_t>(FLAGS_dw/gridCount);
    size_t gridStepY = static_cast<size_t>(FLAGS_dh/gridCount);
    if (gridStepX == 0 || gridStepY == 0) {
        throw std::logic_error("Can't display every input: there are too many of them");
    }
    params.frameSize = cv::Size(gridStepX, gridStepY);

    for (size_t i = 0; i < count; i++) {
        cv::Point p;
        p.x = gridStepX * (i/gridCount);
        p.y = gridStepY * (i%gridCount);
        params.points[i] = p;
    }
    return params;
}

void displayNSources(const std::vector<std::shared_ptr<VideoFrame>>& data,
                     const std::string& stats,
                     const DisplayParams& params,
                     const std::vector<cv::Scalar> &colors,
                     Presenter& presenter,
                     PerformanceMetrics& metrics,
                     bool no_show) {
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            drawDetections(windowPart, elem->detections.get<std::vector<DetectionObject>>(), colors);
        }
    };

    auto drawStats = [&]() {
        if (FLAGS_show_stats && !stats.empty()) {
            static const cv::Point posPoint = cv::Point(3*FLAGS_dw/4, 4*FLAGS_dh/5);
            auto pos = posPoint + cv::Point(0, 25);
            size_t currPos = 0;
            while (true) {
                auto newPos = stats.find('\n', currPos);
                putHighlightedText(windowImage, stats.substr(currPos, newPos - currPos), pos,
                    cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,  cv::Scalar(0, 0, 255), 2);
                if (newPos == std::string::npos) {
                    break;
                }
                pos += cv::Point(0, 25);
                currPos = newPos + 1;
            }
        }
    };

//  #ifdef USE_TBB
#if 0  // disable multithreaded rendering for now
    run_in_arena([&](){
        tbb::parallel_for<size_t>(0, data.size(), [&](size_t i) {
            loopBody(i);
        });
    });
#else
    for (size_t i = 0; i < data.size(); ++i) {
        loopBody(i);
    }
#endif
    presenter.drawGraphs(windowImage);
    drawStats();
    for (size_t i = 0; i < data.size() - 1; ++i) {
        metrics.update(data[i]->timestamp);
    }
    metrics.update(data.back()->timestamp, windowImage, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
    if (!no_show) {
        cv::imshow(params.name, windowImage);
    }

// ADDED STUFF START
    if (FLAGS_stream) {
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", windowImage, buff_bgr, stream_params);
        streamer.publish("/detection_output", std::string(buff_bgr.begin(), buff_bgr.end()));
    }
// ADDED STUFF END
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
#if USE_TBB
        TbbArenaWrapper arena;
#endif
        parse(argc, argv);

        const std::vector<std::string>& inputs = split(FLAGS_i, ',');

// ADDED STUFF START
        YAML::Node config;

        config = YAML::LoadFile("./cameras.yaml");

        slog::info << "Cameras in YAML: " << config["cameras"].size() << slog::endl;

        inputs.clear();

        for (std::size_t i=0;i<num_cameras;i++) {
            inputs.push_back(camera["input"].as<std::string>());
        }
// ADDED STUFF END

        DisplayParams params = prepareDisplayParams(inputs.size() * FLAGS_duplicate_num);

// ADDED STUFF START
        if (!FLAGS_mh.empty()) {
            std::string address = FLAGS_mh;

            slog::info << "Connecting to server '" << address << "'..." << slog::endl;

            mqtt::async_client cli(address, MQTT_CLIENT_ID, MAX_BUFFERED_MSGS);

            mqtt::connect_options connOpts;
            connOpts.set_keep_alive_interval(MAX_BUFFERED_MSGS * PERIOD);
            connOpts.set_clean_session(true);
            connOpts.set_automatic_reconnect(true);
            connOpts.set_user_name(FLAGS_mu);
            connOpts.set_password(FLAGS_mp);

            // Connect to the MQTT broker
            cli.connect(connOpts)->wait();
            slog::info << "OK" << slog::endl;
        }

        if (FLAGS_stream) {
            streamer.start(8080);
        }
// ADDED STUFF END

        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        if (model->get_parameters().size() != 1) {
            throw std::logic_error("Face Detection model must have only one input");
        }
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC");
        for (const ov::Output<ov::Node>& out : model->outputs()) {
            ppp.output(out.get_any_name()).tensor().set_element_type(ov::element::f32);
        }
        model = ppp.build();
        ov::set_batch(model, FLAGS_bs);

        std::vector<std::pair<ov::Output<ov::Node>, YoloParams>> yoloParams;
        for (const ov::Output<ov::Node>& out : model->outputs()) {
            const ov::op::v0::RegionYolo* regionYolo = dynamic_cast<ov::op::v0::RegionYolo*>(out.get_node()->get_input_node_ptr(0));
            if (!regionYolo) {
                throw std::runtime_error("Invalid output type: " + std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
            }
            yoloParams.emplace_back(out, *regionYolo);
        }
        std::vector<cv::Scalar> colors;
        if (yoloParams.size() > 0)
            for (int i = 0; i < static_cast<int>(yoloParams.front().second.classes); ++i)
                colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));

        std::queue<ov::InferRequest> reqQueue = compile(std::move(model),
            FLAGS_m, FLAGS_d, roundUp(params.count, FLAGS_bs), core);
        ov::Shape inputShape = reqQueue.front().get_input_tensor().get_shape();
        if (4 != inputShape.size()) {
            throw std::runtime_error("Invalid model input dimensions");
        }
        IEGraph graph{std::move(reqQueue), FLAGS_show_stats};

        VideoSources::InitParams vsParams;
        vsParams.inputs               = inputs;
        vsParams.loop                 = FLAGS_loop;
        vsParams.queueSize            = FLAGS_n_iqs;
        vsParams.collectStats         = FLAGS_show_stats;
        vsParams.realFps              = FLAGS_real_input_fps;
        vsParams.expectedHeight = static_cast<unsigned>(inputShape[2]);
        vsParams.expectedWidth  = static_cast<unsigned>(inputShape[3]);

        VideoSources sources(vsParams);
        sources.start();

        size_t currentFrame = 0;
        graph.start(FLAGS_bs, [&](VideoFrame& img) {
            img.sourceIdx = currentFrame;
            size_t camIdx = currentFrame / FLAGS_duplicate_num;
            currentFrame = (currentFrame + 1) % (sources.numberOfInputs() * FLAGS_duplicate_num);
            return sources.getFrame(camIdx, img);
        }, [&yoloParams](ov::InferRequest req,
                cv::Size frameSize
                ) {
            unsigned long resized_im_h = 416;
            unsigned long resized_im_w = 416;

            std::vector<DetectionObject> objects;
            // Parsing outputs
            for (const std::pair<ov::Output<ov::Node>, YoloParams>& idxParams : yoloParams) {
                parseYOLOOutput(req.get_tensor(idxParams.first), idxParams.second, resized_im_h, resized_im_w, frameSize.height, frameSize.width, FLAGS_t, objects);
            }
            // Filtering overlapping boxes and lower confidence object
            std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
            for (size_t i = 0; i < objects.size(); ++i) {
                if (objects[i].confidence == 0)
                    continue;
                for (size_t j = i + 1; j < objects.size(); ++j)
                    if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
                        objects[j].confidence = 0;
            }

            std::vector<Detections> detections(1);
            detections[0].set(new std::vector<DetectionObject>);

            for (auto &object : objects) {
                if (object.confidence < FLAGS_t)
                    continue;
                detections[0].get<std::vector<DetectionObject>>().push_back(object);
            }

            return detections;
        });

        std::mutex statMutex;
        std::stringstream statStream;

        cv::Size graphSize{static_cast<int>(params.windowSize.width / 4), 60};
        Presenter presenter(FLAGS_u, params.windowSize.height - graphSize.height - 10, graphSize);
        PerformanceMetrics metrics;

        const size_t outputQueueSize = 1;
        AsyncOutput output(FLAGS_show_stats, outputQueueSize,
        [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
            std::string str;
            if (FLAGS_show_stats) {
                std::unique_lock<std::mutex> lock(statMutex);
                str = statStream.str();
            }
            displayNSources(result, str, params, colors, presenter, metrics, FLAGS_no_show);
            int key = cv::waitKey(1);
            presenter.handleKey(key);

            return (key != 27);
        });

        output.start();

        std::vector<std::shared_ptr<VideoFrame>> batchRes;
        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t perfItersCounter = 0;

        while (sources.isRunning() || graph.isRunning()) {
            bool readData = true;
            while (readData) {
                auto br = graph.getBatchData(params.frameSize);
                if (br.empty()) {
                    break;  // IEGraph::getBatchData had nothing to process and returned. That means it was stopped
                }
                for (size_t i = 0; i < br.size(); i++) {
                    // this approach waits for the next input image for sourceIdx. If provided a single image,
                    // it may not show results, especially if -real_input_fps is enabled
                    auto val = static_cast<unsigned int>(br[i]->sourceIdx);
                    auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->sourceIdx == val; } );
                    if (it != batchRes.end()) {
                        output.push(std::move(batchRes));
                        batchRes.clear();
                        readData = false;
                    }
                    batchRes.push_back(std::move(br[i]));
                }
            }

            if (!output.isAlive()) {
                break;
            }

            auto currTime = timer::now();
            auto deltaTime = (currTime - lastTime);
            if (deltaTime >= samplingTimeout) {
                lastTime = currTime;

                if (FLAGS_show_stats) {
                    if (++perfItersCounter >= FLAGS_n_sp) {
                        break;
                    }
                }

                if (FLAGS_show_stats) {
                    std::unique_lock<std::mutex> lock(statMutex);
                    slog::debug << "------------------- Frame # " << perfItersCounter << "------------------" << slog::endl;
                    writeStats(slog::debug, slog::endl, sources.getStats(), graph.getStats(), output.getStats());
                    statStream.str(std::string());
                    writeStats(statStream, '\n', sources.getStats(), graph.getStats(), output.getStats());
                }
            }
        }
        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

// ADDED STUFF START
    if (FLAGS_stream) {
        streamer.stop();
    }
// ADDED STUFF END

    return 0;
}
