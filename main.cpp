#include <depthai/depthai.hpp>
#include <boost/algorithm/clamp.hpp>
#include <opencv2/opencv.hpp>

constexpr auto rgb_str_ = "rgb";
constexpr auto nn_str_ = "nn";

cv::Rect frameNorm(const cv::Mat& frame, const std::vector<float>& bbox) {
    int width = frame.cols;
    int height = frame.rows;
    
    int x1 = static_cast<int>(boost::algorithm::clamp(bbox[0], 0.0f, 1.0f) * width);
    int y1 = static_cast<int>(boost::algorithm::clamp(bbox[1], 0.0f, 1.0f) * height);
    int x2 = static_cast<int>(boost::algorithm::clamp(bbox[2], 0.0f, 1.0f) * width);
    int y2 = static_cast<int>(boost::algorithm::clamp(bbox[3], 0.0f, 1.0f) * height);
    
    return {x1, y1, x2 - x1, y2 - y1};
}

int main() {
    dai::Pipeline pipeline;
    // nodes
    auto cam_rgb = pipeline.create<dai::node::ColorCamera>();
    auto detection_nn = pipeline.create<dai::node::MobileNetDetectionNetwork>();
    
    // outputs
    auto rgb_out = pipeline.create<dai::node::XLinkOut>();
    rgb_out->setStreamName(rgb_str_);
    auto nn_out = pipeline.create<dai::node::XLinkOut>();
    nn_out->setStreamName(nn_str_);
    
    // properties
    cam_rgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    cam_rgb->setInterleaved(false);
    cam_rgb->setPreviewSize(300, 300);
    detection_nn->setBlobPath("model.blob");
    detection_nn->setConfidenceThreshold(0.5);
    
    // linking
    cam_rgb->preview.link(rgb_out->input);
    cam_rgb->preview.link(detection_nn->input);
    detection_nn->out.link(nn_out->input);
    
    // connect to device & start pipeline
    dai::Device device(pipeline);
    auto rgb_queue = device.getOutputQueue(rgb_str_, 8, false);
    auto nn_queue = device.getOutputQueue(nn_str_);
    
    cv::Mat rgb_frame;
    std::vector<dai::ImgDetection> detections;
   
    while (true) {
        const auto in_rgb = rgb_queue->tryGet<dai::ImgFrame>();
        const auto in_nn = nn_queue->tryGet<dai::ImgDetections>();
        
        if (in_rgb) rgb_frame = in_rgb->getCvFrame();
        if (in_nn) detections = in_nn->detections;
        
        if (rgb_frame.rows > 0 && rgb_frame.cols > 0) {
            for (const auto& d : detections) {
                auto bbox = frameNorm(rgb_frame, {d.xmin, d.ymin, d.xmax, d.ymax});
                cv::rectangle(rgb_frame, bbox, cv::Scalar(255, 0, 0), 2);
            }
            
            cv::imshow("preview", rgb_frame);
        }
        
        if (cv::waitKey(1) == 'q') {
            return 0;
        }
    }
    
    return 0;
}
