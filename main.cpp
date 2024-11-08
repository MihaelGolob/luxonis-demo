#include <depthai/depthai.hpp>
#include <boost/algorithm/clamp.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>

constexpr auto rgb_str_ = "rgb";
constexpr auto depth_str_ = "depth";
constexpr auto out_str_ = "out";

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
    auto mono_left = pipeline.create<dai::node::MonoCamera>();
    auto mono_right = pipeline.create<dai::node::MonoCamera>();
    auto depth = pipeline.create<dai::node::StereoDepth>();
    auto point_cloud = pipeline.create<dai::node::PointCloud>();
    
    // outputs
    auto rgb_out = pipeline.create<dai::node::XLinkOut>();
    auto depth_out = pipeline.create<dai::node::XLinkOut>();
    auto out = pipeline.create<dai::node::XLinkOut>();
    
    rgb_out->setStreamName(rgb_str_);
    depth_out->setStreamName(depth_str_);
    out->setStreamName(out_str_);
    
    // properties
    cam_rgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    cam_rgb->setInterleaved(false);
    
    mono_left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    mono_left->setCamera("left");
    mono_right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    mono_right->setCamera("right");
    
    depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
    depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
    depth->setLeftRightCheck(true);
    depth->setExtendedDisparity(false);
    depth->setSubpixel(true);

    // linking
    cam_rgb->preview.link(rgb_out->input);
    mono_left->out.link(depth->left);
    mono_right->out.link(depth->right);
    depth->depth.link(point_cloud->inputDepth);
    depth->disparity.link(depth_out->input);
    point_cloud->outputPointCloud.link(out->input);
    point_cloud->initialConfig.setSparse(true);
    
    // connect to device & start pipeline
    dai::Device device(pipeline);
    auto rgb_queue = device.getOutputQueue(rgb_str_, 8, false);
    auto depth_queue = device.getOutputQueue(depth_str_, 8, false);
    auto out_queue = device.getOutputQueue(out_str_, 8, false);
    
    cv::Mat rgb_frame;
    std::vector<dai::ImgDetection> detections;
    auto pcl_viewer = std::make_unique<pcl::visualization::PCLVisualizer>("Point Cloud Viewer");
   
    while (true) {
        // show rgb image
        const auto in_rgb = rgb_queue->tryGet<dai::ImgFrame>();
        if (in_rgb) rgb_frame = in_rgb->getCvFrame();
        cv::imshow("preview", rgb_frame);
        
        // process depth 
        auto depth_img = depth_queue->get<dai::ImgFrame>();
        auto pcl_msg = out_queue->get<dai::PointCloudData>();
        if (!pcl_msg) std::cerr << "no depth data!" << std::endl;
        
        auto frame = depth_img->getCvFrame();
        frame.convertTo(frame, CV_8UC1, 255.0f / depth->initialConfig.getMaxDisparity());
        cv::imshow("depth", frame);
        
        if (pcl_msg->getPoints().empty()) std::cerr << "Empty point cloud!" << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pcl_msg->getPclData(); 
        pcl_viewer->updatePointCloud(cloud, "cloud");
        pcl_viewer->spinOnce(10);
        
        if (cv::waitKey(1) == 'q') {
            return 0;
        }
    }
    
    return 0;
}
