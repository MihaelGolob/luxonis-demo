#include <depthai/depthai.hpp>
#include <boost/algorithm/clamp.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

constexpr auto rgb_str_ = "rgb";
constexpr auto depth_str_ = "depth";
constexpr auto out_str_ = "out";

// filter parameters [mm]
constexpr auto min_z_ = 50;
constexpr auto max_z_ = 500; 
constexpr auto down_sample_size_ = 10.0f;

cv::Rect frameNorm(const cv::Mat& frame, const std::vector<float>& bbox) {
    int width = frame.cols;
    int height = frame.rows;
    
    int x1 = static_cast<int>(boost::algorithm::clamp(bbox[0], 0.0f, 1.0f) * width);
    int y1 = static_cast<int>(boost::algorithm::clamp(bbox[1], 0.0f, 1.0f) * height);
    int x2 = static_cast<int>(boost::algorithm::clamp(bbox[2], 0.0f, 1.0f) * width);
    int y2 = static_cast<int>(boost::algorithm::clamp(bbox[3], 0.0f, 1.0f) * height);
    
    return {x1, y1, x2 - x1, y2 - y1};
}

cv::Rect frameNorm(const cv::Mat& frame, const cv::Rect& bbox) {
    int width = frame.cols;
    int height = frame.rows;

    int x1 = static_cast<int>(boost::algorithm::clamp(bbox.x, 0.0f, 1.0f) * width);
    int y1 = static_cast<int>(boost::algorithm::clamp(bbox.y, 0.0f, 1.0f) * height);
    int x2 = static_cast<int>(boost::algorithm::clamp(bbox.width, 0.0f, 1.0f) * width);
    int y2 = static_cast<int>(boost::algorithm::clamp(bbox.height, 0.0f, 1.0f) * height);

    return {x1, y1, x2, y2};
}

cv::Rect detectCylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Compute normals for the point cloud
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // Create a KDTree to search for nearest neighbors
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03);  // Adjust based on your data and cylinder size

    // Estimate normals
    ne.compute(*cloud_normals);

    // Create a SACSegmentationFromNormals object for cylinder segmentation
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.01);  // Adjust based on your data
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(1000);    // Adjust based on your data
    seg.setRadiusLimits(0.0001, 10000);    // Set min and max radius for the cylinder

    // Set the input cloud and normals
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setInputCloud(cloud);
    seg.setInputNormals(cloud_normals);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cout << "No cylinder found." << std::endl;
        return cv::Rect();  // Return an empty rect if no cylinder is found
    }
    std::cout << "Cylinder found." << std::endl;

//    // Extract the points corresponding to the detected cylinder
//    pcl::ExtractIndices<pcl::PointXYZ> extract;
//    extract.setInputCloud(cloud);
//    extract.setIndices(inliers);
//    extract.setNegative(false);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cylinderCloud(new pcl::PointCloud<pcl::PointXYZ>);
//    extract.filter(*cylinderCloud);
//
//    // Use getMinMax3D to find the bounding box in 3D space
//    pcl::PointXYZ minPt, maxPt;
//    pcl::getMinMax3D(*cylinderCloud, minPt, maxPt);
//
//    // Convert the 3D bounding box to 2D (x and y) assuming a top-down projection
//    int x = static_cast<int>(minPt.x);
//    int y = static_cast<int>(minPt.y);
//    int width = static_cast<int>(maxPt.x - minPt.x);
//    int height = static_cast<int>(maxPt.y - minPt.y);
//
//    return {x, y, width, height}; 

return {};
}

void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_z_, max_z_);  // Smaller Z-range for closer objects
    pass.filter(*cloud);
}

void downSamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(down_sample_size_, down_sample_size_, down_sample_size_);
    voxel_grid.filter(*cloud);
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
    depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
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
    auto pcl_viewer = std::make_unique<pcl::visualization::PCLVisualizer>("cloud");
    bool first_pcl_update = true;
   
    while (true) {
        // process point cloud
        auto pcl_msg = out_queue->get<dai::PointCloudData>();
        if (!pcl_msg) std::cerr << "no depth data!" << std::endl;
        std::cout << "min z: " << pcl_msg->getMinZ() << " max z: " << pcl_msg->getMaxZ() << std::endl;
        
        if (pcl_msg->getPoints().empty()) std::cerr << "Empty point cloud!" << std::endl;
        else std::cout << "Point cloud size: " << pcl_msg->getPoints().size() << std::endl;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pcl_msg->getPclData(); 
        filterPointCloud(cloud);
        std::cout << "Filtered cloud size: " << cloud->points.size() << std::endl;
        downSamplePointCloud(cloud);
        std::cout << "Down-sampled cloud size: " << cloud->points.size() << std::endl;
        
        const auto detected_cylinder = detectCylinder(cloud);

        // update all the windows
        // rgb
        const auto in_rgb = rgb_queue->tryGet<dai::ImgFrame>();
        if (in_rgb) rgb_frame = in_rgb->getCvFrame();
        cv::imshow("preview", rgb_frame);
        
        // disparity
        auto depth_img = depth_queue->get<dai::ImgFrame>();
        auto frame = depth_img->getCvFrame();
        frame.convertTo(frame, CV_8UC1, 255.0f / depth->initialConfig.getMaxDisparity());
        cv::imshow("depth", frame);

        // point cloud
        if (first_pcl_update) {
            pcl_viewer->addPointCloud(cloud, "cloud");
            first_pcl_update = false;
        } else {
            pcl_viewer->updatePointCloud(cloud, "cloud");
        }
        pcl_viewer->spinOnce(10);
        
        if (cv::waitKey(1) == 'q') {
            return 0;
        }
    }
    
    return 0;
}
