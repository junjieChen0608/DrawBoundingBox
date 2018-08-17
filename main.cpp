#include <iostream>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

#include <pcl/common/common.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid_covariance.h>

#include <boost/thread/thread.hpp>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/ColorOcTree.h>

/**
 * wrapper class containing info
 * of each cell
 */
struct MyLeaf {
  int pointCount;
  double runningSum, runningAvg;

  MyLeaf() : pointCount(0),
             runningSum(0.0),
             runningAvg(0.0) {}
};

inline int computeSingleCoord(Eigen::Vector3i& divb_mul_, Eigen::Vector3i& min_b_,
                              Eigen::Array3f& inverse_leaf_size, int idx, int i) {
  int ijk;
  if (i == 2) {
    ijk = idx / divb_mul_[i];
  } else {
    ijk = idx % divb_mul_[i + 1] / divb_mul_[i];
  }

  return static_cast<int> (floor((ijk + min_b_[i]) / inverse_leaf_size[i]));
}

inline void
computeCoord(int idx, Eigen::Vector3i &divb_mul_, Eigen::Vector3i &min_b_, Eigen::Array3f &inverse_leaf_size,
             std::unordered_map<int, std::vector<int>> &coordMap) {

  // check if this idx is in the leaf first

  /**
   * solve ijk0
   * idx % divb_mul_[1] / divb_mul_[0]
   * x = floor( (ijk0 + min_b_[0]) / inverse_leaf_size[0] )
   */
  int x = computeSingleCoord(divb_mul_, min_b_, inverse_leaf_size, idx, 0);
  /**
   * solve ijk1
   * idx % divb_mul_[2] / divb_mul_[1]
   * y = floor( ijk1 + min_b_[1] / inverse_leaf_size[1] )
   */
  int y = computeSingleCoord(divb_mul_, min_b_, inverse_leaf_size, idx, 1);

  /**
   * solve ijk2
   * idx / divb_mul_[2]
   * z = floor ( ijk2 + min_b_[2] / inverse_leaf_size[2] )
   */
  int z = computeSingleCoord(divb_mul_, min_b_, inverse_leaf_size, idx, 2);

//  std::cout << "idx: " << idx
//             << " -> (" << x << ", " << y << ", " << z << ")\n";
  coordMap.emplace(idx, std::vector<int>{x, y, z});
}

inline int calculateIndex(pcl::PointXYZ &p,
                          pcl::VoxelGridCovariance<pcl::PointXYZ> &covar,
                          Eigen::Array3f &inverse_leaf_size_,
                          Eigen::Vector3i &min_b_, Eigen::Vector3i &divb_mul_) {

  int ijk0 = static_cast<int> (floor (p.x * inverse_leaf_size_[0]) - min_b_[0]);
  int ijk1 = static_cast<int> (floor (p.y * inverse_leaf_size_[1]) - min_b_[1]);
  int ijk2 = static_cast<int> (floor (p.z * inverse_leaf_size_[2]) - min_b_[2]);
  int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

  return idx;

}

inline double computeProbability(pcl::PointXYZ &p,
                                 pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf &leaf) {

  Eigen::Vector3d& mean = leaf.mean_;
  Eigen::Vector3d diff(p.x - mean[0], p.y - mean[1], p.z - mean[2]);
  Eigen::Matrix<double, 1, 3> diff_transpose = diff.transpose();
  Eigen::Matrix<double, 1, 1> result_mat = diff_transpose * leaf.icov_* diff;

  double result = -result_mat(0, 0) * 0.5;
//  std::cout << "prob " << exp(result) << "\n";
  return exp(result);

}

inline void crossReference(std::unordered_map<int, MyLeaf> &visited,
                           std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf> &leaves,
                           pcl::PointCloud<pcl::PointXYZ> &distCloud) {

  int ones = 0, nonOnes = 0;
  for (std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf>::iterator it = leaves.begin();
       it != leaves.end(); ++it) {
    int idx = it->first;
    int ref_count = it->second.nr_points;
    int actual_count = visited[idx].pointCount;
    if (ref_count != actual_count) {
      std::cout << "point count does not match in index " << idx << "\n";
      break;
    }
    Eigen::VectorXf leafCentroid = it->second.centroid;
//    std::cout << "idx: " << it->first << " " << it->second.nr_points << " pts\n";
    distCloud.points.emplace_back(pcl::PointXYZ(leafCentroid[0], leafCentroid[1], leafCentroid[2]));

//    std::cout << "idx " << idx << ": " << visited[idx].runningAvg << "\n";
    if (visited[idx].runningAvg == 1) {
      ++ones;
    } else {
//      std::cout << "P(" << idx << ") = " << visited[idx].runningAvg << "\n";
      ++nonOnes;
    }
  }

  std::cout << "ones " << ones << ", non ones " << nonOnes
            << "\ndist cloud size " << distCloud.size() << "\n";

}

inline void drawCube(const int &id, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, const float &leafSize,
                     std::vector<int> &coord) {

  std::stringstream ss;
  ss << id;
  int x = coord[0];
  int y = coord[1];
  int z = coord[2];
  viewer->addCube(x, x + leafSize, y, y + leafSize, z, z + leafSize,
                  0.0, 1.0, 0.0, ss.str());

  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                      pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, ss.str());

  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, ss.str());

}

void visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudWithProb,
               std::unordered_map<int, std::vector<int>> &coordMap,
               const float &leafSize) {

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  std::string cloudId = "cloud";
  viewer->addPointCloud(cloudWithProb, cloudId);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.2, cloudId);

  // iterate on each index and draw bounding box according to its coordinate
  for (auto it = coordMap.begin(); it != coordMap.end(); ++it) {
    drawCube(it->first, viewer, leafSize, it->second);
  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

}

inline void initRGBPoint(pcl::PointXYZRGB &newP, pcl::PointXYZ &p, const uint8_t &color) {
  newP.x = p.x;
  newP.y = p.y;
  newP.z = p.z;
  newP.r = color;
  newP.g = color;
  newP.b = color;
}

void generateCoordMap(std::unordered_map<int, std::vector<int>> &coordMap,
                      std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf> &leaves,
                      Eigen::Vector3i &divb_mul_, Eigen::Vector3i &min_b_, Eigen::Array3f &inverse_leaf_size_) {

  for (auto it = leaves.begin(); it != leaves.end(); ++it) {
    computeCoord(it->first, divb_mul_, min_b_, inverse_leaf_size_, coordMap);
  }
}

int main(int argc, char** argv) {

  std::string out_path = "/home/junjie.chen/CLionProjects/pcl_voxelgridcovar/";

  // load target point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud(new pcl::PointCloud<pcl::PointXYZ> ());
  std::string in_file_path = "/media/junjie.chen/data/hitbox/A/200/A_0deg00000.pcd";
//  std::string in_file_path = "room_scan1.pcd";
  pcl::io::loadPCDFile<pcl::PointXYZ>(in_file_path, *targetCloud);
  std::cout << "point cloud size: " << targetCloud->size() << "\n";

  std::vector<pcl::PointXYZ,
              Eigen::aligned_allocator<pcl::PointXYZ>> points = targetCloud->points;

  // initialize a voxel grid covariance
  pcl::VoxelGridCovariance<pcl::PointXYZ> covar;
  float leafSize = 5.0f;
  covar.setLeafSize(leafSize, leafSize, leafSize);
  covar.setInputCloud(targetCloud);
  covar.filter(true);

  std::unordered_map<int, MyLeaf> visited;
  std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf> leaves = covar.getLeaves();
  std::cout << "leaf size " << leafSize << "\nleaves " << leaves.size() << "\n";

  // data needed to compute index of point
  Eigen::Vector3f leaf_size_(covar.getLeafSize());
  Eigen::Array3f inverse_leaf_size_ = Eigen::Array3f::Ones() / leaf_size_.array();
  Eigen::Vector3i min_b_ = covar.getMinBoxCoordinates();
  Eigen::Vector3i divb_mul_ = covar.getDivisionMultiplier();

//  std::cout << "min b: " << min_b_.transpose()
//            << "\ndiv mul: " << divb_mul_.transpose()
//            << "\ninv leaf size: " << inverse_leaf_size_.transpose() << "\n";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudWithProb(new pcl::PointCloud<pcl::PointXYZRGB> ());
  cloudWithProb->points.reserve(targetCloud->points.size());

  // iterate on all points from the target cloud to generate data for MyLeaf
  for (pcl::PointXYZ& p : targetCloud->points) {

    int idx = calculateIndex(p, covar, inverse_leaf_size_, min_b_, divb_mul_);

    if (visited.count(idx) == 0) { // if this index is not visited yet, map this index to a new struct
      visited.emplace(idx, MyLeaf());
    }

    // get the corresponding leaf to which this point belongs
    pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf& refLeaf = leaves[idx];

    // CAUTION BUG: apply the formula to compute probability of point p
    double probability = computeProbability(p, refLeaf);

    // update the point count, running sum, and running avg in MyLeaf
    MyLeaf& currentLeaf = visited[idx];
    ++currentLeaf.pointCount;
    currentLeaf.runningSum += probability;
    currentLeaf.runningAvg = currentLeaf.runningSum / currentLeaf.pointCount;
    // calculate RGB info
    uint8_t color = 255 * probability;
    pcl::PointXYZRGB newP;
    initRGBPoint(newP, p, color);
    cloudWithProb->points.emplace_back(newP);
  }

  std::cout << "visited map size: " << visited.size() << "\n";

  pcl::PointCloud<pcl::PointXYZ> distCloud;
  distCloud.points.reserve(visited.size());
  distCloud.height = 1;
  distCloud.width = visited.size();

  // cross reference the point count of each index in visited map with leaves from covar
  crossReference(visited, leaves, distCloud);

  std::unordered_map<int, std::vector<int>> coordMap;
  generateCoordMap(coordMap, leaves, divb_mul_, min_b_, inverse_leaf_size_);
  visualize(cloudWithProb, coordMap, leafSize);

  return 0;
}