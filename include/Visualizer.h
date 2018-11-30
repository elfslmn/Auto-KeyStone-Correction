#pragma once

#include "Version.h"
#include <vector>
#include <opencv2/core.hpp>

// OpenARK headers
#include "Hand.h"

namespace ark {
    /**
    * Utility class containing various conversions and visualization techniques.
    */
    class Visualizer
    {
    public:
        /**
        * Visualize a (single channel) depth map
        * @param depth_map the depth map
        * @param [out] output output image
        * @return visualization
        */
        static void visualizeDepthMap(const cv::Mat & depth_map, cv::Mat & output);

        /**
        * Visualization for xyz maps (per-pixel point clouds).
        * @param [in] xyz_map input point cloud matrix
        * @param [out] output output image
        * @return a CV_8UC3 representation of the xyz map
        */
        static void visualizeXYZMap(const cv::Mat &xyz_map, cv::Mat & output);

        /**
        * Visualization for normal maps (normalized surface normal vector at each point).
        * @param [in] normal input normal map
        * @param [out] output output image
        * @param resolution resolution of normal map
        * @return a CV_8UC3 representation of the normal map
        */
        static void visualizeNormalMap(const cv::Mat &normal_map, cv::Mat & output,
            int resolution = 3);

        /**
        * Visualization for hand object.
        * @param [in] background the base image to draw on
        * @param [out] output output image
        * @param [in] hand the hand object
        * @param display value to display on the hand; set to >= FLT_MAX to disable
        * @param [in] touch_planes optionally, planes in the current frame
        *                          that the hand may contact
        * @return a CV_8UC3 matrix with the hand drawn on it
        */
        static void visualizeHand(const cv::Mat & background, cv::Mat & output,
            Hand * hand, double display = FLT_MAX,
            const std::vector<std::shared_ptr<FramePlane> > * touch_planes = nullptr);

         static void visualizeHandDebug(const cv::Mat & background, cv::Mat & output,
            Hand * hand, double display = FLT_MAX,
            const std::vector<std::shared_ptr<FramePlane> > * touch_planes = nullptr);

        /**
        * Visualization for plane regression.
        * @param [in] input_mat the base xyzMap on which to draw the visualization
        * @param [out] output output image
        * @param [in] equation equation of the plane
        * @param threshold maximum error distance (mm) allowed for points to be considered covered by regression equation
        * @param clicked whether the finger is currently current contacting the regression equation. Default is FALSE
        * @return a CV_8UC3 representation of the matrix with the regression plane drawn
        */
        static void visualizePlaneRegression(const cv::Mat & input_mat, cv::Mat & output,
                    std::vector<double> &equation, const double threshold, bool clicked = false);

        /**
        * Visualize points that lie on the plane.
        * @param input_mat the input point cloud
        * @param indicies (i,j) coordinates of the points belonging to the plane
        */
        static void visualizePlanePoints(cv::Mat &input_mat, std::vector<Point2i> indicies);

    private:
        /**
        * Visualization for a generic matrix.
        * @param [in] input matrix to be visualized
        * @param [out] output output image
        * @return a CV_8UC3 representation of the input matrix
        */
        static void visualizeMatrix(const cv::Mat & input, cv::Mat & output);

    };
}
