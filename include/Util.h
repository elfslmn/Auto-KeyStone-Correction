#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "Version.h"

namespace ark {
    /**
     * Namespace containing generic helper functions.
     */
    namespace util
    {
        /**
        * Splits a string into components based on delimeter
        * @param string_in string to split
        * @param delimeters c_str of delimeters to split at
        * @return vector of string components
        */
        std::vector<std::string> split(char * string_in, char const * delimeters = " ");

        /**
         * automatically pluralize a string (add 's') base on a given quantity.
         * @param str [in] the string (will not be modified)
         * @param num the quantity (adds 's' if this is not 1)
         * @return pluralized string
         */
        template<class T>
        std::string pluralize(std::string str, T num);

        /**
        * Generates a random RGB color.
        * @return random RGB color in Vec3b format
        */
        Vec3b randomColor();

        /**
        * Get the RGB color at index 'index' of the default palette
        * @param color_index index of color
        * @return RGB color in Vec3b format
        */
        Vec3b paletteColor(uchar color_index);

        /*
        * Compute the euclidean distance between (x1,y1) and (x2,y2)
        * @param pt1 (x1, y1)
        * @param pt2 (x2, y2)
        * @return the euclidean distance between the two points
        */
        template<class T>
        float euclideanDistance(const cv::Point_<T> & pt1, const cv::Point_<T> & pt2);

        /**
        * Get euclidean distance between two 3D points.
        * @param pt1 point 1
        * @param pt2 point 2
        * @return euclidean distance
        */
        template<class T>
        T euclideanDistance(const cv::Vec<T, 3> & pt1, const cv::Vec<T, 3> & pt2);

        /**
        * Compute the closest distance between a point and a plane
        * Where the plane is defined as: ax + by - z + c = 0
        * @param pt the point
        * @param eqn equation of plane in form: [a, b, c]
        * @return euclidean distance in meters
        */
        template<class T> T pointPlaneDistance(const cv::Vec<T, 3> & pt, const cv::Vec<T, 3> & eqn);

        /**
        * Compute the closest distance between a point and a plane
        * Where the plane is defined as: ax + by - z + c = 0
        * @param pt the point
        * @param a, b, c parameters of plane
        * @return euclidean distance in meters
        */
        template<class T> T pointPlaneDistance(const cv::Vec<T, 3> & pt, T a, T b, T c);

        /**
        * Compute the euclidean norm between a point and a plane
        * Where the plane is defined as: ax + by - z + c = 0
        * @param pt the point
        * @param eqn equation of plane in form: [a, b, c]
        * @return euclidean norm in m^2
        */
        template<class T> T pointPlaneNorm(const cv::Vec<T, 3> & pt, const cv::Vec<T, 3> & eqn);

        /**
        * Compute the euclidean norm (distance squared) between a point and a plane
        * Where the plane is defined as: ax + by - z + c = 0
        * @param pt the point
        * @param a, b, c parameters of plane
        * @return euclidean norm in m^2
        */
        template<class T> T pointPlaneNorm(const cv::Vec<T, 3> & pt, T a, T b, T c);

        /**
        * Estimate the euclidean distance per pixel around a point on a depth map
        * @param xyz_map the input depth map
        * @param pt the point
        * @param radius radius to average distance per pixel
        * @return distance per pixel
        */
        double euclideanDistancePerPixel(cv::Mat xyz_map, Point2i pt, int radius);

        /**
        * Removes points on img with the given indicies
        * @param [out] img the image to be operated on
        * @param points list of indicies (i,j) to be set to 0
        */
        void removePoints(cv::Mat & img, const std::vector<Point2i> & points);

        /**
         * Zeros all pixels on 'image' corresponding to points on 'ref_cloud' fitting 'plane_equation'
         * @param [in, out] xyz_map the input point cloud
         * @param [in] plane_equation the equation of the plane
         * @param threshold the thickness of the plane,
         *        i.e. the maximum distance from the plane to a removed point
         * @param mask, mask_color optional mask matrix whose value must equal 'mask_color'
         *                         at the correspoinding index
         *                         for a point to be removed from the point cloud
         */
        template <class T>
        void removePlane(const cv::Mat & ref_cloud, cv::Mat & image, const Vec3f & plane_equation,
                         float threshold, cv::Mat * mask = nullptr, uchar mask_color = 0);

        /**
        * Average all non-zero values around a point.
        * @param img base image to use
        * @param pt the point of interest
        * @param radius number of neighboring points to be used for computing the average
        * @return average (x,y,z) value around the point of interest
        */
        Vec3f averageAroundPoint(const cv::Mat & img, const Point2i & pt, int radius = 5);

        /**
        * Find the approximate surface normal vector at a point on an XYZ map by computing
        * the cross product of two vectors to nearby points.
        * @param img base image to use
        * @param pt the point of interest
        * @param radius length of vectors to use for computing the cross product
        * @return normalized normal vector (the one facing viewer) at the point of interest
        */
        Vec3f normalAtPoint(const cv::Mat & img, const Point2i & pt, int radius = 3);

        /**
        * Eliminate outliers in a point cloud by considering the 'influence' of each point
        * @param data input points
        * @param output output points (should be empty)
        * @param thresh fraction of points to eliminate
        * @param data_aux optionally, auxilliary 2D input vector to reorder along with 'data'
        * @param output_aux optionally, auxilliary 2D output vector to reorder along with 'data'
        * @param num_points number of points in 'data' to use, defaults to all.
        * @return number of output points = floor('num_points' * (1.0 - 'thresh'))
        */
        int removeOutliers(const std::vector<Vec3f> & data,
                               std::vector<Vec3f> & output, double thresh = 0.3,
                               const std::vector<Point2i> * data_aux = nullptr,
                               std::vector<Point2i> * output_aux = nullptr,
                               int num_points = -1);

        /**
          * Perform linear regression on a set of points
          * @param points [in] vector of points
          * @param num_points number of input points to use. By default, uses all.
          * @return the linear least squares equation for the data points,
          *         of the form [a_1, a_2, ..., a_n]:
          *         0 = a_1*x_1 + a_2*x_2 + ... + a_{n-1}*x_{n-1} - x_n + a_n
         */
        template<class T, int N>
        cv::Vec<T, N> linearRegression(const std::vector<cv::Vec<T, N> > & points,
                                       int num_points = -1);

        /**
          * Perform RANSAC-based robust plane regression on a set of points
          * @param points [in] vector of 3D points (must contain at least 3 points)
          * @param thresh maximum euclidean norm (r^2) from a point for consideration as an inlier
          * @param iterations number of RANSAC iterations
          * @param num_points number of input points to use (min 3). By default, uses all.
          * @return best-fitting plane through the given data points, in the form
          *         [a,b,c]: 0 = ax + by - z + c
         */
        template<class T>
        cv::Vec<T, 3> ransacFindPlane(const std::vector<cv::Vec<T, 3> > & points,
            T thresh, int iterations = 300, int num_points = -1);

        /**
        * Compute the surface normal vectors associated with each point in a point cloud
        * @param [in] xyz_map input point cloud
        * @param output_mat output normal matrix
        * @param normal_dist distance at which surface vectors are sampled to compute the normal
        * @param resolution pixel resolution of output normal matrix
        * @param fill_in if true, fills in all pixels of output matrix by copying
        *                else, only fills pixels at interval 'resolution'
        */
        void computeNormalMap(const cv::Mat & xyz_map, cv::Mat & output_mat,
            int normal_dist = 6, int resolution = 2, bool fill_in = true);

        /**
        * Determine whether (x,y) is a non-zero point in the matrix.
        * @param xyz_map Input image
        * @param x, y coordinates of the point
        * @return true if (x.y) is non-zero
        */
        bool isMember(cv::Mat xyz_map, int x, int y);

        /**
        * Find the average depth of a depth image
        * @param xyz_map depth image
        * @return average depth in meters
        */
        double averageDepth(cv::Mat xyz_map);

        /**
        * Find the centroid of the point cloud
        * @param xyz_map input point cloud
        * @return (x,y) coordinate of the centroid
        */
        Point2i findCentroid(cv::Mat xyz_map);

        /**
         * Performs a single floodfill on a depth map starting from seed point (x,y).
         * @param [in] xyz_map the input point cloud
         * @param seed seed point
         * @param thresh maximum euclidean distance allowed between neighbors
         * @param [out] output_ij_points optionally, pointer to a vector for storing ij coords of the points in the component.
         * @param [out] output_xyz_points optionally, pointer to a vector for storing xyz coords of the points in the component.
         * @param [out] output_mask optional output matrix for storing points visited by the floodfill (set to NULL to disable)

         * @param interval1 interval to adjacent points (e.g. if 2, adjacent points are (-2, 0), (0, 2), etc.)
         * @param interval2 additional interval to adjacent points (0 = not used), only works for up/down fill
         * @param interval2_thresh distance theshold for interval2
         * @param [in, out] color an auxiliary matrix (CV_8U)
         *             for recording if a point is being visited (1), has already been visited (0)
         *             or is not yet visited (255)
         *             By default, allocates a temporary matrix for use during flood fill.
         * @return number of points in component
         */
        int floodFill(const cv::Mat & xyz_map, const Point2i & seed,
            float thresh = 0.005f,
            std::vector <Point2i> * output_ij_points = nullptr,
            std::vector <Vec3f> * output_xyz_points = nullptr,
            cv::Mat * output_mask = nullptr,
            int interval1 = 1, int interval2 = 0, float interval2_dist = 0.05f,
            cv::Mat * color = nullptr);

        /**
        * Compute the angle in radians 'pointij' is at from the origin, going CCW starting from (0, 1), if y-axis is facing up.
        * @param pointij input point in ij coordinates
        * @return angle from origin, CCW from (0, 1)
        */
        double pointToAngle(const Point2f & pointij);

        /**
        * Compute the point with unit magnitude 'angle' radians CCW from (0, 1), if y-axis is facing up
        * @param angle input angle
        * @return point with unit magnitude 'angle' radians CCW from (0, 1)
        */
        Point2f angleToPoint(double angle);

        /**
        * Compute the angle in radians between two points in ij coordinates, optionally through a third point (defaults to origin)
        * @param a, b the points, in ij oordinates
        * @param center optional center point (angle goes from a - center - b)
        * @return angle between points
        */
        double angleBetweenPoints(const Point2f & a, const Point2f & b, const Point2f & center = Point2f(0, 0));

        /**
        * Normalize a point by dividing it by its magnitude
        * @param pt input point
        * @return normalized point (if pt is (0, 0), returns pt without modifying it)
        */
        Point2f normalize(const Point2f & pt);

        /**
        * Normalize a vector and make sure it points towards the viewer (negative z)
        * @param vec input vector
        * @return normalized vector
        */
        Vec3f normalize(const Vec3f & vec);

        /**
         * Compute the magnitude of a point.
         * @param pt input point
         * @return magnitude of point
         */
        template <class T>
        double magnitude(cv::Point_<T> pt);

        /**
         * Compute the magnitude of a point.
         * @param pt input point
         * @return magnitude of point
         */
        template <class T>
        double magnitude(cv::Point3_<T> pt);

        /**
         * Compute the magnitude of a point.
         * @param pt input point
         * @return magnitude of point
         */
        template <class T, int n>
        double magnitude(cv::Vec<T, n> pt);

        /**
         * Compute the L2 norm of a point.
         * @param pt input point
         * @return norm of point
         */
        template <class T>
        double norm(cv::Point_<T> pt);

        /**
         * Compute the L2 norm of a point.
         * @param pt input point
         * @return norm of point
         */
        template <class T>
        double norm(cv::Point3_<T> pt);

        /**
         * Compute the L2 norm of a vector.
         * @param pt input point
         * @return norm of point
         */
        template <class T, int n>
        double norm(cv::Vec<T, n> pt);

        /**
         * Compute the angle between two 3D vectors.
         * @param a, b the vectors
         * @param center optionally, vector to subtract both a and b by before computing angle
         * @return angle between vectors
         */
        double angleBetween3DVec(Vec3f a, Vec3f b, Vec3f center = Vec3f(0, 0, 0));

        /**
         * Checks if a point is within the bounds of an image.
         * @param img the image
         * @param pt the point
         */
        bool pointInImage(const cv::Mat & img, const Point2i pt);

        /**
         * Checks if a point is within the bounds of a rectangle.
         * @param rect the rectangle
         * @param pt the point
         */
        bool pointInRect(const cv::Rect & rect, const Point2i pt);

        /**
         * Checks if a point is on the edge of a rectangle with top left point at (0,0) and size 'size'
         * @param size the size object
         * @param pt the point
         * @param margin_tb max number of pixels from the top/bottom edges to be considered on edge
         * @param margin_lr max number of pixels from the left/right edges to be considered on edge
         */
        bool pointOnEdge(const cv::Size size, const Point2i pt,
            int margin_tb = 30, int margin_lr = 30);

        /**
         * Checks if a point is on the edge of a rectangle
         * @param rect the rectangle
         * @param pt the point
         * @param margin_tb max number of pixels from the top/bottom edges to be considered on edge
         * @param margin_lr max number of pixels from the left/right edges to be considered on edge
         * @param scale the scale of the point's coordinates relative to the scale of the image
         */
        bool pointOnEdge(const cv::Rect rect, const Point2i pt,
            int margin_tb = 30, int margin_lr = 30);

        /**
         * Checks if a point is on the edge of an image
         * @param [in] img the image
         * @param pt the point
         * @param margin_tb max number of pixels from the top/bottom edges to be considered on edge
         * @param margin_lr max number of pixels from the left/right edges to be considered on edge
         */
        bool pointOnEdge(const cv::Mat & img, const Point2i pt,
            int margin_tb = 30, int margin_lr = 30);

        /**
         * Computes the area of the triangle defined by three vertices
         * @param a first vertex
         * @param b second vertex
         * @param c third vertex
         * @return area of triangle, in real meters squared
         */
        float triangleArea(Vec3f a, Vec3f b, Vec3f c = Vec3f(0, 0, 0));

        /**
         * Computes the area of the quadrangle defined by four vertices
         * @param pts the vertices of the quadrangle
         * @return area of quadrangle, in real meters squared
         */
        float quadrangleArea(Vec3f pts[4]);

        /**
         * Computes the approximate surface area of all visible clusters on a depth map.
         * @param [in] depthMap the input depth map. All points with 0 z-coordinate will be excluded.
         * @return surface area, in meters squared
         */
        double surfaceArea(const cv::Mat & depthMap);

        /**
         * Computes the approximate surface area of a cluster containing the specified points
         * @param [in] frame_size size of the image frame
         * @param [in] points_ij ij coords of points in the cluster. Assumed to be ordered by y, then by x.
         * @param [in] points_xyz xyz coords of points in the cluster. Assumed to be in the same order as points_ij.
         * @param [in] sorted if true, assumes that 'cluster' is sorted and does not sort it again
         * @param [in] cluster_size number of points in this cluster. By default, uses all points in the 'cluster' vector.
         * @return surface area, in meters squared
         */
        double surfaceArea(const cv::Size & frame_size,
            const std::vector<Point2i> & points_ij,
            const std::vector<Vec3f> & points_xyz,
            int cluster_size = -1);

        /**
          * Approximates the surface area of a depth map cluster, using circles
          * of the smallest possible radius among adjacent points.
          * @param shape the depth map input
          * @return surface area, meters squared
          */
        double surfaceAreaCircle(cv::Mat shape);

        /**
          * Approximates the surface area of a depth map cluster by triangulation
          * among adjacent points
          * @param shape the depth map input
          * @return surface area, meters squared
          */
        double surfaceAreaTriangulate(cv::Mat shape);

        /**
        * compute the diameter of a set of 2D points.
        * outputs the indices of the furthest points and returns the distance between them.
        * @param[in] points the input points
        * @param[out] a, b outputs the indices of the furthest points
        * @return 2D euclidean norm (distance^2) between the two points
        */
        double diameter(const std::vector<cv::Point> & points, int & a, int & b);

        /**
          * Sort points by y and then x coordinate using radix sort
          * @param points[in] vector of points to sort
          * @param width, height width and height of overall depth image
          * @param num_points number of points in'points' to use. By default, uses all.
          * @param points_xyz pointer to vector of associated xyz points. If provided, sorted along with ij points
         */
        void radixSortPoints(std::vector<Point2i> & points,
            int width, int height,
            int num_points = -1,
            std::vector<Vec3f> * points_xyz = nullptr);

        /**
         * Find a nonzero point on 'cluster' close to 'starting_point' by searching in a spiral from the starting point.
         * Used for snapping computed centroids, etc. to actual points on the object.
         * If the value at 'starting_point' is nonzero, then returns 'starting_point' without proceeding.
         * @param cluster the depth map representing the cluster
         * @param starting_point startin point of search
         * @param max_attempts maximum number of points to look at before cutting off the
                            search and simply returning the starting point
         * @return first nonzero point on 'cluster' close to 'starting_point' encountered by travelling in a spiral
         */
        Point2i nearestPointOnCluster(const cv::Mat cluster, Point2i starting_point, int max_attempts = 500);

        /**
         * Find the center and radius of the largest inscribed circle within a contour
         * @param[in] contour the input contour
         * @param[in] xyz_map the input point cloud
         * @param bounds the bounds of the object
         * @param top_point top point, in xyz coordinates, or the cluster
         * @param top_dist_thresh maximum distance between the center of the circle and the top point in the cluster
         * @param[out] radius output pointer for the radius of circle
         * @param samples approximate number of points to sample from the contour
         * @return the center of the circle
         */
         Point2f largestInscribedCircle(const std::vector<Point2i> & contour,
            const cv::Mat & xyz_map,
            const cv::Rect bounds,
            const Vec3f top_point = Vec3f(0, 0,0),
            float top_dist_thresh = FLT_MAX,
            double * radius = nullptr, int samples = 100);

        /**
         * Find the approximate curvature of a contour (1/R) near the specified point
         * @param[in] contour the input contour
         * @param index the index of the target point within the contour
         * @param float the 2D euclidean distance from the target point to points for sampling derivatives
         * @param max_tries maximum number of attempts to find the side points. set to -1 to disable.
         * @return curvature in radians at the point
         */
        float contourCurvature(const std::vector<Point2i> & contour, int index,
            float radius = 30.0, int max_tries = 60);

        /**
         * Find the angle of curvature of a contour in radians near the specified point
         * @param[in] contour the input contour
         * @param index the index of the target point within the contour
         * @param start the number of points from the target point to begin averaging curvature
         * @param end the number of points from the target point to stop averaging curvature
         * @return angle in radians at the point
         */
        float contourLocalAngle(const std::vector<Point2i> & contour, int index,
            int start = 2, int end = 5);

        /** find the 2D distance, in pixels, between a given point and the farthest point in a given direction
          * that has a nonzero value on an xyz map.
          * @param xyz_map the XYZ map (point cloud)
          * @param center the center point
          * @param angle the direction, specified by an angle CCW from (0, 1) if y-axis is facing up
          * @param angle_offset the offset angle to be added to angle (default 0.0)
          */
        float radiusInDirection(const cv::Mat & xyz_map, const Point2i & center,
                             double angle, double angle_offset = 0.0);

         /**
         * myRansac
         */
         Vec3f myRansac(const std::vector<Vec3f>& points, const std::vector<Point2i>& indices,
             float thresh, int iterations, int num_points=-1);

        /**
         * Compares two points (Point, Point2f, Vec3i or Vec3f),
         * first by x, then y, then z (if available). Used for sorting points.
         */
        template<class T>
        class PointComparer {
        public:
            PointComparer(bool reverse = false, bool compare_y_then_x = false)
                : reverse(reverse), compare_y_then_x(compare_y_then_x){ }

            // Compare two points. Returns true if a is less than b.
            bool operator()(T a, T b);
        private:
            bool reverse = false, compare_y_then_x = false;
        };
    };
}
