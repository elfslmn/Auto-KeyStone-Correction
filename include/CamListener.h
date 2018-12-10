
//
// Created by esalman17 on 5.10.2018.
//

#include <royale/LensParameters.hpp>
#include <royale/IDepthDataListener.hpp>
#include "opencv2/opencv.hpp"
#include <mutex>

#include "PlaneDetector.h"

using namespace royale;
using namespace std;
using namespace cv;
using namespace ark;

class CamListener : public royale::IDepthDataListener {

const int MARGIN = 10;

const float hor_fov = 0.328; // in radian
const float ver_fov = 0.190;
const int proj_width = 1280;
const int proj_height = 720;

public:
    // Constructors
    CamListener();
    void initialize(uint16_t width, uint16_t height);
    void setLensParameters (LensParameters lensParameters);
    void startRecord(string destFolder);
    void stopRecord();
    void processImages();
    void saveCenterPoint();
    void calculateProjectionAxis();

    Mat xyzMap, confMap;
    Mat grayImage;
    Mat depthImage8, grayImage8;
    Mat cameraMatrix, distortionCoefficients;
    vector<Vec3f> centers;
    Vec6f projAxis;

    bool isRecording = false;

private:
    // Private methods
    void onNewData (const DepthData *data);
    void updateImages(const DepthData* data, Mat & depth, Mat & gray, int min_confidence = 0, bool flip=false);
    bool visualizeImage(const Mat & src, Mat & dest, float resize_factor = 1.0, bool color=false);
    void updateMaps(const DepthData* data, bool flip=false);
    bool saveFrame(int frame);
    void setChannel(Mat & xyzMap, Mat & zChannel);
    void calculateProjCornerPos(Vec3f plane);
    Point2i distort(Point2i point);
    Point2i findPixelCoord(Point3f p);
    void calculateProjectionCornerVectors();
    bool correctKeyStone(FramePlane::Ptr plane);
    void showPoint(const Point3f p);

    // Private variables
    uint16_t cam_width, cam_height;
    mutex flagMutex;
    vector<Vec3f> projCornerVectors;
    vector<Point3f> projCornerXyz;
    Vec3f translation;
    Mat image;

    int frame = 0;
    string recordFolder = "record\\";

    PlaneDetector::Ptr planeDetector;
    vector<FramePlane::Ptr> planes;
    FramePlane::Ptr mainPlane;

};
