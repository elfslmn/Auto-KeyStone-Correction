
//
// Created by esalman17 on 5.10.2018.
//

#include <royale/LensParameters.hpp>
#include <royale/IDepthDataListener.hpp>
#include "opencv2/opencv.hpp"
#include <mutex>

#include "stdafx.h"
#include "PlaneDetector.h"
#include "Util.h"
#include "Visualizer.h"

using namespace royale;
using namespace std;
using namespace cv;
using namespace ark;

class CamListener : public royale::IDepthDataListener {

const int MARGIN = 10;
// 2 ye bolunce daha iyi sonuç veriyor ama bolmeden zaten yarım fov??
const float hor_fov = 0.328/2; // in radian
const float ver_fov = 0.190/2;

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
    Point2i findPixelCoord(Vec3f xyz);
    void calculateProjectionCornerVectors();

    // Private variables
    uint16_t cam_width, cam_height;
    mutex flagMutex;
    vector<Vec3f> projCornerVectors;
    vector<Vec3f> projCornerXyz;
    Vec3f translation;

    int frame = 0;
    string recordFolder = "record\\";

    PlaneDetector::Ptr planeDetector;
    vector<FramePlane::Ptr> planes;

};
