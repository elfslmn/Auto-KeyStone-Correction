//
// Created by esalman17 on 5.10.2018.
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"

#include "CamListener.h"

#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define LOGD(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define LOGE(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)


CamListener::CamListener(){}

void CamListener::initialize(uint16_t width, uint16_t height)
{
    cam_height = height;
    cam_width = width;
    grayImage.create (Size (cam_width,cam_height), CV_16UC1);
    xyzMap.create(Size (cam_width,cam_height), CV_32FC3);
    confMap.create(Size (cam_width,cam_height), CV_8UC1);
    
    planeDetector = std::make_shared<PlaneDetector>();

    LOGD("Cam listener initialized with (%d,%d)", width, height);
}

void CamListener::setLensParameters (LensParameters lensParameters)
{
    // Construct the camera matrix
    // (fx   0    cx)
    // (0    fy   cy)
    // (0    0    1 )
    cameraMatrix = (Mat1d (3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
            0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
            0, 0, 1);
    /*LOGI("Camera params fx fy cx cy: %f,%f,%f,%f", lensParameters.focalLength.first, lensParameters.focalLength.second,
           lensParameters.principalPoint.first, lensParameters.principalPoint.second); */

    // Construct the distortion coefficients
    // k1 k2 p1 p2 k3
    distortionCoefficients = (Mat1d (1, 5) << lensParameters.distortionRadial[0],
            lensParameters.distortionRadial[1],
            lensParameters.distortionTangential.first,
            lensParameters.distortionTangential.second,
            lensParameters.distortionRadial[2]);
    /*LOGI("Dist coeffs k1 k2 p1 p2 k3 : %f,%f,%f,%f,%f", lensParameters.distortionRadial[0],
         lensParameters.distortionRadial[1],
         lensParameters.distortionTangential.first,
         lensParameters.distortionTangential.second,
         lensParameters.distortionRadial[2]); */
}


void CamListener::onNewData (const DepthData *data)
{
    lock_guard<mutex> lock (flagMutex);
	updateMaps(data);
    
    if(isRecording)
    {
    	saveFrame(frame);
    	frame++;
    }
    
    processImages();
}

void CamListener::processImages()
{
	if(visualizeImage(grayImage, grayImage8, 1.0))
    {
    	imshow("Gray", grayImage8);
    }
    vector<Mat> channels(3);
    split(xyzMap, channels);
    medianBlur(channels[2], channels[2], 3); // 7 for low exposure
    setChannel(xyzMap, channels[2]);
    if(visualizeImage(channels[2], depthImage8, 1.0, true))
    {
    	imshow("Depth", depthImage8);
    }
    
    /*planeDetector->update(xyzMap);
    planes = planeDetector -> getPlanes();
	for(auto plane: planes){
		auto eq = plane->getNormalVector();
		LOGD("Normal: (%.3f, %.3f, %.3f)", eq[0], eq[1], eq[2]);
	}
    
    Mat normalMap = planeDetector -> getNormalMap();
    normalize (normalMap, normalMap, 0, 255, NORM_MINMAX, CV_8UC3);
    resize(normalMap, normalMap, normalMap.size()*3,0, 0, INTER_NEAREST);
    imshow("NormalMap", normalMap); */
    
    
}

void CamListener::updateImages(const DepthData* data, Mat & depth, Mat & gray, int min_confidence, bool flip)
{
    bool isDepth =true , isGray = true;
    if(depth.empty()) {isDepth = false; LOGD("depth not found");}
    if(gray.empty()) {isGray = false; LOGD("gray not found");}
    if(!isGray && !isDepth){
    	LOGE("Both depth and gray image is null");
    	return;
    }
    
    // save data as image matrix
    int k = 0;
    if(flip)
    {
    	k = cam_height * cam_width -1 ; 
    	k -= MARGIN*cam_width;
    }
    else
    {
    	k += MARGIN*cam_width;
    }
    
    for (int y = MARGIN; y < cam_height-MARGIN; y++)
    {
    	float *zRowPtr;
    	uint16_t *gRowPtr;
        if(isDepth) zRowPtr = depth.ptr<float> (y);
        if(isGray ) gRowPtr = gray.ptr<uint16_t> (y);
        k = flip ? k-MARGIN : k+MARGIN;
        for (int x = MARGIN; x < cam_width-MARGIN; x++)
        {
            auto curPoint = data->points.at (k);
            if (curPoint.depthConfidence >= min_confidence)
            {
                if(isDepth) zRowPtr[x] = curPoint.z;
                if(isGray ) gRowPtr[x] = curPoint.grayValue;
            }
            else
            {
            	if(isDepth) zRowPtr[x] = 0;
                if(isGray ) gRowPtr[x] = 0;
            }
            
            k = flip ? k-1 : k+1;
            
        }
        k = flip ? k-MARGIN : k+MARGIN;
    }
}

bool CamListener::visualizeImage(const Mat & src, Mat & dest, float resize_factor, bool color){
	if(!src.empty()){
        normalize(src, dest, 0, 255, NORM_MINMAX, CV_8UC1);
        if(color) applyColorMap(dest, dest, COLORMAP_JET);
    	resize(dest, dest, Size(), resize_factor,resize_factor);
    	return true;
    }
    return false;
}

bool CamListener::saveFrame(int frame)
{
    stringstream ss;
    ss <<  recordFolder << "/";
    if(frame < 10) ss << "0" ;
    if(frame < 100) ss << "0" ;
    if(frame < 1000) ss << "0" ;
    ss << frame;
	cv::FileStorage fs(ss.str(), cv::FileStorage::WRITE);

	fs << "xyzMap" << xyzMap;
	fs << "grayImage" << grayImage;
	fs << "confMap" << confMap;
	fs.release();
	return true;
}

void CamListener::startRecord(string destFolder)
{
    lock_guard<mutex> lock (flagMutex);
    recordFolder = destFolder;
	isRecording = true;
	frame = 0;
	LOGD("Recording has started");
}

void CamListener::stopRecord()
{
	lock_guard<mutex> lock (flagMutex);
	isRecording = false;
	LOGD("%d frames are saved into %s", frame, recordFolder.c_str());
}

void CamListener::updateMaps(const DepthData* data, bool flip)
{
	int k;
    if(flip) k = cam_height * cam_width -1 ; 
    else k = 0;
    for (int y = 0; y < cam_height; y++)
    {
    	Vec3f *xyzptr = xyzMap.ptr<Vec3f>(y);
    	uint8_t *confptr = confMap.ptr<uint8_t>(y);
    	uint16_t *grayptr= grayImage.ptr<uint16_t> (y);

        for (int x = 0; x < cam_width; x++)
        {
            auto curPoint = data->points.at (k);
            xyzptr[x][0] = curPoint.x;
            xyzptr[x][1] = curPoint.y;
            xyzptr[x][2] = curPoint.z;
            confptr[x] = curPoint.depthConfidence;
            grayptr[x] = curPoint.grayValue;
            
            k = flip ? k-1 : k+1;
        }

    }
}

void CamListener::setChannel(Mat & xyzMap, Mat & zChannel)
{
    const int cols = xyzMap.cols;
    const int step = xyzMap.channels();
    const int rows = xyzMap.rows;
    for (int y = 0; y < rows; y++) {
        Vec3f *xyzptr = xyzMap.ptr<Vec3f>(y);
        float *zptr = zChannel.ptr<float>(y);
        for (int x = 0; x < cols; x++)
        {
        	xyzptr[x][2] = zptr[x];
        }
    }
}





