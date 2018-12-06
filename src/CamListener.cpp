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
   cameraMatrix = (Mat1f (3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
   0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
   0, 0, 1);
/*   LOGI("Camera params fx fy cx cy: %f,%f,%f,%f", lensParameters.focalLength.first, lensParameters.focalLength.second,
   lensParameters.principalPoint.first, lensParameters.principalPoint.second); */

   // Construct the distortion coefficients
   // k1 k2 p1 p2 k3
   distortionCoefficients = (Mat1f (1, 5) << lensParameters.distortionRadial[0],
   lensParameters.distortionRadial[1],
   lensParameters.distortionTangential.first,
   lensParameters.distortionTangential.second,
   lensParameters.distortionRadial[2]);
 /* LOGI("Dist coeffs k1 k2 p1 p2 k3 : %f,%f,%f,%f,%f", lensParameters.distortionRadial[0],
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
   visualizeImage(grayImage, grayImage8, 1.0);

   vector<Mat> channels(3);
   split(xyzMap, channels);
   // blur image (7 for low exposure)
   //medianBlur(channels[2], channels[2], 3);
   //setChannel(xyzMap, channels[2]);

   if(visualizeImage(channels[2], depthImage8, 1.0, true))
   {
      imshow("Depth", depthImage8);
   }
  
   planeDetector->update(xyzMap);
   planes = planeDetector -> getPlanes();
   for(auto plane: planes){
      Vec3f corner = findProjectionCorner(plane->equation, hor_fov*-1, ver_fov);
      Point2f im(corner[0]*cameraMatrix.at<float>(0,0)/corner[2] + cameraMatrix.at<float>(0,2),
      			    corner[1]*cameraMatrix.at<float>(1,1)/corner[2] + cameraMatrix.at<float>(1,2));
      Point2i distIm = distort(im);
      int x = distIm.x; 
      int y = distIm.y;
      if(x >= 0 && x < cam_width && y >=0 && y <cam_height) grayImage8.at<uint8_t>(y,x) = 0;
      LOGD("TopLeft(cm):\t (%.3f, %.3f, %.3f)\t x,y = %d,%d", corner[0]*100, corner[1]*100, corner[2]*100, x,y);
      
      corner = findProjectionCorner(plane->equation, hor_fov, ver_fov);
      im = Point2f(corner[0]*cameraMatrix.at<float>(0,0)/corner[2] + cameraMatrix.at<float>(0,2),
      			    corner[1]*cameraMatrix.at<float>(1,1)/corner[2] + cameraMatrix.at<float>(1,2));
      distIm = distort(im); // Distortion for debug. Keystone yaparken distort etme.
      x = distIm.x; 
      y = distIm.y;
      if(x >= 0 && x < cam_width && y >=0 && y <cam_height) grayImage8.at<uint8_t>(y,x) = 0;
      LOGD("TopRight:\t (%.3f, %.3f, %.3f)\t x,y = %d,%d", corner[0]*100, corner[1]*100, corner[2]*100,x,y);
      
      corner = findProjectionCorner(plane->equation, hor_fov*-1, ver_fov*-1);
      im = Point2f(corner[0]*cameraMatrix.at<float>(0,0)/corner[2] + cameraMatrix.at<float>(0,2),
      			    corner[1]*cameraMatrix.at<float>(1,1)/corner[2] + cameraMatrix.at<float>(1,2));
      distIm = distort(im);
      x = distIm.x; 
      y = distIm.y;
      if(x >= 0 && x < cam_width && y >=0 && y <cam_height) grayImage8.at<uint8_t>(y,x) = 0;
      LOGD("BottomLeft:\t (%.3f, %.3f, %.3f)\t x,y = %d,%d", corner[0]*100, corner[1]*100, corner[2]*100,x,y);
      
      corner = findProjectionCorner(plane->equation, hor_fov, ver_fov*-1);
      im = Point2f(corner[0]*cameraMatrix.at<float>(0,0)/corner[2] + cameraMatrix.at<float>(0,2),
      			    corner[1]*cameraMatrix.at<float>(1,1)/corner[2] + cameraMatrix.at<float>(1,2));
      distIm = distort(im);
      x = distIm.x; 
      y = distIm.y;
      if(x >= 0 && x < cam_width && y >=0 && y <cam_height) grayImage8.at<uint8_t>(y,x) = 0;
      LOGD("BottomRight:\t (%.3f, %.3f, %.3f)\t x,y = %d,%d", corner[0]*100, corner[1]*100, corner[2]*100,x,y);      
   }
   resize(grayImage8, grayImage8, Size(), 3,3);
   imshow("Gray", grayImage8);
   
   
   /*Mat normalMap = planeDetector -> getNormalMap();
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
      if(resize_factor != 1) resize(dest, dest, Size(), resize_factor,resize_factor);
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

void CamListener::saveCenterPoint()
{
   lock_guard<mutex> lock (flagMutex);
   // Find retro in gray
   vector<vector<Point> > retro_contours;
   Mat grayBin;
   threshold(grayImage, grayBin, 500, 255, CV_THRESH_BINARY);
   grayBin.convertTo(grayBin, CV_8UC1);
   findContours(grayBin, retro_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
   
   if(retro_contours.size() != 1){
   	  printf("More than 1 retro. Point not added. Current pts count = %d\n", centers.size());
   	  return ;
   }
   
   Moments mu = moments(retro_contours[0], false);
   int u = mu.m10 / mu.m00;
   int v = mu.m01 / mu.m00;
   if(u >= cam_width || u < 0 || v >= cam_height || v < 0 ){
   		printf("Center(%d,%d) outside of the range. Point not added. Current pts count = %d\n",u,v, centers.size());
   	    return ;
   }
   
   Vec3f point = xyzMap.at<Vec3f>(v,u);
   uint8_t confidence = confMap.at<uint8_t>(v,u);
   printf("Depth point: x=%.2f\ty=%.2f\tz=%.2f\t u=%d\tv=%d\t conf=%.2f\n"
            ,point[0], point[1], point[2], u, v, (float)confidence*100/255 );
   if(confidence > 200){
   		centers.push_back(point);
   		printf("Point added. Current pts count = %d\n", centers.size());
   }
   else{
   		printf("Low confidence. Point not added. Current pts count = %d\n", centers.size());
   }
}

void CamListener::calculateProjectionAxis()
{
   cv::fitLine(centers, projAxis, CV_DIST_L2, 0, 0.01, 0.01);
   printf("Vector : (%.3f,%.3f,%.3f)\n", projAxis[0], projAxis[1], projAxis[2]);
   printf("Point  : (%.3f,%.3f,%.3f) -> mean of points\n", projAxis[3], projAxis[4], projAxis[5]);
   cv::FileStorage fs("Projector", cv::FileStorage::WRITE);
   fs << "projAxis" << projAxis;
   fs.release();
}

Vec3f rotate(const Vec3f &in, const Vec3f &axis, float angle)
{
	float a = std::cos(angle);
	float s = std::sin(angle);
	Vec3f w( s*axis[0], s*axis[1], s*axis[2]);
	Vec3f cr = w.cross(in);
	return ( in + 2*a*cr + 2*(w.cross(cr)));
}

Vec3f NearestPointOnLine(Vec3f linePnt, Vec3f lineDir, Vec3f pnt)
{
    auto v = pnt - linePnt;
    auto d = v[0]*lineDir[0] + v[1]*lineDir[1] + v[2]*lineDir[2]; // dot product
    return linePnt + lineDir * d;
}

// ( p is plane equation z = p[0]x + p[1]y + p[2] ) 
Vec3f CamListener::findProjectionCorner(Vec3f p, float angle_h, float angle_v) 
{
	Vec3f v1(projAxis[0], projAxis[1], projAxis[2]);
	Vec3f tr = NearestPointOnLine(Vec3f(projAxis[3], projAxis[4], projAxis[5]), v1, Vec3f(0,0,0));
	float gamma = std::atan(v1[1]/v1[2]);
	Vec3f r1(0, std::cos(gamma), std::sin(gamma));
	Vec3f v2 = rotate(v1, r1, angle_h);
    Vec3f r2 = rotate(Vec3f(1,0,0), r1, angle_h);
    Vec3f v3 = rotate(v2,r2,angle_v);
    // TODO burdan yukarıyı bir kere hesaplayıp kaydet. ( Kose vektorleri değişmiyor çunku)
    
    float t = (p[0]*tr[0] + p[1]*tr[1] + p[2] - tr[2]) / (v3[2] - p[0]*v3[0] - p[1]*v3[1] ) ;
    return Vec3f(v3[0]*t + tr[0], v3[1]*t + tr[1], v3[2]*t + tr[2]);
}

Point2i CamListener::distort(Point2f point)
{
	float cx = cameraMatrix.at<float>(0,2);
	float cy = cameraMatrix.at<float>(1,2);
	float fx = cameraMatrix.at<float>(0,0);
	float fy = cameraMatrix.at<float>(1,1);
	float k1 = distortionCoefficients.at<float>(0,0);
	float k2 = distortionCoefficients.at<float>(0,1);
	float p1 = distortionCoefficients.at<float>(0,2);
	float p2 = distortionCoefficients.at<float>(0,3);
	float k3 = distortionCoefficients.at<float>(0,4);

    // To relative coordinates <- this is the step you are missing.
    double x = (point.x - cx) / fx;
    double y = (point.y - cy) / fy;

    double r2 = x*x + y*y;

    // Radial distorsion
    double xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    double yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    // Back to absolute coordinates.
    xDistort = xDistort * fx + cx;
    yDistort = yDistort * fy + cy;
	
    return Point2d((int)xDistort, (int)yDistort);
}




















