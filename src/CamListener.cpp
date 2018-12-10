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

   calculateProjectionCornerVectors();
   projCornerXyz = vector<Point3f>(4,Point3f(0,0,0));
   image = imread("checker.jpeg",1);
   resize(image,image, Size(1280,720));

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
   //blur image (7 for low exposure)
   medianBlur(channels[2], channels[2], 3);
   setChannel(xyzMap, channels[2]);

   if(visualizeImage(channels[2], depthImage8, 1.0, true))
   {
      imshow("Depth", depthImage8);
   }

   planeDetector->update(xyzMap);
   planes = planeDetector -> getPlanes();
   mainPlane = nullptr;
   for(auto plane: planes)
   {
      if(mainPlane == nullptr || plane->getSurfArea() > mainPlane->getSurfArea())
      {
         mainPlane = plane;
      }
   }
   if(mainPlane != nullptr)
   {
      Vec3f normal = mainPlane->getNormalVector();
      cout << "Main plane -------------------------------"
      << "\nEquation: "  << mainPlane->equation
      << "\nNormal  : "  << normal
      << "\nhorizontal: "<< atan(normal[0]/normal[2])*180/PI
      << "\nvertical  : "<< atan(normal[1]/normal[2])*180/PI << endl;

      correctKeyStone(mainPlane);
   }
   else
   {
      Mat output = Mat(Size(1280, 720),CV_8UC1,Scalar(0));
      putText(output, "No plane found", Point(580,300), FONT_HERSHEY_COMPLEX, 1, Scalar(255), 1, CV_AA);
      imshow("Projector", output);
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

// p is plane equation z = p[0]x + p[1]y + p[2]
void CamListener::calculateProjCornerPos(Vec3f p)
{
   float t;
   for(int i=0; i < 4; i++){
      Vec3f v = projCornerVectors[i];
      t = (p[0]*translation[0] + p[1]*translation[1] + p[2] - translation[2])
      / (v[2] - p[0]*v[0] - p[1]*v[1] ) ;
      projCornerXyz[i] = Point3f(v[0]*t + translation[0], v[1]*t + translation[1], v[2]*t + translation[2]);
   }
   tl = findPixelCoord(projCornerXyz[0]);
   bl = findPixelCoord(projCornerXyz[1]);
   tr = findPixelCoord(projCornerXyz[2]);
   br = findPixelCoord(projCornerXyz[3]);

   cout <<"0 - TL: " << projCornerXyz[0] <<"\t" << tl << endl;
   cout <<"1 - BL: " << projCornerXyz[1] <<"\t" << bl << endl;
   cout <<"2 - TR: " << projCornerXyz[2] <<"\t" << tr << endl;
   cout <<"3 - BR: " << projCornerXyz[3] <<"\t" << br << endl;

}

void CamListener::calculateProjectionCornerVectors(){
   if(projAxis[0] == 0)
   {
      LOGE("Projection axis not found!");
      return;
   }
   Vec3f v1(projAxis[0], projAxis[1], projAxis[2]);
   translation = NearestPointOnLine(Vec3f(projAxis[3], projAxis[4], projAxis[5]), v1, Vec3f(0,0,0));

   float gamma = std::atan(v1[1]/v1[2]);
   Vec3f r1(0, std::cos(gamma), std::sin(gamma));

   // Top Left
   // 2 ye bolunce daha iyi sonuç veriyor ama bolmeden zaten yarım fov??
   float angle_h = hor_fov/-2;
   float angle_v = ver_fov/2;
   Vec3f v2 = rotate(v1, r1, angle_h);
   Vec3f r2 = rotate(Vec3f(1,0,0), r1, angle_h);
   Vec3f v3 = rotate(v2,r2,angle_v);
   projCornerVectors.push_back(v3);
   // Bottom Left
   angle_v = ver_fov/-2;
   v3 = rotate(v2,r2,angle_v);
   projCornerVectors.push_back(v3);
   // Top Right
   angle_h = hor_fov/2;
   angle_v = ver_fov/2;
   v2 = rotate(v1, r1, angle_h);
   r2 = rotate(Vec3f(1,0,0), r1, angle_h);
   v3 = rotate(v2,r2,angle_v);
   projCornerVectors.push_back(v3);
   // Bottom Right
   angle_v = ver_fov/-2;
   v3 = rotate(v2,r2,angle_v);
   projCornerVectors.push_back(v3);
}

Point2i CamListener::distort(Point2i point)
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

   // To relative coordinates
   float x = ((float)point.x - cx) / fx;
   float y = ((float)point.y - cy) / fy;

   float r2 = x*x + y*y;

   // Radial distorsion
   float xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
   float yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

   // Tangential distorsion
   xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
   yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

   // Back to absolute coordinates.
   xDistort = xDistort * fx + cx;
   yDistort = yDistort * fy + cy;

   return Point2i((int)xDistort, (int)yDistort);
}

Point2i CamListener::findPixelCoord(Point3f p)
{
   int x = p.x*cameraMatrix.at<float>(0,0)/p.z + cameraMatrix.at<float>(0,2);
   int y = p.y*cameraMatrix.at<float>(1,1)/p.z + cameraMatrix.at<float>(1,2);
   return Point2i(x,y);
}

void CamListener::showPoint(const Point3f p){
   Point2i distIm = distort(findPixelCoord(p));
   int x = distIm.x;
   int y = distIm.y;
   if(x >= 0 && x < cam_width && y >=0 && y <cam_height){
      grayImage8.at<uint8_t>(y,x) = (grayImage8.at<uint8_t>(y,x)+128)%256;
   }
}

// plane->equation: z = p[0]x + p[1]y + p[2]
bool CamListener::correctKeyStone(FramePlane::Ptr plane)
{
   calculateProjCornerPos(plane->equation);

   for(int i=0; i<4; i++){
      showPoint(projCornerXyz[i]);
   }

   Vec3f normal = mainPlane->getNormalVector();
   float angle_h = atan(normal[0]/normal[2])*180/PI;
   float angle_v = atan(normal[1]/normal[2])*180/PI;

   if(abs(angle_h) < 3.0 && abs(angle_v) < 3.0)
   {
      cout << "No severe keystone" << endl;
      imshow("Projector", image);
      return true;
   }

   Point2f inputQuad[4];
   Point2f outputQuad[4];

   inputQuad[0] = Point2f(0,0);// tl;
   inputQuad[1] = Point2f(0,720); // bl;
   inputQuad[2] = Point2f(1280,0);// tr;
   inputQuad[3] = Point2f(1280,720); // br;

   // correct horizontal distortion
   //if(abs(angle_h) > 5.0 && abs(angle_v) < 5.0 )
   if(abs(angle_h) > abs(angle_v))
   {
      // right side is farther
      if(angle_h < 0)
      {
         Point3f corrTR = Point3f(projCornerXyz[2].x, projCornerXyz[0].y, projCornerXyz[2].z);
         Point3f corrBR = Point3f(projCornerXyz[3].x, projCornerXyz[1].y, projCornerXyz[3].z);

         showPoint(corrTR);
         showPoint(corrBR);

         float hr = br.y - tr.y ;
         float yt = findPixelCoord(corrTR).y;
         float yb = findPixelCoord(corrBR).y;

         outputQuad[0] = Point2f(0,0);
         outputQuad[1] = Point2f(0,720);
         outputQuad[2] = Point2f(1280, (float)720/hr*( yt - tr.y ));
         outputQuad[3] = Point2f(1280, (float)720/hr*( yb - tr.y ));
      }
      // left side is farther
      else
      {
         Point3f corrTL = Point3f(projCornerXyz[0].x, projCornerXyz[2].y, projCornerXyz[0].z);
         Point3f corrBL = Point3f(projCornerXyz[1].x, projCornerXyz[3].y, projCornerXyz[1].z);

         showPoint(corrTL);
         showPoint(corrBL);

         float hl = bl.y - tl.y ;
         float yt = findPixelCoord(corrTL).y;
         float yb = findPixelCoord(corrBL).y;

         outputQuad[0] = Point2f(0, (float)720/hl*( yt - tl.y ));
         outputQuad[1] = Point2f(0, (float)720/hl*( yb - tl.y ));
         outputQuad[2] = Point2f(1280,0);
         outputQuad[3] = Point2f(1280,720);
      }
   }
   // correct vertical distortion
   //else if(abs(angle_h) < 5.0 && abs(angle_v) > 5.0 )
   else
   {
      // top side is farther
      if(angle_v > 0)
      {
         Point3f corrTL = Point3f(projCornerXyz[1].x, projCornerXyz[0].y, projCornerXyz[0].z);
         Point3f corrTR = Point3f(projCornerXyz[3].x, projCornerXyz[2].y, projCornerXyz[2].z);

         showPoint(corrTL);
         showPoint(corrTR);

         float top_width = tr.x - tl.x;
         float xl = findPixelCoord(corrTL).x;
         float xr = findPixelCoord(corrTR).x;

         outputQuad[0] = Point2f((float)1280/top_width*( xl - tl.x ), 0);
         outputQuad[1] = Point2f(0,720);
         outputQuad[2] = Point2f((float)1280/top_width*( xr - tl.x ), 0);
         outputQuad[3] = Point2f(1280,720);
      }
      else
      {
         Point3f corrBL = Point3f(projCornerXyz[0].x, projCornerXyz[1].y, projCornerXyz[1].z);
         Point3f corrBR = Point3f(projCornerXyz[2].x, projCornerXyz[3].y, projCornerXyz[3].z);

         showPoint(corrBL);
         showPoint(corrBR);

         float bot_width = br.x - bl.x;
         float xl = findPixelCoord(corrBL).x;
         float xr = findPixelCoord(corrBR).x;

         outputQuad[0] = Point2f(0,0);
         outputQuad[1] = Point2f((float)1280/bot_width*( xl - bl.x ), 720);
         outputQuad[2] = Point2f(1280,0);
         outputQuad[3] = Point2f((float)1280/bot_width*( xr - bl.x ), 720);
      }

   }
   // Correct both horizontal and vertical
   /*else{
      outputQuad[0] = Point2f(0,0);// tl;
      outputQuad[1] = Point2f(0,720); // bl;
      outputQuad[2] = Point2f(1280,0);// tr;
      outputQuad[3] = Point2f(1280,720); // br;

      Point3f corr = Point3f(0,0,0);

      if(angle_h > 0 && angle_v > 0 ){ // correct TL
         corr.x = projCornerXyz[1].x;
         corr.y = projCornerXyz[2].y;
         corr.z = plane->getZ(corr.x,corr.y);

         outputQuad[0] =  findPixelCoord(corr);
         outputQuad[0].x = outputQuad[0].x*1280/cam_width;
         outputQuad[0].y = outputQuad[0].y*720/cam_height;
      }
      else if(angle_h > 0 && angle_v < 0 ){ // correct BL
         corr.x = projCornerXyz[0].x;
         corr.y = projCornerXyz[3].y;
         corr.z = plane->getZ(corr.x,corr.y);

         outputQuad[1] =  findPixelCoord(corr);
         outputQuad[1].x = outputQuad[1].x*1280/cam_width;
         outputQuad[1].y = outputQuad[1].y*720/cam_height;
      }
      else if(angle_h < 0 && angle_v > 0 ){ // correct TR
         corr.x = projCornerXyz[3].x;
         corr.y = projCornerXyz[0].y;
         corr.z = plane->getZ(corr.x,corr.y);

         outputQuad[2] =  findPixelCoord(corr);

         float top_width = tr.x - tl.x;
         outputQuad[2].x = (float)1280/top_width*( outputQuad[2].x - tl.x );

         float right_height = br.y - tr.y ;
         outputQuad[2].y = (float)720/right_height*( outputQuad[2].y - tr.y );
      }
      else{ // correct BR
         corr.x = projCornerXyz[2].x;
         corr.y = projCornerXyz[1].y;
         corr.z = plane->getZ(corr.x,corr.y);

         outputQuad[3] =  findPixelCoord(corr);
         float bottom_width = br.x - bl.x;
         outputQuad[3].x = (float)1280/bottom_width*( outputQuad[3].x - bl.x );

         float right_height = br.y - tr.y ;
         outputQuad[3].y = (float)720/right_height*( outputQuad[3].y - tr.y );
      }
      showPoint(corr);
   }*/

   Mat output;
   Mat pers = getPerspectiveTransform( inputQuad, outputQuad );
   warpPerspective(image,output,pers,image.size());
   imshow("Projector", output);

}
