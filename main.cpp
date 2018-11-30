#include <royale.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "CamListener.h"
#include <cxxopts.hpp>

using namespace royale;
using namespace std;
using namespace cv;

// this represents the main camera device object
static std::unique_ptr<ICameraDevice> cameraDevice;
static CamListener listener;

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;
    Vec3f point = listener.xyzMap.at<Vec3f>(y, x);
	printf("Depth point in(cm): x=%.2f\ty=%.2f\tz=%.2f \n", point[0]*100, point[1]*100, point[2]*100 );
}

int main (int argc, char *argv[])
{
	// Parse options
	cxxopts::Options options("Retro Plane");
	options
      .allow_unrecognised_options()
      .add_options()
    ("f", "Record folder", cxxopts::value<string>(), "file_path")
    ("e", "Exposure times", cxxopts::value<int>(), "exp_time");
    cout << options.help() << endl;
    cout << "Press 'p' for pause/resume" << endl;
	cout << "Press 'ESC' for exit" << endl << endl;
	
	auto result = options.parse(argc, argv);
	
	// windows
	namedWindow ("Gray", WINDOW_AUTOSIZE);
    namedWindow ("Depth", WINDOW_AUTOSIZE);
    namedWindow ("FloodFill", WINDOW_AUTOSIZE);
    namedWindow ("[Plane Debug]", WINDOW_AUTOSIZE);
    namedWindow ("NormalMap", WINDOW_AUTOSIZE);
    
    setMouseCallback( "Depth", onMouse, 0 );
	
    {
        CameraManager manager;

        // check if any record folder is given
        if (result.count("f"))
        {
			string file_path = result["f"].as<std::string>();
			cout << "Trying to open : " << file_path << endl;
			vector<string> file_names;
	
			boost::filesystem::path image_dir(file_path);
			if (is_directory(image_dir)) 
			{
				boost::filesystem::directory_iterator end_iter;
				for (boost::filesystem::directory_iterator dir_itr(image_dir); dir_itr != end_iter; ++dir_itr) {
					const auto next_path = dir_itr->path().generic_string();
					file_names.push_back(next_path);
				}
			}
			sort(file_names.begin(), file_names.end());
			listener.initialize(224, 171);
	
			string path;
			bool stop = false;
			for(int i=0; i< file_names.size();) 
			{
				FileStorage fs2(file_names[i], FileStorage::READ);
		
				fs2["grayImage"] >> listener.grayImage;
				fs2["xyzMap"] >> listener.xyzMap;
				listener.processImages();

				int currentKey = waitKey (100) & 255;
				if (currentKey == 'p' || stop)
				{
					stop = true;
			 		while(1){
			 			currentKey = waitKey (0) & 255;
			 			if(currentKey == 83){
							i++;
							break;
						}
						else if(currentKey == 81 && i>0){
							i--;
							break;
						}
						else if(currentKey == 27){
							i = file_names.size();
							break;
						}
						else if(currentKey == 'p'){
							stop = false;
							break;
						}
			 		}	
				}
				if(!stop) i++;

		
			}
			return 0;
        }
        else
        {
            // if no argument was given try to open the first connected camera
            royale::Vector<royale::String> camlist (manager.getConnectedCameraList());
            cout << "Detected " << camlist.size() << " camera(s)." << endl;

            if (!camlist.empty())
            {
                cameraDevice = manager.createCamera (camlist[0]);
            }
            else
            {
                cerr << "No suitable camera device detected." << endl
                     << "Please make sure that a supported camera is plugged in, all drivers are "
                     << "installed, and you have proper USB permission" << endl;
                return 0;
            }

            camlist.clear();
        }
    }
    // the camera device is now available and CameraManager can be deallocated here

    if (cameraDevice == nullptr)
    {
		cerr << "cameraDevice is null" << endl;
		return 0;
    }

    // IMPORTANT: call the initialize method before working with the camera device
    auto status = cameraDevice->initialize();
    if (status != CameraStatus::SUCCESS)
    {
        cerr << "Cannot initialize the camera device, error string : " << getErrorString (status) << endl;
        return 1;
    }
    
    royale::Vector<royale::String> opModes;
    status = cameraDevice->getUseCases (opModes);
    if (status != CameraStatus::SUCCESS)
    {
        printf("Failed to get use cases, CODE %d\n", (int) status);
        return 1;
    }
    
    // set an operation mode
    status = cameraDevice->setUseCase (opModes[0]);
    if (status != CameraStatus::SUCCESS)
    {
        printf("Failed to set use case, CODE %d\n", (int) status);
        return 1;
    }
    
	uint16_t cam_width, cam_height;
    status = cameraDevice->getMaxSensorWidth (cam_width);
    if (CameraStatus::SUCCESS != status)
    {
        cerr << "failed to get max sensor width: " << getErrorString (status) << endl;
        return 1;
    }

    status = cameraDevice->getMaxSensorHeight (cam_height);
    if (CameraStatus::SUCCESS != status)
    {
        cerr << "failed to get max sensor height: " << getErrorString (status) << endl;
        return 1;
    }
    listener.initialize(cam_width, cam_height);

	if(result.count("e"))
	{
	   //set exposure mode to manual
	   status = cameraDevice->setExposureMode (ExposureMode::MANUAL);
	   if (status != CameraStatus::SUCCESS)
	   {
		  printf ("Failed to set exposure mode, CODE %d\n", (int) status);
	   }

		//set exposure time
		status = cameraDevice->setExposureTime(result["e"].as<int>());
		if (status != CameraStatus::SUCCESS)
		{
		    printf ("Failed to set exposure time, CODE %d\n", (int) status);
		}
	}
	else
	{
		//set exposure mode to auto
	   status = cameraDevice->setExposureMode (ExposureMode::AUTOMATIC);
	   if (status != CameraStatus::SUCCESS)
	   {
		  printf ("Failed to set exposure mode, CODE %d\n", (int) status);
	   }
	}
 

    // retrieve the lens parameters from Royale
    LensParameters lensParameters;
    status = cameraDevice->getLensParameters (lensParameters);
    if (status != CameraStatus::SUCCESS)
    {
        cerr << "Can't read out the lens parameters" << endl;
        return 1;
    }
    listener.setLensParameters (lensParameters);

    // register a data listener
    if (cameraDevice->registerDataListener (&listener) != CameraStatus::SUCCESS)
    {
        cerr << "Error registering data listener" << endl;
        return 1;
    }

    // start capture mode
    if (cameraDevice->startCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error starting the capturing" << endl;
        return 1;
    }
    else cout << "Capture started" << endl;

    int currentKey = 0;
    bool isCapturing;

    while (currentKey != 27)
    {
        // wait until a key is pressed
        currentKey = waitKey (0) & 255;
        if (currentKey == 'p')
        {
        	cameraDevice->isCapturing(isCapturing);
        	 if(isCapturing)
        	 {
        	 	if (cameraDevice->stopCapture() != CameraStatus::SUCCESS)
        			cerr << "Error stopping the capturing" << endl;
        	 	else cout << "Capture stopped" << endl;
        	 }
        	 else
        	 {
        	 	if (cameraDevice->startCapture() != CameraStatus::SUCCESS)
        			cerr << "Error stopping the capturing" << endl;
        		else cout << "Capture started" << endl;
        	 }
        	 
    	}
    }
	
    // stop capture mode
    if (isCapturing && cameraDevice->stopCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error stopping the capturing" << endl;
        return 1;
    }
    
    destroyAllWindows();

    return 0;
}
