#include "Detector.h"
#include <DetectionParams.h>

namespace ark {
    Detector::Detector(DetectionParams::Ptr params)
        : params(params ? params : DetectionParams::create()) {
        //callback = std::bind(&Detector::callbackHelper, this, std::placeholders::_1);
    }

    void Detector::update(const cv::Mat & image)
    {
        this->image = image;
        detect(this->image);
        //lastCamera = nullptr;
        onSameFrame = false;
    }

    void Detector::setParams(const DetectionParams::Ptr params)
    {
        this->params = params;
    }
}
