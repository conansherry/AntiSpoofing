#ifndef OPTICAL_FLOW_UTIL_H
#define OPTICAL_FLOW_UTIL_H

#include <opencv2/opencv.hpp>

namespace readsense
{

void CalcOpticalFlow(cv::Mat &prev, cv::Mat &next, cv::Mat &flow, cv::Mat &mag, cv::Mat &ang);

}

#endif