#include "optical_flow_util.h"

namespace readsense
{

void CalcOpticalFlow(cv::Mat &prev, cv::Mat &next, cv::Mat &flow, cv::Mat &mag, cv::Mat &ang)
{
    cv::calcOpticalFlowFarneback(prev, next, flow, 0.5, 1, 15, 3, 5, 1.2, 0);

    std::vector<cv::Mat> split_flow;
    cv::split(flow, split_flow);
    cv::cartToPolar(split_flow[0], split_flow[1], mag, ang);
}

}