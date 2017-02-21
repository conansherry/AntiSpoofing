#ifndef ANTI_SPOOFING_H
#define ANTI_SPOOFING_H

#include <opencv2/opencv.hpp>

namespace readsense
{

class AntiSpoofing
{
public:
    AntiSpoofing();
    virtual ~AntiSpoofing();

    // Algorithm 1 Spoofing Detection in a video
    bool Predict(const std::vector<cv::Mat> &video, const std::vector<cv::Rect> &faces, int w, int N, int k);

    void Train(const std::vector<cv::Mat> &video, const std::vector<cv::Rect> &faces, int w, int N, int k, cv::Mat &traindata_frame_lbp, cv::Mat &traindata_face_lbp, cv::Mat &traindata_frame_hoof, cv::Mat &traindata_face_hoof);

    void RotateAndCrop_bySimilaryTransform(const cv::Mat& src, const std::vector<cv::Point2f>& coord, cv::Mat &dst, cv::Size dsize);

    std::vector<float> ExtractLBPFeature(const cv::Mat &img, const cv::Rect &face);
    std::vector<float> ExtractHOOFFeature(const cv::Mat &img_a, const cv::Rect &face_a, const cv::Mat &img_b, const cv::Rect &face_b);

    cv::Ptr<cv::ml::SVM> svm_lbp_face;
    cv::Ptr<cv::ml::SVM> svm_lbp_frame;
    cv::Ptr<cv::ml::SVM> svm_hoof_face;
    cv::Ptr<cv::ml::SVM> svm_hoof_frame;

    int w_;
    int N_;
    int k_;

    double w1_lbp;
    double w2_lbp;
    double w1_hoof;
    double w2_hoof;
    double wa;
    double wb;

    double T;
};

}

#endif