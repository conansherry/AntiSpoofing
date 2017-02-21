#include "AntiSpoofing.h"
#include "util/histogram_util.h"
#include "util/lbp_util.h"
#include "util/optical_flow_util.h"

namespace readsense
{

AntiSpoofing::AntiSpoofing()
{
    w1_lbp = 0.5;
    w2_lbp = 0.5;
    w1_hoof = 0.5;
    w2_hoof = 0.5;
    wa = 0.5;
    wb = 0.5;
    T = 0.5;
}

AntiSpoofing::~AntiSpoofing()
{
}

bool AntiSpoofing::Predict(const std::vector<cv::Mat>& video, const std::vector<cv::Rect>& faces, int w, int N, int k)
{
    CV_Assert(w / 2 % k == 0);
    w_ = w;
    N_ = N;
    k_ = k;

    std::vector<std::vector<float> > frame_lbps;
    std::vector<std::vector<float> > face_lbps;
    std::vector<std::vector<float> > frame_hoofs;;
    std::vector<std::vector<float> > face_hoofs;
    CV_Assert(video.size() == faces.size());
    //Compute LBP features
    for (int i = 0; i < video.size(); i++)
    {
        frame_lbps.push_back(ExtractLBPFeature(video[i], cv::Rect()));
        face_lbps.push_back(ExtractLBPFeature(video[i], faces[i]));
    }
    //Compute HOOF features
    for (int i = 0; i + k_ < video.size(); i = i + k_ + 1)
    {
        frame_hoofs.push_back(ExtractHOOFFeature(video[i], cv::Rect(), video[i + k_], cv::Rect()));
        face_hoofs.push_back(ExtractHOOFFeature(video[i], faces[i], video[i + k_], faces[i + k_]));
    }

    //number of videolets
    int videolets_size = 2 * N_ / w_ - 1;
    double score_frame_lbps = 0;
    double score_face_lbps = 0;
    double score_frame_hoofs = 0;
    double score_face_hoofs = 0;

    for (int i = 0; i < videolets_size; i++)
    {
        int start = i * w_ / 2;
        int end = start + w_;
        std::vector<float> frame_lbp;
        std::vector<float> face_lbp;
        std::vector<float> frame_hoof;;
        std::vector<float> face_hoof;

        //Obtain videolet features
        for (int v = start; v < end; v++)
        {
            frame_lbp.insert(frame_lbp.end(), frame_lbps[v].begin(), frame_lbps[v].end());
            face_lbp.insert(face_lbp.end(), face_lbps[v].begin(), face_lbps[v].end());
        }
        int hoof_index = 0;
        for (int v = start; v + k_ < end; v = v + k_ + 1)
        {
            frame_hoof.insert(frame_hoof.end(), frame_lbps[hoof_index].begin(), frame_lbps[hoof_index].end());
            face_hoof.insert(face_hoof.end(), face_lbps[hoof_index].begin(), face_lbps[hoof_index].end());
            hoof_index++;
        }

        //Score Computation
        cv::Mat frame_lbp_mat(1, frame_lbp.size(), CV_32FC1, frame_lbp.data());
        cv::Mat frame_lbp_response;
        svm_lbp_frame->predict(frame_lbp_mat, frame_lbp_response, cv::ml::StatModel::RAW_OUTPUT);
        score_frame_lbps += frame_lbp_response.at<float>(0, 0);

        cv::Mat face_lbp_mat(1, face_lbp.size(), CV_32FC1, face_lbp.data());
        cv::Mat face_lbp_response;
        svm_lbp_frame->predict(face_lbp_mat, face_lbp_response, cv::ml::StatModel::RAW_OUTPUT);
        score_face_lbps += face_lbp_response.at<float>(0, 0);

        cv::Mat frame_hoof_mat(1, frame_hoof.size(), CV_32FC1, frame_hoof.data());
        cv::Mat frame_hoof_response;
        svm_lbp_frame->predict(frame_hoof_mat, frame_hoof_response, cv::ml::StatModel::RAW_OUTPUT);
        score_frame_hoofs += frame_hoof_response.at<float>(0, 0);

        cv::Mat face_hoof_mat(1, face_hoof.size(), CV_32FC1, face_hoof.data());
        cv::Mat face_hoof_response;
        svm_lbp_frame->predict(face_hoof_mat, face_hoof_response, cv::ml::StatModel::RAW_OUTPUT);
        score_face_hoofs += face_hoof_response.at<float>(0, 0);
    }

    score_frame_lbps /= videolets_size;
    score_face_lbps /= videolets_size;
    score_frame_hoofs /= videolets_size;
    score_face_hoofs /= videolets_size;

    double Ql = w1_lbp * score_frame_lbps + w2_lbp * score_face_lbps;
    double Qh = w1_hoof * score_frame_hoofs + w2_hoof * score_face_hoofs;
    double R = wa * Ql + wb * Qh;

    if (R > T)
    {
        return false;
    }
    else
        return true;
}

std::vector<float> AntiSpoofing::ExtractLBPFeature(const cv::Mat & img, const cv::Rect & face)
{
    cv::Mat crop;
    if (face.area() > 0)
        crop = img(face);
    else
        crop = img;
    std::cout << crop.channels() << std::endl;
    cv::Mat dst8_1, dst8_2, dst16_2;
    ELBP_<uint8_t>(crop, dst8_1, 1, 8);
    ELBP_<uint8_t>(crop, dst8_2, 2, 8);
    //ELBP_<uint8_t>(crop, dst16_2, 2, 16);

    cv::Mat feature1 = histogram(dst8_1, 59);
    cv::Mat feature2 = histogram(dst8_2, 59);
    feature1.convertTo(feature1, CV_64FC1);
    feature1.convertTo(feature2, CV_64FC1);
    cv::normalize(feature1, feature1, 1.0, cv::NORM_L2);
    cv::normalize(feature2, feature2, 1.0, cv::NORM_L2);
    feature1.convertTo(feature1, CV_32FC1);
    feature1.convertTo(feature2, CV_32FC1);
    std::vector<float> hist;
    hist.insert(hist.end(), (float*)feature1.data, (float*)feature1.data + feature1.size().area());
    hist.insert(hist.end(), (float*)feature2.data, (float*)feature2.data + feature2.size().area());
    return hist;
}

std::vector<float> AntiSpoofing::ExtractHOOFFeature(const cv::Mat & img_a, const cv::Rect & face_a, const cv::Mat & img_b, const cv::Rect & face_b)
{
    cv::Mat prev;
    if (face_a.area() > 0)
        prev = img_a(face_a);
    else
        prev = img_a;
    cv::Mat next;
    if (face_b.area() > 0)
        next = img_b(face_b);
    else
        next = img_b;
    cv::Mat flow, mag, ang;
    CalcOpticalFlow(prev, next, flow, mag, ang);
    ang = ang * 180 / CV_PI;
    std::vector<float> hist(81);
    cv::Rect cell(0, 0, mag.cols / 3, mag.rows / 3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::vector<float> hist_9(9);
            double l2norm = 0;
            for (int x = i * cell.width; x < (i + 1) * cell.width; x++)
            {
                for (int y = j * cell.height; y < (j + 1) * cell.height; y++)
                {
                    int bin = ((int)ang.at<float>(y, x) % 360) / 40 % 9;
                    float mag_v = mag.at<float>(y, x);
                    l2norm += mag_v * mag_v;
                    hist_9[bin] += mag_v;
                }
            }
            for (int k = 0; k < hist_9.size(); k++)
            {
                hist_9[k] /= (l2norm + 0.000001);
            }
            hist.insert(hist.end(), hist_9.begin(), hist_9.end());
        }
    }

    return hist;
}

}