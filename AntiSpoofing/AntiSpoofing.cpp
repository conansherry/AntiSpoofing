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
    std::vector<std::vector<float> > frame_hoofs;
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
        std::vector<float> frame_hoof;
        std::vector<float> face_hoof;

        //Obtain videolet features
        for (int v = start; v < end; v++)
        {
            frame_lbp.insert(frame_lbp.end(), frame_lbps[v].begin(), frame_lbps[v].end());
            face_lbp.insert(face_lbp.end(), face_lbps[v].begin(), face_lbps[v].end());
        }
        for (int v = start; v + k_ < end; v = v + k_ + 1)
        {
            frame_hoof.insert(frame_hoof.end(), frame_hoofs[(v - 1) / k].begin(), frame_hoofs[(v - 1) / k].end());
            face_hoof.insert(face_hoof.end(), face_hoofs[(v - 1) / k].begin(), face_hoofs[(v - 1) / k].end());
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
        return false;
    else
        return true;
}

void AntiSpoofing::Train(const std::vector<cv::Mat>& video, const std::vector<cv::Rect>& faces, int w, int N, int k, cv::Mat &traindata_frame_lbp, cv::Mat &traindata_face_lbp, cv::Mat &traindata_frame_hoof, cv::Mat &traindata_face_hoof)
{
    //traindata_frame_lbp.release();
    //traindata_face_lbp.release();
    //traindata_frame_hoof.release();
    //traindata_face_hoof.release();

    CV_Assert(w / 2 % k == 0);
    w_ = w;
    N_ = N;
    k_ = k;

    std::vector<std::vector<float> > frame_lbps;
    std::vector<std::vector<float> > face_lbps;
    std::vector<std::vector<float> > frame_hoofs;
    std::vector<std::vector<float> > face_hoofs;
    CV_Assert(video.size() == faces.size());
    //Compute LBP features
    for (int i = 0; i < video.size(); i++)
    {
        frame_lbps.push_back(ExtractLBPFeature(video[i], cv::Rect()));
        face_lbps.push_back(ExtractLBPFeature(video[i], faces[i]));
    }
    //Compute HOOF features
    for (int i = 0; i + k_ < video.size(); i++)
    {
        frame_hoofs.push_back(ExtractHOOFFeature(video[i], cv::Rect(), video[i + k_], cv::Rect()));
        face_hoofs.push_back(ExtractHOOFFeature(video[i], faces[i], video[i + k_], faces[i + k_]));
    }

    int videolets_size = N_ - w_ + 1;

    for (int i = 0; i < videolets_size; i++)
    {
        int start = i;
        int end = start + w_;

        std::vector<float> frame_lbp;
        std::vector<float> face_lbp;
        for (int v = start; v < end; v++)
        {
            frame_lbp.insert(frame_lbp.end(), frame_lbps[v].begin(), frame_lbps[v].end());
            face_lbp.insert(face_lbp.end(), face_lbps[v].begin(), face_lbps[v].end());
        }

        std::vector<float> frame_hoof;
        std::vector<float> face_hoof;
        {
            for (int v = start; v + k_ < end; v = v + k_ + 1)
            {
                frame_hoof.insert(frame_hoof.end(), frame_hoofs[v].begin(), frame_hoofs[v].end());
                face_hoof.insert(face_hoof.end(), face_hoofs[v].begin(), face_hoofs[v].end());
            }
        }

        if (traindata_frame_lbp.empty())
        {
            traindata_frame_lbp.create(0, frame_lbp.size(), CV_32FC1);
        }
        cv::Mat frame_lbp_mat(1, frame_lbp.size(), CV_32FC1, frame_lbp.data());
        cv::vconcat(traindata_frame_lbp, frame_lbp_mat, traindata_frame_lbp);

        if (traindata_face_lbp.empty())
        {
            traindata_face_lbp.create(0, face_lbp.size(), CV_32FC1);
        }
        cv::Mat face_lbp_mat(1, face_lbp.size(), CV_32FC1, face_lbp.data());
        cv::vconcat(traindata_face_lbp, face_lbp_mat, traindata_face_lbp);

        if (traindata_frame_hoof.empty())
        {
            traindata_frame_hoof.create(0, frame_hoof.size(), CV_32FC1);
        }
        cv::Mat frame_hoof_mat(1, frame_hoof.size(), CV_32FC1, frame_hoof.data());
        cv::vconcat(traindata_frame_hoof, frame_hoof_mat, traindata_frame_hoof);

        if (traindata_face_hoof.empty())
        {
            traindata_face_hoof.create(0, face_hoof.size(), CV_32FC1);
        }
        cv::Mat face_hoof_mat(1, face_hoof.size(), CV_32FC1, face_hoof.data());
        cv::vconcat(traindata_face_hoof, face_hoof_mat, traindata_face_hoof);
    }
}

static float g_Average_5point[] = {
    0.344727, 0.349693,
    0.652331, 0.346436,
    0.496704, 0.511176,
    0.364636, 0.651934,
    0.637645, 0.649573
};

static void GetSimilarityTransform(const std::vector<float>& shapeSrc, const std::vector<float>& shapeDst, std::vector<float>& translate, std::vector<float>& rotate, float & scale, float & theta_para)
{
    int numLandmarks = shapeSrc.size() / 2;
    translate.resize(2);
    rotate.resize(4);
    translate[0] = 0;
    translate[1] = 0;
    rotate[0] = 0;
    rotate[1] = 0;
    rotate[2] = 0;
    rotate[3] = 0;
    scale = 0;
    theta_para = 0;

    float center_x_1 = 0;
    float center_y_1 = 0;
    float center_x_2 = 0;
    float center_y_2 = 0;
    for (int i = 0; i < numLandmarks; i++)
    {
        center_x_1 += shapeDst[2 * i];
        center_y_1 += shapeDst[2 * i + 1];
        center_x_2 += shapeSrc[2 * i];
        center_y_2 += shapeSrc[2 * i + 1];
    }
    center_x_1 /= numLandmarks;
    center_y_1 /= numLandmarks;
    center_x_2 /= numLandmarks;
    center_y_2 /= numLandmarks;

    translate[0] = center_x_1 - center_x_2;
    translate[1] = center_y_1 - center_y_2;

    std::vector<float> temp1(shapeDst);
    std::vector<float> temp2(shapeSrc);
    float srcCovXY = 0, srcCovXX = 0, srcCovYY = 0;
    float dstCovXY = 0, dstCovXX = 0, dstCovYY = 0;
    for (int i = 0; i < numLandmarks; i++)
    {
        float srcCovMean = 0, dstCovMean = 0;

        float &temp1_x = temp1[2 * i];
        float &temp1_y = temp1[2 * i + 1];
        temp1_x -= center_x_1;
        temp1_y -= center_y_1;
        dstCovMean = (temp1_x + temp1_y) / 2;
        float dstMinusMeanX = temp1_x - dstCovMean;
        float dstMinusMeanY = temp1_y - dstCovMean;

        dstCovXY += dstMinusMeanX * dstMinusMeanY;
        dstCovXX += dstMinusMeanX * dstMinusMeanX;
        dstCovYY += dstMinusMeanY * dstMinusMeanY;

        float &temp2_x = temp2[2 * i];
        float &temp2_y = temp2[2 * i + 1];
        temp2_x -= center_x_2;
        temp2_y -= center_y_2;
        srcCovMean = (temp2_x + temp2_y) / 2;
        float srcMinusMeanX = temp2_x - srcCovMean;
        float srcMinusMeanY = temp2_y - srcCovMean;

        srcCovXY += srcMinusMeanX * srcMinusMeanY;
        srcCovXX += srcMinusMeanX * srcMinusMeanX;
        srcCovYY += srcMinusMeanY * srcMinusMeanY;
    }
    float scaleSrc = std::sqrt(std::sqrt(srcCovXX * srcCovXX + 2 * srcCovXY * srcCovXY + srcCovYY * srcCovYY));
    float scaleDst = std::sqrt(std::sqrt(dstCovXX * dstCovXX + 2 * dstCovXY * dstCovXY + dstCovYY * dstCovYY));
    scale = scaleDst / scaleSrc;
    for (int i = 0; i < numLandmarks; i++)
    {
        temp1[2 * i] /= scaleDst;
        temp1[2 * i + 1] /= scaleDst;
        temp2[2 * i] /= scaleSrc;
        temp2[2 * i + 1] /= scaleSrc;
    }

    float num = 0;
    float den = 0;
    for (int i = 0; i < numLandmarks; i++)
    {
        float &temp1_x = temp1[2 * i];
        float &temp1_y = temp1[2 * i + 1];
        float &temp2_x = temp2[2 * i];
        float &temp2_y = temp2[2 * i + 1];
        num += temp1_y * temp2_x - temp1_x * temp2_y;
        den += temp1_x * temp2_x + temp1_y * temp2_y;
    }

    float norm = std::sqrt(num * num + den * den);
    float sin_theta = num / norm;
    theta_para = asin(sin_theta);
    float cos_theta = den / norm;
    rotate[0] = cos_theta;
    rotate[1] = -sin_theta;
    rotate[2] = sin_theta;
    rotate[3] = cos_theta;
}

void AntiSpoofing::RotateAndCrop_bySimilaryTransform(const cv::Mat & src, const std::vector<cv::Point2f>& coord, cv::Mat & dst, cv::Size dsize)
{
    if (dsize.area() == 0)
    {
        return;
    }

    cv::Rect rect = cv::boundingRect(coord);
    int addleft = 0, addtop = 0, addright = 0, addbottom = 0;
    float addRatio = 3;//important
    int realSize = 0;
    if (rect.width < rect.height)
    {
        realSize = rect.height * addRatio;
    }
    else
    {
        realSize = rect.width * addRatio;
    }
    addleft = (realSize - rect.width) / 2;
    addright = realSize - addleft - rect.width;
    addtop = (realSize - rect.height) / 2;
    addbottom = realSize - addtop - rect.height;
    int paddingLeft = 0, paddingRight = 0, paddingTop = 0, paddingBottom = 0;
    rect.x -= addleft;
    rect.y -= addtop;
    rect.width += (addleft + addright);
    rect.height += (addtop + addbottom);
    if (rect.x < 0)
        paddingLeft = -rect.x;
    if ((rect.x > 0 ? rect.x : 0) + rect.width > src.cols)
        paddingRight = (rect.x > 0 ? rect.x : 0) + rect.width - src.cols;
    if (rect.y < 0)
        paddingTop = -rect.y;
    if ((rect.y > 0 ? rect.y : 0) + rect.height > src.rows)
        paddingBottom = (rect.y > 0 ? rect.y : 0) + rect.height - src.rows;
    if (rect.x < 0)
        rect.x = 0;
    if (rect.y < 0)
        rect.y = 0;
    cv::Mat ext_src;
    cv::copyMakeBorder(src, ext_src, paddingTop, paddingBottom, paddingLeft, paddingRight, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::resize(ext_src(rect), ext_src, dsize);

    std::vector<float> shapeSrc(10);
    std::vector<float> shapeDst(10);
    std::vector<float> translate;
    std::vector<float> rotate;
    float scale = 1.0;
    float theta_para = 0;
    for (int i = 0; i < 5; i++)
    {
        shapeSrc[i * 2] = (coord[i].x + paddingLeft - rect.x) / rect.width * ext_src.cols;
        shapeSrc[i * 2 + 1] = (coord[i].y + paddingTop - rect.y) / rect.height * ext_src.rows;
        shapeDst[i * 2] = g_Average_5point[2 * i] * ext_src.cols;
        shapeDst[i * 2 + 1] = g_Average_5point[2 * i + 1] * ext_src.rows;
    }

    GetSimilarityTransform(shapeSrc, shapeDst, translate, rotate, scale, theta_para);
    float angle_a = -180 * theta_para / (CV_PI);
    cv::Point2f center;
    for (int i = 0; i < 5; i++)
    {
        center.x += shapeSrc[i * 2];
        center.y += shapeSrc[i * 2 + 1];
    }
    center.x /= 5;
    center.y /= 5;
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_a, scale);

    rot_mat.at<double>(0, 2) += translate[0];
    rot_mat.at<double>(1, 2) += translate[1];

    cv::Mat tmp;
    cv::Size sd;
    sd.width = ext_src.cols;
    sd.height = ext_src.rows;
    warpAffine(ext_src, tmp, rot_mat, sd, cv::INTER_LINEAR, IPL_BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    dst = tmp;
}

std::vector<float> AntiSpoofing::ExtractLBPFeature(const cv::Mat & img, const cv::Rect & face)
{
    cv::Mat crop;
    if (face.area() > 0)
        crop = img(face);
    else
        crop = img;
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
    if (next.size() != prev.size())
    {
        cv::resize(next, next, prev.size());
    }
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