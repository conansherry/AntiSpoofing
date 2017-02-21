#include "histogram_util.h"
#include <vector>

using namespace std;

namespace readsense
{

const static unsigned char uniform_lbps_59[] = {
    0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58,
    58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58,
    20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58,
    58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58,
    58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58,
    58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48,
    58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57
};

const static unsigned char uniform_lbps_243[] = { 0 };

template <typename _Tp>
void histogram_(const cv::Mat& src, cv::Mat& hist, int numPatterns) {
    hist = cv::Mat::zeros(1, numPatterns, CV_32SC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (numPatterns == 59)
            {
                int bin = uniform_lbps_59[src.at<_Tp>(i, j)];
                hist.at<int>(0, bin) += 1;
            }
            else if (numPatterns == 243)
            {
                int bin = uniform_lbps_243[src.at<_Tp>(i, j)];
                hist.at<int>(0, bin) += 1;
            }
            else
            {
                CV_Error(CV_StsBadArg, "numPatterns must be 59 or 243");
            }
        }
    }
}

template <typename _Tp>
double chi_square_(const cv::Mat& histogram0, const cv::Mat& histogram1) {
    if (histogram0.type() != histogram1.type())
        CV_Error(CV_StsBadArg, "Histograms must be of equal type.");
    if (histogram0.rows != 1 || histogram0.rows != histogram1.rows || histogram0.cols != histogram1.cols)
        CV_Error(CV_StsBadArg, "Histograms must be of equal dimension.");
    double result = 0.0;
    for (int i = 0; i < histogram0.cols; i++) {
        double a = histogram0.at<_Tp>(0, i) - histogram1.at<_Tp>(0, i);
        double b = histogram0.at<_Tp>(0, i) + histogram1.at<_Tp>(0, i);
        if (abs(b) > numeric_limits<double>::epsilon()) {
            result += (a*a) / b;
        }
    }
    return result;
}


void spatial_histogram(const cv::Mat& src, cv::Mat& hist, int numPatterns, const cv::Size& window, int overlap) {
    int width = src.cols;
    int height = src.rows;
    vector<cv::Mat> histograms;
    for (int x = 0; x < width - window.width; x += (window.width - overlap)) {
        for (int y = 0; y < height - window.height; y += (window.height - overlap)) {
            cv::Mat cell = cv::Mat(src, cv::Rect(x, y, window.width, window.height));
            histograms.push_back(histogram(cell, numPatterns));
        }
    }
    hist.create(1, histograms.size()*numPatterns, CV_32SC1);
    // i know this is a bit lame now... feel free to make this a bit more efficient...
    for (int histIdx = 0; histIdx < histograms.size(); histIdx++) {
        for (int valIdx = 0; valIdx < numPatterns; valIdx++) {
            int y = histIdx*numPatterns + valIdx;
            hist.at<int>(0, y) = histograms[histIdx].at<int>(valIdx);
        }
    }
}

// wrappers
void histogram(const cv::Mat& src, cv::Mat& hist, int numPatterns) {
    switch (src.type()) {
    case CV_8SC1: histogram_<char>(src, hist, numPatterns); break;
    case CV_8UC1: histogram_<unsigned char>(src, hist, numPatterns); break;
    case CV_16SC1: histogram_<short>(src, hist, numPatterns); break;
    case CV_16UC1: histogram_<unsigned short>(src, hist, numPatterns); break;
    case CV_32SC1: histogram_<int>(src, hist, numPatterns); break;
    }
}

double chi_square(const cv::Mat& histogram0, const cv::Mat& histogram1) {
    switch (histogram0.type()) {
    case CV_8SC1: return chi_square_<char>(histogram0, histogram1); break;
    case CV_8UC1: return chi_square_<unsigned char>(histogram0, histogram1); break;
    case CV_16SC1: return chi_square_<short>(histogram0, histogram1); break;
    case CV_16UC1: return chi_square_<unsigned short>(histogram0, histogram1); break;
    case CV_32SC1: return chi_square_<int>(histogram0, histogram1); break;
    }
}

void spatial_histogram(const cv::Mat& src, cv::Mat& dst, int numPatterns, int gridx, int gridy, int overlap) {
    int width = static_cast<int>(floor(src.cols / gridx));
    int height = static_cast<int>(floor(src.rows / gridy));
    spatial_histogram(src, dst, numPatterns, cv::Size_<int>(width, height), overlap);
}

// Mat return type functions
cv::Mat histogram(const cv::Mat& src, int numPatterns) {
    cv::Mat hist;
    histogram(src, hist, numPatterns);
    return hist;
}


cv::Mat spatial_histogram(const cv::Mat& src, int numPatterns, const cv::Size& window, int overlap) {
    cv::Mat hist;
    spatial_histogram(src, hist, numPatterns, window, overlap);
    return hist;
}


cv::Mat spatial_histogram(const cv::Mat& src, int numPatterns, int gridx, int gridy, int overlap) {
    cv::Mat hist;
    spatial_histogram(src, hist, numPatterns, gridx, gridy);
    return hist;
}

}
