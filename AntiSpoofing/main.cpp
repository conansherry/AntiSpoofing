#include <iostream>
#include <opencv2/opencv.hpp>
#include "AntiSpoofing.h"
#include <sstream>
#include <fstream>

int w_ = 16;
int N_ = 200;
int k_ = 2;

void train()
{

    {
        float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
        float trainingData[4][2] = { { 501, 10 },{ 255, 10 },{ 501, 255 },{ 10, 501 } };
        cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
        cv::Mat labelsMat(4, 1, CV_32FC1, labels);
        labelsMat.convertTo(labelsMat, CV_32SC1);
        cv::Ptr<cv::ml::TrainData> traindata = cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->train(traindata);
        cv::Mat result;
        svm->predict(trainingDataMat, result, true);
        std::cout << result << std::endl;
        system("pause");
    }

    readsense::AntiSpoofing as;
    cv::Mat traindata_frame_lbp;
    cv::Mat traindata_face_lbp;
    cv::Mat traindata_frame_hoof;
    cv::Mat traindata_face_hoof;
    cv::Mat label(0, 1, CV_32SC1);
    {
        std::vector<cv::Mat> video;
        std::vector<cv::Rect> faces;
        std::ifstream fin("D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/pos/1.txt");
        for (int i = 0; i < N_; i++)
        {
            std::stringstream ss;
            ss << "D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/pos/1/" << i << ".jpg";
            cv::Mat frame = cv::imread(ss.str());
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
            cv::resize(frame, frame, cv::Size(), 0.3, 0.3);
            cv::Rect face;
            fin >> face.x >> face.y >> face.width >> face.height;
            face.x *= 0.3;
            face.y *= 0.3;
            face.width *= 0.3;
            face.height *= 0.3;
            video.push_back(frame);
            faces.push_back(face);
        }
        int row = traindata_frame_lbp.rows;
        as.Train(video, faces, w_, N_, k_, traindata_frame_lbp, traindata_face_lbp, traindata_frame_hoof, traindata_face_hoof);
        row = traindata_frame_lbp.rows - row;
        cv::Mat label_one(row, 1, CV_32SC1, cv::Scalar(1));
        cv::vconcat(label, label_one, label);
        std::cout << traindata_frame_lbp.size() << std::endl;
        std::cout << traindata_face_lbp.size() << std::endl;
        std::cout << traindata_frame_hoof.size() << std::endl;
        std::cout << traindata_face_hoof.size() << std::endl;
        system("pause");
    }
    {
        std::vector<cv::Mat> video;
        std::vector<cv::Rect> faces;
        std::ifstream fin("D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/pos/2.txt");
        for (int i = 0; i < N_; i++)
        {
            std::stringstream ss;
            ss << "D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/pos/2/" << i << ".jpg";
            cv::Mat frame = cv::imread(ss.str());
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
            cv::resize(frame, frame, cv::Size(), 0.3, 0.3);
            cv::Rect face;
            fin >> face.x >> face.y >> face.width >> face.height;
            face.x *= 0.3;
            face.y *= 0.3;
            face.width *= 0.3;
            face.height *= 0.3;
            video.push_back(frame);
            faces.push_back(face);
        }
        int row = traindata_frame_lbp.rows;
        as.Train(video, faces, w_, N_, k_, traindata_frame_lbp, traindata_face_lbp, traindata_frame_hoof, traindata_face_hoof);
        row = traindata_frame_lbp.rows - row;
        cv::Mat label_one(row, 1, CV_32SC1, cv::Scalar(1));
        cv::vconcat(label, label_one, label);
        std::cout << traindata_frame_lbp.size() << std::endl;
        std::cout << traindata_face_lbp.size() << std::endl;
        std::cout << traindata_frame_hoof.size() << std::endl;
        std::cout << traindata_face_hoof.size() << std::endl;
        system("pause");
    }

    {
        std::vector<cv::Mat> video;
        std::vector<cv::Rect> faces;
        std::ifstream fin("D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/neg/1.txt");
        for (int i = 0; i < N_; i++)
        {
            std::stringstream ss;
            ss << "D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/neg/1/" << i << ".jpg";
            cv::Mat frame = cv::imread(ss.str());
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
            cv::resize(frame, frame, cv::Size(), 0.3, 0.3);
            cv::Rect face;
            fin >> face.x >> face.y >> face.width >> face.height;
            face.x *= 0.3;
            face.y *= 0.3;
            face.width *= 0.3;
            face.height *= 0.3;
            video.push_back(frame);
            faces.push_back(face);
        }
        int row = traindata_frame_lbp.rows;
        as.Train(video, faces, w_, N_, k_, traindata_frame_lbp, traindata_face_lbp, traindata_frame_hoof, traindata_face_hoof);
        row = traindata_frame_lbp.rows - row;
        cv::Mat label_zero(row, 1, CV_32SC1, cv::Scalar(-1));
        cv::vconcat(label, label_zero, label);
        std::cout << traindata_frame_lbp.size() << std::endl;
        std::cout << traindata_face_lbp.size() << std::endl;
        std::cout << traindata_frame_hoof.size() << std::endl;
        std::cout << traindata_face_hoof.size() << std::endl;
        system("pause");
    }
    {
        std::vector<cv::Mat> video;
        std::vector<cv::Rect> faces;
        std::ifstream fin("D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/neg/2.txt");
        for (int i = 0; i < N_; i++)
        {
            std::stringstream ss;
            ss << "D:/Workspace/AntiSpoofing/AntiSpoofing/train_data/neg/2/" << i << ".jpg";
            cv::Mat frame = cv::imread(ss.str());
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
            cv::resize(frame, frame, cv::Size(), 0.3, 0.3);
            cv::Rect face;
            fin >> face.x >> face.y >> face.width >> face.height;
            face.x *= 0.3;
            face.y *= 0.3;
            face.width *= 0.3;
            face.height *= 0.3;
            video.push_back(frame);
            faces.push_back(face);
        }
        int row = traindata_frame_lbp.rows;
        as.Train(video, faces, w_, N_, k_, traindata_frame_lbp, traindata_face_lbp, traindata_frame_hoof, traindata_face_hoof);
        row = traindata_frame_lbp.rows - row;
        cv::Mat label_zero(row, 1, CV_32SC1, cv::Scalar(-1));
        cv::vconcat(label, label_zero, label);
        std::cout << traindata_frame_lbp.size() << std::endl;
        std::cout << traindata_face_lbp.size() << std::endl;
        std::cout << traindata_frame_hoof.size() << std::endl;
        std::cout << traindata_face_hoof.size() << std::endl;
        system("pause");
    }

    std::cout << label.size() << std::endl;

    {
        
        system("pause");
        cv::Ptr<cv::ml::TrainData> traindata = cv::ml::TrainData::create(traindata_frame_lbp, cv::ml::ROW_SAMPLE, label);
        system("pause");
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        system("pause");
        svm->trainAuto(traindata);
        svm->save("svm_frame_lbp");
        system("pause");
    }
}

void test()
{

}

int main(int argc, char* argv[])
{
    train();
    //cv::VideoCapture cap(0);
    //cv::Mat frame;
    //for (int i = 0; i < 200; i++)
    //{
    //    cap >> frame;
    //    cv::imshow("frame", frame);
    //    cv::waitKey(20);
    //    std::stringstream ss;
    //    ss << i << ".jpg";
    //    cv::imwrite("train_data/neg/1/" + ss.str(), frame);
    //}
    return 0;
}