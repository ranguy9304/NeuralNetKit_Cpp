#ifndef LOADER_H // Include guards to prevent multiple inclusion

#define LOADER_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <filesystem>
#include <string>
#include <thread>
#include <Eigen/Dense>
namespace fs = std::filesystem;

using namespace cv;
using namespace std;

namespace loader
{
    class LoadedData
    {
    public:
        vector<Mat> inputs;
        vector<int> targets;
        mutex mu;
    };

    class DataLoader
    {

    public:
        static void loadImages(const string folder, LoadedData *res);
        
        static void loadImagesInThread(const string &folder, LoadedData &res);
       

        static void loadImagesParallel(const string &folder, LoadedData& res);
      
        // static vector<vector<double>> FlattenData(vector<Mat> images);

        static void FlattenData(vector<Mat> images, vector<Eigen::MatrixXd> &result);

        static vector<Eigen::MatrixXd> OneHotEncode(const vector<int> &targets);
        
    };

}

#endif