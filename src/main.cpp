
#include <opencv2/opencv.hpp>
#include <stdio.h>
// #include <filesystem>
// #include <string>
// #include <thread>
#include "loader.hpp"
#include "model.hpp"
namespace fs = std::filesystem;

using namespace cv;
using namespace std;
using namespace loader;

// class LoadedData
// {
// public:
//     vector<Mat> inputs;
//     vector<int> targets;
// };

// class ImageFunctions
// {
// public:
//     static vector<double> flatten(Mat image)
//     {
//         int rows = image.rows;
//         int cols = image.cols;
//         int channels = image.channels();
//         int total = rows * cols * channels;
//         int count = 0;
//         vector<double> result(total, 0);
//         for (int i = 0; i < rows; i++)
//         {
//             for (int j = 0; j < cols; j++)
//             {
//                 for (int k = 0; k < channels; k++)
//                 {
//                     result[count] = image.at<Vec3b>(i, j)[k];
//                     count++;
//                 }
//             }
//         }
//         return result;
//     }
// };

// void printVector(vector<double> vec)
// {
//     for (int i = 0; i < vec.size(); i++)
//     {
//         cout << vec[i] << " ";
//     }
//     cout << endl;
// }

// class DataLoader
// {

// public:
//     static void loadImages(const string folder,LoadedData *res)
//     {
        
//         // vector<Mat> res;
//         for (const auto &entry : fs::directory_iterator(folder))
//         {
//             const auto filenameStr = entry.path().filename().string();

//             if (entry.is_directory())
//             {
//                 // get last directory in the string filenamestr
//                 cout << filenameStr << "\n";
//                 // string dataClass = filenameStr.substr(10, 1);
//                 // cout << dataClass << " --\n";
//                 for (const auto &entry2 : fs::directory_iterator(entry.path().string()))
//                 {
//                     auto filenameStrInner = entry2.path().filename().string();

//                     if (entry2.is_regular_file())
//                     {
//                         res->inputs.push_back(imread(entry2.path().string(), 1));
//                         res->targets.push_back(atoi(filenameStr.c_str()));
//                         // std::cout << "   file: " << entry2.path().string() << '\n';
//                     }
//                 }
//             }
//         }
//         // Load all images in the folder
//         // return res;
//     }
//     static void loadImagesInThread(const string &folder, LoadedData &res)
//     {
//         for (const auto &entry : fs::directory_iterator(folder))
//         {
//             if (entry.is_regular_file())
//             {
//                 Mat img = imread(entry.path().string(), 1);
//                 // Lock here if necessary to push_back
//                 res.targets.push_back(atoi(fs::directory_entry(folder).path().filename().string().c_str()));
//                 cout << res.targets.back() << "--- \n";
//                 res.inputs.push_back(img);
//                 cout << "ello" <<endl;

//             }
//         }
//     }

//     static LoadedData loadImagesParallel(const string &folder)
//     {
//         LoadedData res;
//         vector<thread> threads;

//         for (const auto &entry : fs::directory_iterator(folder))
//         {
//             if (entry.is_directory())
//             {
//                 cout << entry.path().string() << "\n";
//                 threads.emplace_back(DataLoader::loadImagesInThread, entry.path().string(), std::ref(res));
//             }
//         }

//         for (auto &th : threads)
//         {
//             th.join();
//         }

//         return res;
//     }
//     static vector<vector<double>> FlattenData(vector<Mat> images)
//     {
//         vector<vector<double>> result;
//         for (auto image : images)
//         {
//             result.push_back(ImageFunctions::flatten(image));
//         }
//         // Load all images in the folder
//         // vector<vector<double>> result;
//         return result;
//     }
// };
void shuffleInUnison(vector<Eigen::MatrixXd> &result, vector<Eigen::MatrixXd> &oneHot) {
    // Ensure the vectors are of the same size
    if (result.size() != oneHot.size()) {
        throw runtime_error("Vectors must be of the same size to shuffle in unison.");
    }

    // Initialize a random engine
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine engine(seed);

    // Create an index vector
    vector<int> indices(result.size());
    iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., n-1

    // Shuffle the indices
    shuffle(indices.begin(), indices.end(), engine);

    // Apply the shuffled order to both vectors
    vector<Eigen::MatrixXd> shuffledResult(result.size());
    vector<Eigen::MatrixXd> shuffledOneHot(oneHot.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        shuffledResult[i] = result[indices[i]];
        shuffledOneHot[i] = oneHot[indices[i]];
    }

    // Swap the shuffled vectors back into the original ones
    result.swap(shuffledResult);
    oneHot.swap(shuffledOneHot);
}

int main(int argc, char **argv)
{
    Mat image;
    Mat temp;
    LoadedData data;
    DataLoader::loadImages("/home/ranguy/main/projects/NNfs/dataset/train",&data);

    vector<Eigen::MatrixXd> result;
    DataLoader::FlattenData(data.inputs, result);
    cout << result.size() << endl;
    // size of matrix
    cout << result[0].rows() << " " << result[0].cols() << endl;

    vector<Eigen::MatrixXd> oneHot = DataLoader::OneHotEncode(data.targets);
    shuffleInUnison(result, oneHot);

    // image = imread("/home/ranguy/Pictures/bed.jpg", 1);
    // cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    // cout << image[1][1];
    // func("/home/ranguy/main/projects/NNfs/");
    // printVector(ImageFunctions::flatten(temp));

    // if (!image.data)
    // {
    //     printf("No image data \n");
    //     return -1;
    // }
    model::Model model;
    model.setInput(result[0].cols());
    model.addLayer(12,&model::ActivationFunction::sigmoid);
    model.addLayer(10,&model::ActivationFunction::sigmoid);
    model.addLayer(10,&model::ActivationFunction::sigmoid);
    model.setOutput(10,&model::ActivationFunction::softmax);
    model.compile();
    // Eigen::MatrixXd inw(1,4);
    // inw << 1,2,3,4;
   
    // Eigen::MatrixXd exout(1,5);
    // exout << 1,1,0,1,0;
    


   
    // cout << model.layers.back().outputs << endl;

    for(int i = 0; i < 1000; i++)
    {
        for(int j = 0; j < result.size(); j+=1000){

            model.forwardSingle(result[j]);
            model.backwardSingle(model.layers.back().outputs,oneHot[j]);
            cout << oneHot[j] << endl;
            cout << model.layers.back().outputs << endl;
            int row, col;
            double maxVal = model.layers.back().outputs.maxCoeff(&row, &col);

            cout << "Max value: " << maxVal << " at (row, col): (" << row << ", " << col << ")" << endl;
            cout << "---------" << endl;
            cout << model.layers.back().outputs.maxCoeff() << endl;
  
            cout <<"---------" <<endl;
        }
    }



    // namedWindow("Display Image", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL);
    // imshow("Display Image", data.inputs[2]);
    // cout << " hello " << endl;
    // cout << data.targets[2] << "---------"<<endl;
    // waitKey(0);

    return 0;
}
