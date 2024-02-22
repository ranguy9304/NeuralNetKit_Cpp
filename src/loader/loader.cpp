

#include "loader.hpp"
namespace loader
{

    void DataLoader::loadImages(const string folder, LoadedData *res)
    {

        // vector<Mat> res;
        for (const auto &entry : fs::directory_iterator(folder))
        {
            const auto filenameStr = entry.path().filename().string();

            if (entry.is_directory())
            {
                // get last directory in the string filenamestr
                cout << filenameStr << "\n";
                // string dataClass = filenameStr.substr(10, 1);
                // cout << dataClass << " --\n";
                for (const auto &entry2 : fs::directory_iterator(entry.path().string()))
                {
                    auto filenameStrInner = entry2.path().filename().string();

                    if (entry2.is_regular_file())
                    {
                        res->inputs.push_back(imread(entry2.path().string(), 1));
                        res->targets.push_back(atoi(filenameStr.c_str()));
                        // std::cout << "   file: " << entry2.path().string() << '\n';
                    }
                }
            }
        }
        // Load all images in the folder
        // return res;
    }

    // void loadImagesInThread(const string &path, LoadedData &res)
    // {
    //     for (const auto &entry2 : fs::directory_iterator(path))
    //     {
    //         if (entry2.is_regular_file())
    //         {
    //             Mat img = imread(entry2.path().string(), 1);

    //             lock_guard<mutex> guard(res.mu);
    //             res.inputs.push_back(img);
    //             res.targets.push_back(atoi(fs::path(path).filename().string().c_str()));
    //         }
    //     }
    // }

    // void DataLoader::loadImagesParallel(const string &folder, LoadedData &res)
    // {
    //     vector<thread> threads;

    //     for (const auto &entry : fs::directory_iterator(folder))
    //     {
    //         if (entry.is_directory())
    //         {
    //             threads.emplace_back(loadImagesInThread, entry.path().string(), res);
    //         }
    //     }

    //     for (auto &th : threads)
    //     {
    //         th.join();
    //     }
    // }
    // static void loadImagesInThread(const string &folder, LoadedData &res)
    // {
    //     for (const auto &entry : fs::directory_iterator(folder))
    //     {
    //         if (entry.is_regular_file())
    //         {
    //             Mat img = imread(entry.path().string(), 1);
    //             // Lock here if necessary to push_back
    //             res.targets.push_back(atoi(fs::directory_entry(folder).path().filename().string().c_str()));
    //             cout << res.targets.back() << "--- \n";
    //             res.inputs.push_back(img);
    //             cout << "ello" <<endl;

    //         }
    //     }
    // }

    // static LoadedData loadImagesParallel(const string &folder)
    // {
    //     LoadedData res;
    //     vector<thread> threads;

    //     for (const auto &entry : fs::directory_iterator(folder))
    //     {
    //         if (entry.is_directory())
    //         {
    //             cout << entry.path().string() << "\n";
    //             threads.emplace_back(DataLoader::loadImagesInThread, entry.path().string(), std::ref(res));
    //         }
    //     }

    //     for (auto &th : threads)
    //     {
    //         th.join();
    //     }

    //     return res;
    // }
    // vector<vector<double>> Dataloader::FlattenData(vector<Mat> images)
    // {
    //     vector<vector<double>> result;
    //     for (auto image : images)
    //     {
    //         result.push_back(ImageFunctions::flatten(image));
    //     }
    //     // Load all images in the folder
    //     // vector<vector<double>> result;
    //     return result;
    // }
    void DataLoader::FlattenData(vector<Mat> images, vector<Eigen::MatrixXd> &result) {
        // vector<Eigen::MatrixXd> result;
        for (const auto &image : images) {
            // Flatten the image to a single row (1, X) format
            Eigen::MatrixXd flattened(1, image.rows * image.cols * image.channels());

            int k = 0;
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    for (int c = 0; c < image.channels(); ++c) {
                        flattened(0, k++) = image.at<Vec3b>(i, j)[c];
                    }
                }
            }

            result.push_back(flattened);
        }
        // return result;
    }
    vector<Eigen::MatrixXd> DataLoader::OneHotEncode(const vector<int> &targets) {
        // Find the number of classes
        // set<int> uniqueClasses(targets.begin(), targets.end());
        int numClasses = 10;

        // Create a vector of Eigen::MatrixXd for one-hot encoded vectors
        vector<Eigen::MatrixXd> oneHotEncoded;

        for (int target : targets) {
            Eigen::MatrixXd oneHot = Eigen::MatrixXd::Zero(1, numClasses);
            oneHot(0, target) = 1; // Set the appropriate position to 1
            oneHotEncoded.push_back(oneHot);
        }

        return oneHotEncoded;
    }

}