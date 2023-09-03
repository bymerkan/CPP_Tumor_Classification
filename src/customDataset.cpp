#include "../include/customDataset.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using DataType = std::vector<std::pair<std::string, std::string>>;


DataType shuffle(DataType toShuffle) {
    std::random_device rd;

    std::mt19937 g(rd());

    std::shuffle(toShuffle.begin(), toShuffle.end(), g);

    return toShuffle;
};

std::pair<DataType, DataType> readData(const std::string& path) {
    DataType train, test;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_directory(entry.path())) {
            std::string dirpath(entry.path());

            // Read image files and their labels

            for (const auto& path :
                 std::filesystem::directory_iterator(dirpath)) {
                std::string filepath(path.path());

                std::size_t lastOf = filepath.find_last_of("/");

                std::string dir = filepath.substr(0, lastOf);

                std::string file =
                    filepath.substr(lastOf + 1, filepath.length());

                std::size_t lastOfClass = dir.find_last_of("/");

                std::string classOf = dir.substr(lastOfClass + 1, dir.length());

                bool split = (std::rand() % 100) < 80;

                if (split) {
                    // Train
                    train.push_back(std::make_pair(filepath, classOf));
                } else {
                    // Test
                    test.push_back(std::make_pair(filepath, classOf));
                }

                // data.emplace_back(filepath, classOf);
            };
        };
    };

    return std::make_pair(shuffle(train), shuffle(test));
};

torch::data::Example<> CustomDataset::get(size_t index){

    std::string filePath = data[index].first;

    auto image = cv::imread(filePath, 1);

    cv::resize(image, image, cv::Size(options.imageSize, options.imageSize), cv::INTER_CUBIC);


    /*
    std::vector<cv::Mat> channels(3);

    cv::split(image, channels);

    auto R = torch::from_blob(channels[2].ptr(),
                              {options.imageSize, options.imageSize},
                              torch::kUInt8);
    auto G = torch::from_blob(channels[1].ptr(),
                              {options.imageSize, options.imageSize},
                              torch::kUInt8);
    auto B = torch::from_blob(channels[0].ptr(),
                              {options.imageSize, options.imageSize},
                              torch::kUInt8);

    auto imageTensor = torch::cat({R, G, B})
                     .view({3, options.imageSize, options.imageSize})
                     .to(torch::kFloat);
    */

    torch::Tensor imageTensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
    imageTensor = imageTensor.permute({2, 0, 1});

    int classInt;
    classInt = (data[index].second == "yes") ? 1 : 0;

    torch::Tensor labelTensor = torch::from_blob(&classInt, {1}, torch::kInt);

    return {imageTensor.clone(), labelTensor.clone()};
};

std::string downloadModel(Options options){

    std::string modelPath = "../models/" + options.modelName + ".pt";

    std::string pyCommand = "python3 ../include/models.py -a " + options.modelName;

    std::system(pyCommand.data());

    std::cout << "[INFO] --- "<< options.modelName <<" model is ready to use." << std::endl;

    return modelPath;
};
