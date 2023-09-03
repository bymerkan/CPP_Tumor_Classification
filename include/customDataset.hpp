#pragma once

#include <filesystem>
#include <iostream>

#include <torch/torch.h>

using DataType = std::vector<std::pair<std::string, std::string>>;

struct Options{
    std::string modelName = "resnet50";
    int imageSize = 512;
    size_t trainBatchSize = 32;
    size_t testBatchSize = 64;
    size_t epochs = 1;
    size_t linearSize = 1000;
    size_t numClasses = 2; // Fetch automatically?
    std::string dataPath = "../data/";
    torch::DeviceType device = torch::kCPU;
};

std::string downloadModel(Options options);

DataType shuffle(DataType toShuffle);

std::pair<DataType, DataType> readData(const std::string& path);

class CustomDataset : public torch::data::Dataset<CustomDataset>{

    private:

        int datasetSize;

    public:

        DataType data;

        Options options;

        CustomDataset(const DataType& data) : data(data) {
            datasetSize = data.size();
        };

        torch::optional<size_t> size() const{
            return datasetSize;
        };

        torch::data::Example<> get(size_t index);
};

