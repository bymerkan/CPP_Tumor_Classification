#include <stdio.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "../include/customDataset.hpp"

int main() {
    
    torch::manual_seed(1);
    std::srand(time(NULL));
    Options options;

    if (torch::cuda::is_available()) {
        options.device = torch::kCUDA;
    };

    std::cout << "[INFO] --- Running on: "
              << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    // Read Data

    std::pair<DataType, DataType> data = readData(options.dataPath);

    auto trainSet =
        CustomDataset(data.first).map(torch::data::transforms::Stack<>());

    std::cout << "[INFO] --- Train set has been prepared." << std::endl;

    // Generate a data loader.
    auto trainLoader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(trainSet), options.trainBatchSize);

    std::cout << "[INFO] --- Train loader has been prepared." << std::endl;

    auto testSet =
        CustomDataset(data.second).map(torch::data::transforms::Stack<>());

    std::cout << "[INFO] --- Test set has been prepared." << std::endl;

    // Generate a data loader.
    auto testLoader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(testSet), options.testBatchSize);

    std::cout << "[INFO] --- Test loader has been prepared." << std::endl;

    std::string modelPath = downloadModel(options);


    torch::jit::script::Module model = torch::jit::load(modelPath.data());

    model.to(options.device);


    torch::nn::Linear linear(options.linearSize, options.numClasses);
    
    //torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));
    torch::optim::Adam optimizer(linear->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));


    //Training

    float bestAccuracy = 0.0;
    std::string bestModelPath, bestLinearPath;

    for(int i = 0; i < options.epochs; i++){

        std::cout << "[TRAINING] --- Epoch: " << (i+1) << std::endl;

        float accuracy = 0.0;
        float MSE = 0;

        size_t batchCount = 1;

        for(auto& batch: *trainLoader){

            auto data = batch.data;

            auto target = batch.target.squeeze();
            
            // Should be of length: batch_size
            data = data.to(torch::kF32).to(options.device);
            target = target.to(torch::kInt64).to(options.device);

            std::vector<torch::jit::IValue> input;

            input.push_back(data);

            optimizer.zero_grad();

            auto output = model.forward(input).toTensor();

            output = output.view({output.size(0), -1});

            output = linear(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);

            loss.backward();
            optimizer.step();
            
            auto batchAccuracy = output.argmax(1).eq(target).sum();

            auto batchLoss = loss.template item <float>() / (float) options.trainBatchSize;

            std::cout << "[TRAINING] --- Batch: " << batchCount << "   Batch Loss: " << batchLoss << "   Batch Accuracy: " << (batchAccuracy.template item<float>() / (float) options.trainBatchSize) << std::endl;

            accuracy += batchAccuracy.template item<float>();
            MSE += loss.template item<float>();

            batchCount++;

        };

        MSE = MSE / (float) options.trainBatchSize;

        accuracy = (float) accuracy / (float) trainSet.size().value();

        std::cout << "[TRAINING] --- Epoch: " << (i+1) << "   Loss: " << MSE << "   Accuracy: " << accuracy << std::endl;

        if(accuracy >= bestAccuracy){

            bestAccuracy = accuracy;

            std::string modelSavePath = "../models/epoch" + std::to_string(i+1) + "_" + options.modelName + ".pt";

            std::string linearSavePath = "../models/epoch" + std::to_string(i+1) + "_linear_" + options.modelName + ".pt";

            model.save(modelSavePath);
            
            torch::save(linear, linearSavePath);

            bestModelPath = modelSavePath;
            bestLinearPath = linearSavePath;

            std::cout << "[SAVE] --- Best model and linear layer saved." << std::endl;

        };

    };


    // Testing

    std::cout << "[TESTING] --- Test started." << std::endl;

    if(options.device == torch::kCUDA){
        torch::jit::script::Module model = torch::jit::load(bestModelPath.data() , torch::kCUDA);
        torch::jit::script::Module linear = torch::jit::load(bestLinearPath.data() , torch::kCUDA);
    }else{
        torch::jit::script::Module model = torch::jit::load(bestModelPath.data());
        torch::jit::script::Module linear = torch::jit::load(bestLinearPath.data());
    };

    float accuracy = 0.0;
    float MSE = 0;

    size_t batchCount = 1;

    for(auto& batch: *testLoader){

        auto data = batch.data;

        auto target = batch.target.squeeze();
        
        // Should be of length: batch_size
        data = data.to(torch::kF32).to(options.device);
        target = target.to(torch::kInt64).to(options.device);

        std::vector<torch::jit::IValue> input;

        input.push_back(data);

        auto output = model.forward(input).toTensor();

        output = output.view({output.size(0), -1});

        output = linear(output);

        auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
        
        auto batchAccuracy = output.argmax(1).eq(target).sum();

        auto batchLoss = loss.template item <float>() / (float) options.trainBatchSize;

        std::cout << "[TESTING] --- Batch: " << batchCount << "   Batch Loss: " << batchLoss << "   Batch Accuracy: " << (batchAccuracy.template item<float>() / (float) options.testBatchSize) << std::endl;

        accuracy += batchAccuracy.template item<float>();
        MSE += loss.template item<float>();

        batchCount++;

    };

    MSE = MSE / (float) options.testBatchSize;

    accuracy = (float) accuracy / (float) testSet.size().value();

    std::cout << "[TESTING] --- Loss: " << MSE << "   Accuracy: " << accuracy << std::endl;

    return 0;
}