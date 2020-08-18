//
// Created by dewe on 8/18/20.
//

#ifndef DRLCPP_FCV_H
#define DRLCPP_FCV_H

#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>

using std::vector;
using namespace torch;

struct FCVImpl : public nn::Module
{
    nn::Linear inLayer{ nullptr };
    nn::Linear fc1{ nullptr };
    nn::Linear outLayer{ nullptr };
    nn::Linear valueOut{nullptr};
    bool value=false;

    FCVImpl(int64_t  inDim, Device device)
    {

        inLayer = register_module("inLayer", torch::nn::Linear(inDim, 256));
        fc1 = register_module("fc1", torch::nn::Linear(256, 128));
        outLayer = register_module("outLayer", torch::nn::Linear(128, 1));
        this->to(device);

    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(inLayer->forward(x));
        x = torch::relu(fc1->forward(x));
        auto a = outLayer->forward(x);

        return a;
    }

};

TORCH_MODULE(FCV);

#endif //DRLCPP_FCV_H

