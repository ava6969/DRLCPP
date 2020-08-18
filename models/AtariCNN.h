//
// Created by dewe on 8/17/20.
//

#ifndef DRLCPP_ATARICNN_H
#define DRLCPP_ATARICNN_H

#include "Model.h"

struct AtariCNN : public nn::Module, Model
{
    nn::Conv2d conv1, conv2, conv3;
    nn::Linear outLayer{ nullptr };
    nn::Linear valueOut{nullptr};
    Device device = kCPU;
    bool value=false;
    AtariCNN(int32_t  outDim, Device device, bool _value=false):
    conv1(torch::nn::Conv2dOptions (4, 32, 8).stride(4)),
    conv2(torch::nn::Conv2dOptions (32, 64, 4).stride(4)),
    conv3(torch::nn::Conv2dOptions (64, 64, 3).stride(1))
    {
        value = _value;
        device = device;

        conv1 = register_module("conv1", conv1);
        conv2 = register_module("conv2", conv2);
        conv3 = register_module("conv3", conv3);
        int64_t out = calculateConvOutPutDims();
        outLayer = register_module("outLayer", torch::nn::Linear(out, outDim));
        if (value)
            valueOut = register_module("valueOut", torch::nn::Linear(out,1));
        this->to(device);

    }

    int calculateConvOutPutDims()
    {
        auto x =  torch::zeros({1, 4, 84, 84}, device);
        x = conv1(x);
        x = conv2(x);
        x = conv3(x);
        return torch::prod_intlist(x.sizes());
    }

    torch::Tensor forward(torch::Tensor x) override
    {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = x.view({x.size(0),-1});

        auto a = outLayer->forward(x);
        if (value)
        {
            auto v = valueOut(x);
            v = v.expand_as(a);
            return v + a - a.mean(1,true).expand_as(a);
        }
        return a;
    }

    std::vector<torch::Tensor> Parameters(bool recurse) override
    {
        return parameters(recurse);
    }

    void Save(std::string const& name) override
    {
        printf("Saving CheckPoint");
        torch::serialize::OutputArchive output_archive;
        this->save(output_archive);
        output_archive.save_to(name);
    }

    void Load(std::string const& name) override
    {
        printf("Loading CheckPoint");
        torch::serialize::InputArchive input_archive;
        this->load(input_archive);
        input_archive.load_from(name);
    }
};


#endif //DRLCPP_ATARICNN_H
