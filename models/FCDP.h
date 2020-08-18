#pragma once
#include <torch/torch.h>
#include "Model.h"
#include "../utility/categorical.h"
#include <iostream>
#include <utility>
#include <vector>

using std::vector;
using namespace torch;

struct FCDP : public nn::Module, Model
{
    nn::Linear inLayer{ nullptr };
    nn::Linear fc1{ nullptr };
    nn::Linear outLayer{ nullptr };


    FCDP(int64_t  inDim, int32_t  outDim, Device device)
    {

        inLayer = register_module("inLayer", torch::nn::Linear(inDim, 128));
        fc1 = register_module("fc1", torch::nn::Linear(128, 64));
        outLayer = register_module("outLayer", torch::nn::Linear(64, outDim));

        this->to(device);

    }

    torch::Tensor forward(torch::Tensor x) override
    {
        x = torch::relu(inLayer->forward(x));
        x = torch::relu(fc1->forward(x));
        auto a = outLayer->forward(x);
        return a;
    }

    std::tuple<float, bool, Tensor, Tensor> fullPass(Tensor state) override
    {
        auto logits = forward(std::move(state));
        cpprl::Categorical dist(nullptr, &logits);
        auto action = dist.sample();
        auto logpa =  dist.log_prob(action).unsqueeze(-1);
        auto entropy =  dist.entropy().unsqueeze(-1);
        auto isExploratory = action != torch::argmax(logits.detach());
        return {action.item<float>(), isExploratory.item<bool>(), logpa, entropy};

    }
    vector<torch::Tensor> Parameters(bool recurse) override
    {
        return parameters(recurse);
    }

    float selectAction(Tensor state) override
    {
        auto logits = forward(std::move(state));
        cpprl::Categorical dist(nullptr, &logits);
        auto action = dist.sample();
        return action.item<float >();
    }

    float selectGreedyAction(Tensor state) override
    {
        auto logits = forward(std::move(state));
        return torch::argmax(logits.detach()).item<float >();
    }

    void Save(std::string const& name) override
    {
        torch::serialize::OutputArchive output_archive;
        this->save(output_archive);
        output_archive.save_to(name);
    }

    void Load(std::string const& name) override
    {
        torch::serialize::InputArchive input_archive;
        this->load(input_archive);
        input_archive.load_from(name);
    }
};

