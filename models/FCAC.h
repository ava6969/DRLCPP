//
// Created by dewe on 8/18/20.
//

#pragma once
#include <torch/torch.h>
#include "Model.h"
#include "../utility/categorical.h"
#include <iostream>
#include <utility>
#include <vector>

using std::vector;
using namespace torch;

struct FCAC : public nn::Module, Model
{
    nn::Linear inLayer{ nullptr };
    nn::Linear fc1{ nullptr };
    nn::Linear poutLayer{ nullptr };
    nn::Linear voutLayer{ nullptr };


    FCAC(int64_t  inDim, int32_t  outDim)
    {

        inLayer = register_module("inLayer", torch::nn::Linear(inDim, 128));
        fc1 = register_module("fc1", torch::nn::Linear(128, 64));
        poutLayer = register_module("poutLayer", torch::nn::Linear(64, outDim));
        voutLayer = register_module("voutLayer", torch::nn::Linear(64, 1));

    }

    torch::Tensor forward(torch::Tensor x) override
    {
        x = torch::relu(inLayer->forward(x));
        x = torch::relu(fc1->forward(x));
        auto a = poutLayer->forward(x);
        auto v = voutLayer->forward(x);
        vector<Tensor> k = {a, v};
        return cat(k);
    }

    std::tuple<Tensor, Tensor, Tensor, Tensor> fullPass(Tensor state) override
    {
        auto logits = forward(std::move(state)).slice(0,0,-1);
        cpprl::Categorical dist(nullptr, &logits);
        auto action = dist.sample();
        auto logpa =  dist.log_prob(action).unsqueeze(-1);
        auto entropy =  dist.entropy().unsqueeze(-1);
        auto isExploratory = action != torch::argmax(logits.detach());
        return {action, isExploratory, logpa, entropy};

    }
    vector<torch::Tensor> Parameters(bool recurse) override
    {
        return parameters(recurse);
    }

    Tensor selectAction(Tensor state) override
    {
        auto logits = forward(std::move(state)).slice(0,0,-1);
        cpprl::Categorical dist(nullptr, &logits);
        auto action = dist.sample();
        return action;
    }

    Tensor selectGreedyAction(Tensor state) override
    {
        auto logits = forward(std::move(state)).slice(0,0,-1);
        return torch::argmax(logits.detach());
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
    Tensor evaluate_state(Tensor& state)
    {
        auto v = forward(state).index({-1});
        return v;
    }
};

