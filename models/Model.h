//
// Created by dewe on 8/17/20.
//
#pragma once
#include <torch/torch.h>
#include <vector>
using namespace torch;

struct Model
{
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual std::vector<torch::Tensor> Parameters(bool recurse=true) = 0;
    virtual void Save(std::string const& name) = 0;
    virtual void Load(std::string const& name) = 0;
    virtual std::tuple<float, bool, Tensor, Tensor> fullPass(Tensor state)
    {
        return {};
    }
    virtual float selectAction(Tensor state)
    {
        return 0;
    }
    virtual float selectGreedyAction(Tensor state)
    {
        return 0;
    }
};
