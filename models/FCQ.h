#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>

using std::vector;
using namespace torch;

struct FCQImpl : nn::Module
{
	nn::Linear inLayer{ nullptr }; 
	nn::Linear fc1{ nullptr }; 
	//nn::Linear fc2{ nullptr };
	nn::Linear outLayer{ nullptr };
	nn::Linear valueOut{nullptr};
    bool value=false;

	FCQImpl(int64_t  inDim, int32_t  outDim, Device device, bool _value=false)
	{
        value = _value;
		inLayer = register_module("inLayer", torch::nn::Linear(inDim, 512));
		fc1 = register_module("fc1", torch::nn::Linear(512, 128));
		outLayer = register_module("outLayer", torch::nn::Linear(128, outDim));
		if (value)
		    valueOut = register_module("valueOut", torch::nn::Linear(128,1));
		to(device);
	}

	torch::Tensor forward(torch::Tensor x)
	{

		x = torch::relu(inLayer->forward(x));
		x = torch::relu(fc1->forward(x));
		auto a = outLayer->forward(x);
        if (value)
        {
            auto v = valueOut(x);
            v = v.expand_as(a);
            return v + a - a.mean(1,true).expand_as(a);
        }
		return a;
	}


};

TORCH_MODULE(FCQ);

