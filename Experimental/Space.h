#pragma once
#include <vector>
#include <torch/torch.h>
using namespace torch;


template <typename dtype>
class Space
{
protected:
	Device device = kCPU;
	c10::IntArrayRef shape;
public:
	Space(c10::IntArrayRef _shape):shape(_shape){}

	virtual Tensor sample() = 0;

	void seed(int seed)
	{
		torch::manual_seed(seed);
	}
	virtual bool contains() = 0;

};

