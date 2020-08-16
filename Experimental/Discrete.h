#pragma once
#include "Space.h"
class Discrete : public Space <int64_t>
{
	int64_t n{};
public:
	Discrete(int64_t _n) : Space({})
	{
		assert(_n >= 0);
		n = _n;
	}

	Tensor sample()
	{
		return torch::randint(n, { 1 }).to(device);
	}

	bool contains(int64_t x)
	{
		// TODO : check multiple indices
		return x >= 0 && x < n;
	}

	// TODO iostream and == overload

};