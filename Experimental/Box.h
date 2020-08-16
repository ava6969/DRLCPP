#pragma once
#include "Space.h"
template<class T>
class Box : public Space < T>
{
	Tensor low;
	Tensor high;
	bool bounded_below;
	bool bounded_above;
public:
	Box(T _low, T _high, c10::IntArrayRef shape = {})
	{
		if (shape != {})
		{
			//TODO: check this out 
			assert(std::is_scalar<T>::value || std::is_same<T, Tensor>::value);
			assert(_low.sizes() == shape.size);
			assert(_high.sizes() == shape.sizes());
			assert(_high.sizes() == _low.sizes());
		}

		if (std::is_scalar<T>::value)
		{
			low = torch::full(shape, _low, device);
			high = torch::full(shape, _high, device);
		}
		else
		{
			this->shape = shape;
			low = _low;
			high = _high;

			// TOD0: perform precision check

			bounded_below = std::numeric_limits<double>::min() < low
		}


	}

	Tensor sample()
	{
		return torch::randint(shape, { 1 }).to(device);
	}

	bool contains(int64_t x)
	{
		return x >= 0 && x < n;
	}

	// TODO iostream and == overload

};