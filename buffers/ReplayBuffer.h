#pragma once
#include <cstdint>
#include "../utility/utils.h"
#include <utility>
#include <vector>
#include <torch/torch.h>

using namespace torch;
using namespace Utils;
class ReplayBuffer
{
private:
	Tensor ssMem;
	Tensor asMem;
	Tensor rsMem;
	Tensor psMem;
	Tensor dsMem;

	int64_t maxSize;
	int64_t batchSz;
	int64_t idx{0};
	int64_t size{0};

	Device device ;

public:
	ReplayBuffer(vector<int64_t> stateSpace, int64_t actionSpace, int64_t maxSize, int64_t batchSz, Device device, int seed);


	void store(ExperienceTuple sample);

	std::vector<Tensor> sample(int batch_size = -1);

	int64_t Size() const;
	inline int64_t BatchSize() const { return batchSz; }
};


inline void ReplayBuffer::store(ExperienceTuple sample)
{
	auto [s, a, r, p, d] = std::move(sample);

	ssMem[idx] = s;
	asMem[idx] = a;
	rsMem[idx] = r;
	psMem[idx] = p;
	dsMem[idx] = d;

	++idx;
	idx = idx % maxSize;
	size += 1;
	size = std::min<int64_t>(size, maxSize);
}