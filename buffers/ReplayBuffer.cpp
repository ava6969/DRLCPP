#include "ReplayBuffer.h"

using namespace std;
ReplayBuffer::ReplayBuffer(vector<int64_t> stateSpace, int64_t actionSpace, int64_t _maxSize, int64_t _batchSz, Device device, int seed):maxSize(_maxSize), batchSz(_batchSz), device(device)
{
	auto shape = vector<int64_t>{ _maxSize };
	std::copy(begin(stateSpace), end(stateSpace), std::back_inserter(shape));

	ssMem = torch::empty(shape, device);
	asMem = torch::empty({ maxSize, actionSpace }, torch::TensorOptions().device(device).dtype(kInt64));
	rsMem = torch::empty({ maxSize, 1  }, device);
	psMem = torch::empty(shape, device);
	dsMem = torch::empty({ maxSize, 1 }, device);
	torch::manual_seed(seed);
}

std::vector<Tensor> ReplayBuffer::sample(int batch_size)
{
	if (batch_size > -1)
		batchSz = batch_size;

	auto idxVect = torch::slice(torch::randperm(size, TensorOptions().dtype(kLong).device(this->device)), 0, 0, batchSz);

	vector<Tensor> experience = { torch::cat(ssMem.index({ idxVect })),
		torch::cat(asMem.index({idxVect})),
		torch::cat(rsMem.index({idxVect})),
		torch::cat(psMem.index({idxVect})),
		torch::cat(dsMem.index({idxVect}))};

	return experience;
}


int64_t ReplayBuffer::Size() const
{
	return size;
}


