//
// Created by dewe on 8/16/20.
//

#include "PrioritizedReplayBuffer.h"

void PrioritizedReplayBuffer::update(int idx, const Tensor &tdErrors)
{
    memory.index_put_({idx, tdErrorIdx}, torch::abs(tdErrors));
    if(rankBased)
    {
        auto sortedArg = memory.index({Slice(0, nEnteries, None), tdErrorIdx}).argsort();
        auto indexedSorted = sortedArg.index({Slice(None,None,-1)});
        memory.index_put_({Slice(None, nEnteries)}, memory.index({indexedSorted}));
    }
}

template<typename T>
void PrioritizedReplayBuffer::store(ExperienceTuple<T> sample)
{

}

std::vector<Tensor> PrioritizedReplayBuffer::sample(int batch_size) {
    return std::vector<Tensor>();
}
