//
// Created by dewe on 8/16/20.
//

#include "PrioritizedReplayBuffer.h"


const double EPS = std::pow(10,-6);

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


void PrioritizedReplayBuffer::store(Tensor const& sample)
{
    float priority = 1.0;
    if (nEnteries > 0)
        priority = memory.index({Slice(None,nEnteries),tdErrorIdx}).max().item<float>();
    memory.index_put_({nextIdx, tdErrorIdx}, priority);
    memory.index_put_({nextIdx,sampleIdx},sample);
    nEnteries = std::min<int>(nEnteries+1, maxSamples);
    nextIdx++;
    nextIdx = nextIdx % maxSamples;
}

std::vector<Tensor> PrioritizedReplayBuffer::sample(int batch_size)
{
    batchSz = batch_size == -1 ? batchSz : batch_size;
    updateBeta();
    auto entries = memory.index({Slice(None,nEnteries)});
    Tensor priorities;
    if (rankBased)
        priorities = 1 / (torch::arange(nEnteries) + 1);
    else
        priorities = entries.index({Slice(None,None),tdErrorIdx}) + EPS;

    auto scaledPriorites = torch::pow(priorities, alpha);
    auto probs = scaledPriorites / scaledPriorites.sum();

    auto weights = torch::pow(nEnteries * probs, -beta);
    auto normalizedWeights = weights / weights.max();
    // test
    auto idxs = torch::multinomial(probs, batch_size, false );


    auto samples = entries.index({idxs});
    auto batchTypes = torch::cat(samples.index({Slice(None, None), sampleIdx})).unsqueeze(-1).transpose(0,1);
    auto samplesStacks = torch::cat(batchTypes).unsqueeze(-1);
    auto idxStacks = torch::cat(idxs).unsqueeze(-1);
    auto weightsStack = torch::cat(normalizedWeights.index({idxs}));

//    vector<Tensor> experience = { torch::cat(ssMem.index({ idxVect })),
//                                  torch::cat(asMem.index({idxVect})),
//                                  torch::cat(rsMem.index({idxVect})),
//                                  torch::cat(psMem.index({idxVect})),
//                                  torch::cat(dsMem.index({idxVect}))};
    return {idxStacks, weightsStack, samplesStacks};



}
