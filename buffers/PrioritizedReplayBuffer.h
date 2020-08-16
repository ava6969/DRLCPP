//
// Created by dewe on 8/16/20.
//

#ifndef DRLCPP_PRIORITIZEDREPLAYBUFFER_H
#define DRLCPP_PRIORITIZEDREPLAYBUFFER_H

#include <torch/torch.h>
#include "../utility/utils.h"

using namespace torch;
using namespace Utils;
using namespace torch::indexing;

class PrioritizedReplayBuffer {

public:
    PrioritizedReplayBuffer(int maxSamples, int batchSz, Device device ,bool rankBased=false,
                            float alpha=0.6, float beta0=0.1, float betaRate=0.99992):
                            device(device),
                            maxSamples(maxSamples),
                            batchSz(batchSz),
                            rankBased(rankBased),
                            alpha(alpha),
                            beta(beta0),
                            beta0(beta0),
                            betaRate(betaRate){

        memory = torch::empty({maxSamples, 2}, dtype<IntArrayRef>().device(device)); // test this

    }

    void update(int idx, Tensor const& tdErrors);

    template<typename T>
    void store(ExperienceTuple<T> sample);

    std::vector<Tensor> sample(int batch_size = -1);

    [[nodiscard]] int64_t Size() const {return nEnteries;}

    [[nodiscard]] inline int64_t BatchSize() const { return batchSz; }

private:
    void updateBeta();

    int maxSamples;
    Tensor memory;
    int batchSz;
    int nEnteries{0};
    int nextIdx{0};
    int tdErrorIdx{0};
    int sampleIdx{1};
    bool rankBased;
    float alpha;
    float beta;
    float beta0;
    float betaRate;
    Device device ;
};


#endif //DRLCPP_PRIORITIZEDREPLAYBUFFER_H
