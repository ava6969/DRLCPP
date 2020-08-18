

#include "../algorithms/DQN.h"
#include <SDL/SDL.h>
#include "../Experimental/ALEWrapper.h"
#include "ale_interface.hpp"
#include "../models/AtariCNN.h"
#include <c10/cuda/CUDACachingAllocator.h>

int main()
{

    c10::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    c10::cuda::CUDACachingAllocator::emptyCache();
    // hyperParams
    int seed = 5;
    int nWarmUpBatches = 500;
    int updateEvery = 100;
    float gamma = 0.99;
    int maxMinutes = 20;
    int maxEpisode = 10000;

//  ALE INterface
    auto id = "../atari_roms/pong.bin";
    Env* mainEnv = new ALEWrapper(id, device, seed, true, false);
    Env* evalEnv = new ALEWrapper(id, device, seed, false, false);

    int32_t nA = mainEnv->nA;

    float optimizer_lr = 0.00025;
    unique_ptr<Model> onlineModel = std::make_unique<AtariCNN>(nA, device, true);
    unique_ptr<Model> targetModel = std::make_unique<AtariCNN>(nA, device, true);

    auto evalStrategy = new Strategy();
    auto trainStrategy = new EGreedyLinearStrategy(1.0, 0.1, 10000);

    auto buffer = new ReplayBuffer({ 4, 84, 84 }, nA, 10000, 32, device, seed);
    auto optimizer(optim::RMSprop(onlineModel->Parameters(),
                                  optim::RMSpropOptions(optimizer_lr).
                                  momentum(0.95).eps(0.01)));

    DQN agent(onlineModel.get(), targetModel.get(), buffer, trainStrategy, evalStrategy,
              device, nWarmUpBatches, updateEvery, std::numeric_limits<float>::max(),true,0.1);

    agent.train(mainEnv, evalEnv, optimizer, seed, gamma, 100, maxMinutes, maxEpisode, 19);

    // delete CartPole;
    delete mainEnv;
    delete evalEnv;
    delete evalStrategy;
    delete trainStrategy;
    delete buffer;
    return 0;
}
