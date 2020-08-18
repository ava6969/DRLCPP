

#include "../algorithms/DQN.h"
#include "../Experimental/OpenAIGymWrapper.h"
#include <SDL/SDL.h>
#include "../Experimental/ALEWrapper.h"
#include "ale_interface.hpp"

int main()
{
    c10::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // hyperParams
    int seed = 5;
    int nWarmUpBatches = 5;
    int updateEvery = 10;
    int gamma = 1.0;
    int maxMinutes = 20;
    int maxEpisode = 10000;


    // Custom Env
    //Env* CartPole = new CartPoleEnv(device, seed);
    /*    int64_t nS = CartPole->getStateSpace();
    int32_t nA = CartPole->getActionSpace();*/

    // OpenAI Gym
    auto id = "CartPole-v1";
    Env* mainEnv = new OpenAIGymWrapper(id, device, seed, false);
    Env* evalEnv = new OpenAIGymWrapper(id, device, seed, true);

    int64_t nS = mainEnv->nS;
    int32_t nA = mainEnv->nA;

//  ALE INterface
//    auto id = "../atari_roms/breakout.bin";
//    Env* mainEnv = new ALEWrapper(id, device, seed);
//    Env* evalEnv = new ALEWrapper(id, device, seed, true, true);



    float value_optimizer_lr = 0.0005;
    unique_ptr<Model> onlineModel = std::make_unique<FCQ>(nS, nA, device, true);
    unique_ptr<Model> targetModel = std::make_unique<FCQ>(nS, nA, device, true);

    auto evalStrategy = new Strategy();
    auto trainStrategy = new EGreedyLinearStrategy(1.0, 0.3);

   auto buffer = new ReplayBuffer({ nS }, nA, 50000, 64, device, seed);
    auto optimizer(optim::RMSprop(onlineModel->Parameters(), value_optimizer_lr));

    DQN agent(onlineModel.get(), targetModel.get(), buffer, trainStrategy, evalStrategy,
              device, nWarmUpBatches, updateEvery, std::numeric_limits<float>::max(),true,0.1);

    agent.train(mainEnv, evalEnv, optimizer, seed, gamma, 100, maxMinutes, maxEpisode, 495);

    // delete CartPole;
    delete mainEnv;
    delete evalEnv;
    delete evalStrategy;
    delete trainStrategy;
    delete buffer;
    return 0;
}
