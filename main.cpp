#include "../models/FCAC.h"
#include "../algorithms/A2C.h"
#include "../Experimental/OpenAIGymWrapper.h"
#include "../Experimental/MultiprocessGymEnv.h"
#include "memory"



using std::unique_ptr;
using std::make_unique;

int main()
{
    c10::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // hyperParams
    int seed = 15;
    float gamma = 0.99;
    int maxMinutes = 10;
    int maxEpisode = 10000;
    int nWorkers = 8;
    float tau = 0.95;
    // OpenAI Gym
    auto id = "CartPole-v1";
    unique_ptr<MultiProcessEnv> mainEnv = std::make_unique<MultiprocessGymEnv>(id, kCPU, seed, false,nWorkers);
    unique_ptr<Env> evalEnv = std::make_unique<OpenAIGymWrapper>(id, kCPU, seed, false);

    int64_t nS = mainEnv->nS;
    int32_t nA = mainEnv->nA;

    float policy_optimizer_lr = 0.0005;
    unique_ptr<Model> policyModel = std::make_unique<FCAC>(nS, nA);


    auto pOptim(optim::Adam(policyModel->Parameters(), policy_optimizer_lr));
    float entropyLossWeight = 0.001;
    float policyLossWeight = 1.0;
    float valueLossWeight = 0.6;

    A2C agent(policyModel.get(), policyLossWeight, valueLossWeight, entropyLossWeight, 10, nWorkers,1);
    agent.train(mainEnv.get(), evalEnv.get(), pOptim, seed, gamma, tau, 100, maxMinutes, maxEpisode, 475);


    return 0;
}

