//#include "../models/FCAC.h"
//#include "../algorithms/VPG.h"
//#include "../Experimental/OpenAIGymWrapper.h"
//#include "memory"
//
//
//
//using std::unique_ptr;
//using std::make_unique;
//
//int main()
//{
//    c10::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//
//    // hyperParams
//    int seed = 5;
//    float gamma = 1.0;
//    int maxMinutes = 20;
//    int maxEpisode = 10000;
//
//
//    // Custom Env
//
//    // OpenAI Gym
//    auto id = "CartPole-v1";
//    Env* mainEnv = new OpenAIGymWrapper(id, device, seed, false);
//    Env* evalEnv = new OpenAIGymWrapper(id, device, seed, true);
//
//    int64_t nS = mainEnv->nS;
//    int32_t nA = mainEnv->nA;
//
//    float value_optimizer_lr = 0.0007;
//    float policy_optimizer_lr = 0.0005;
//    unique_ptr<FCAC> policyModel = std::make_unique<FCAC>(nS, nA);
//
//    auto state = torch::rand({nS});
//    print(policyModel->evaluate_state(state).slice(0,0,-1));
////    FCV valueModel(nS, device);
////    auto pOptim(optim::Adam(policyModel->Parameters(), policy_optimizer_lr));
////    auto vOptim(optim::RMSprop(valueModel->parameters(), value_optimizer_lr));
////    float entropyLossWeight = 0.001;
////
////
////    VPG agent(valueModel, policyModel.get(), device, INFINITY, 1, entropyLossWeight);
////    agent.train(mainEnv, evalEnv, pOptim, vOptim, seed, gamma, 100, maxMinutes, maxEpisode, 475);
////
//
//
//    // delete CartPole;
//    delete mainEnv;
//    delete evalEnv;
//    return 0;
//}

#include "../models/FCDP.h"
#include "../algorithms/VPG.h"
#include "../Experimental/OpenAIGymWrapper.h"
#include "memory"
#include "../models/FCV.h"
#include "../algorithms/REINFORCE.h"


using std::unique_ptr;
using std::make_unique;
int main()
{
    c10::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // hyperParams
    int seed = 5;
    float gamma = 1.0;
    int maxMinutes = 20;
    int maxEpisode = 10000;


    // Custom Env

    // OpenAI Gym
    auto id = "CartPole-v1";
    Env* mainEnv = new OpenAIGymWrapper(id, device, seed, false);
    Env* evalEnv = new OpenAIGymWrapper(id, device, seed, true);

    int64_t nS = mainEnv->nS;
    int32_t nA = mainEnv->nA;

    float value_optimizer_lr = 0.0007;
    float policy_optimizer_lr = 0.0005;
    unique_ptr<Model> policyModel = std::make_unique<FCDP>(nS, nA, device);
    FCV valueModel(nS, device);
    auto pOptim(optim::Adam(policyModel->Parameters(), policy_optimizer_lr));
    auto vOptim(optim::RMSprop(valueModel->parameters(), value_optimizer_lr));
    float entropyLossWeight = 0.001;

//    REINFORCE agent(policyModel.get(), device);
//    agent.train(mainEnv, evalEnv, pOptim, seed, gamma, 100, maxMinutes, maxEpisode, 475);
    VPG agent(valueModel, policyModel.get(), device, INFINITY, 1, entropyLossWeight);
    agent.train(mainEnv, evalEnv, pOptim, vOptim, seed, gamma, 100, maxMinutes, maxEpisode, 475);



    // delete CartPole;
    delete mainEnv;
    delete evalEnv;
    return 0;
}
