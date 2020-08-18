//
// Created by dewe on 8/17/20.
//

#ifndef DRLCPP_REINFORCE_H
#define DRLCPP_REINFORCE_H

#include "../models/Model.h"
#include "../Experimental/Env.h"
#include "../utility/utils.h"
#include "../models/FCV.h"

using ResultVec = vector< vector<double> >;

class VPG {
public:
    /*Constructor*/
    VPG(FCV valueModel, Model* policyModel, Device& _device,
        float valueMaxGradNorm = INFINITY, float policyMaxGradNorm = INFINITY, float entropyLossWeight=0.001
    ):policyModel(policyModel),valueModel(valueModel),
    valueMaxGradNorm(valueMaxGradNorm), policyMaxGradNorm(policyMaxGradNorm), entropyLossWeight(entropyLossWeight),
    device(_device){}

    void OptimizeModel(optim::Adam& pOptim, optim::Adam& vOptim, float gamma);

    std::tuple<torch::Tensor, bool> interaction_step(Tensor& state, Env* env);

    std::tuple<ResultVec, double, double, double> train(Env* mainEnv, Env* evalEnv,optim::Adam& pOptim,
                                                        optim::Adam& vOptim,
                                                        int seed, float gamma, int saveFREQ,
                                                        int64_t max_minutes, int64_t max_episodes,
                                                        int64_t goal_mean_100_reward);

    std::tuple<double, double> evaluate(Env* evalEnv,  Model* EvalPolicyModel = nullptr, int64_t nEpisode = 1);

    void saveCheckpoint(int64_t episode=-1, Model* model=nullptr);

private:
    Model* policyModel{ nullptr };
    FCV valueModel{ nullptr };
    float valueMaxGradNorm;
    float policyMaxGradNorm;
    float entropyLossWeight;
    bool exploratoryActionTaken = false;
    Device device = torch::kCPU;
    Utils::TrainingInfo trainingInfo{};

    vector<float> rewards;
    vector<Tensor> logPas, entropies, values;
};


#endif //DRLCPP_REINFORCE_H
