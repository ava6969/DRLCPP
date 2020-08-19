//
// Created by dewe on 8/17/20.
//

#pragma once

#include "../models/Model.h"
#include "../Experimental/Env.h"
#include "../utility/utils.h"

using ResultVec = vector< vector<double> >;

class REINFORCE
        {
public:
    /*Constructor*/
    REINFORCE(Model* onlineModel, Device& _device
    ):onlineModel(onlineModel),device(_device){}

    void OptimizeModel(optim::Adam& optim, float gamma);

    std::tuple<torch::Tensor, bool> interaction_step(Tensor& state, Env* env);

    std::tuple<ResultVec, double, double, double> train(Env* mainEnv, Env* evalEnv, optim::Adam& optimizer,
                                                        int seed, float gamma, int saveFREQ,
                                                        int64_t max_minutes, int64_t max_episodes,
                                                        int64_t goal_mean_100_reward);

    std::tuple<double, double> evaluate(Env* evalEnv,  Model* EvalPolicyModel = nullptr, int64_t nEpisode = 1);

    void saveCheckpoint(int64_t episode=-1, Model* model=nullptr);

private:
    Model* onlineModel{ nullptr };
    bool exploratoryActionTaken = false;
    Device device = torch::kCPU;
    Utils::TrainingInfo trainingInfo{};
    vector<float> rewards;
    vector<Tensor> logPas;
};
