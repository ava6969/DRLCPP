//
// Created by dewe on 8/18/20.
//

#ifndef DRLCPP_A2C_H
#define DRLCPP_A2C_H


#include "../models/Model.h"
#include "../Experimental/Env.h"
#include "../utility/utils.h"

using ResultVec = vector< vector<double> >;

class A2C {

private:
    Model* acModel;
    float policyLossWeight;
    float valueLossWeight;
    float entropyLossWeight;
    int maxNSteps;
    int nWorkers;

    vector<vector<double>> rewards;
    vector<Tensor> logPas, entropies, values;
    float policyMaxGradNorm;
    Utils::TrainingInfo trainingInfo{};
public:
    A2C(Model* acModel, float policyLossWeight, float valueLossWeight, float entropyLossWeight, int maxNSteps, int nWorkers,
        float policyMaxGradNorm):
    acModel(acModel), policyLossWeight(policyLossWeight), valueLossWeight(valueLossWeight), entropyLossWeight(entropyLossWeight),
    maxNSteps(maxNSteps), nWorkers(nWorkers), policyMaxGradNorm(policyMaxGradNorm){ assert( nWorkers > 1);}

    void OptimizeModel(optim::Adam &pOptim, float gamma, float tau);

    std::tuple<torch::Tensor, vector<bool>> interaction_step(Tensor &states, Env *env);

    std::tuple<ResultVec, double, double, double> train(Env* mainEnv, Env* evalEnv,
                                                        optim::Adam& pOptim, optim::RMSprop& vOptim,
                                                        int seed, float gamma, int saveFREQ,
                                                        int64_t max_minutes, int64_t max_episodes,
                                                        int64_t goal_mean_100_reward);

    std::tuple<double, double> evaluate(Env* evalEnv,  Model* EvalPolicyModel = nullptr, int64_t nEpisode = 1);

    void saveCheckpoint(int64_t episode=-1, Model* model=nullptr);

};


#endif //DRLCPP_A2C_H
