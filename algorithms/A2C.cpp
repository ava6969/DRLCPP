//
// Created by dewe on 8/18/20.
//

#include "A2C.h"

using namespace torch::indexing;
std::tuple<torch::Tensor, vector<bool>> A2C::interaction_step(Tensor &states, Env *env)
{
    float actions;
    Tensor logPa;
    Tensor newState;
    vector<double> reward;
    Tensor val;
    vector<bool> isTerminals;
    Tensor entropy;
    string infos;

    std::tie(actions, is_exploratory, logPa, entropies, val) = acModel->fullPass(states);
    tie(newState, reward, isTerminals, std::ignore) = env->step(actions);
    logPas.push_back(logPa); entropies.push_back(entropy);
    rewards.push_back(reward); values.push_back(val);

    trainingInfo.episodeReward.back() += reward;
    trainingInfo.episodeTimestep.back() += 1.0;
    trainingInfo.episodeExploration.back() += int(is_exploratory);

    return { newState, isTerminals };

}

void A2C::OptimizeModel(optim::Adam &pOptim, float gamma, float tau)
{
    auto stackedLogPa = torch::stack(logPas).squeeze();
    auto stackedEntropies = torch::stack(entropies).squeeze();
    auto stackedVals = torch::stack(values).squeeze();

    int T = rewards.size();
    auto rew = torch::from_blob(rewards.data(),{T, nWorkers});
    auto discounts = torch::logspace(0,T,T,gamma,dtype(kF32));

    // change to GPU
    vector<vector<double>> returns(nWorkers);
    for (int w =0; w < nWorkers; w++)
    {
        vector<double> temp(nWorkers);
        for(int t = 0; t < T; t++)
            temp.push_back(torch::dot(discounts.index({Slice(None,T-t)}),
                                         rew.index({Slice(t, None), w})).item<float>());
        returns.push_back(temp);
    }

    auto vals = stackedVals.clone();
    auto tauDiscounts = torch::logspace(0,T-1,T-1,gamma*tau,dtype(kF32));
// TODO CHECK THIS
    auto advs = rew.slice(1,0,-1) +
            gamma * vals.slice(0,1,vals.size(0)) - vals.slice(0,0,-1);

    vector<vector<double>>gaes(nWorkers);
    for (int w =0; w < nWorkers; w++)
    {
        vector<double> temp(nWorkers);
        for(int t = 0; t < T-1; t++)
            temp.push_back(torch::dot(tauDiscounts.index({Slice(None,T-1-t)}),
                                      advs.index({Slice(t, None), w})).item<float>());
        gaes.push_back(temp);
    }
    Tensor gaesT = torch::from_blob(gaes.data(),{T, nWorkers});
    auto discountedGaes = discounts.slice(0,0,-1) * gaesT.slice(1,0,T);

    auto flattenedValues = stackedVals.index({Slice(0,-1)} ).view(-1).unsqueeze(1);
    auto flattenedLogPas = stackedLogPa.view(-1).unsqueeze(1);
    auto flattenedEntropies = stackedEntropies.view(-1).unsqueeze(1);
    Tensor ret = torch::from_blob(gaes.data(),{nWorkers , T});
    auto flattenedReturns = ret.slice(0,0,-1).view(-1).unsqueeze(1);
    discountedGaes = discountedGaes.transpose(0,1).view(-1).unsqueeze(1);

    T -= 1;
    T *= nWorkers;
    assert((flattenedReturns.sizes() == {T, 1}));
    assert((flattenedValues.sizes() == {T, 1}));
    assert((flattenedLogPas.sizes() == {T, 1}));
    assert((flattenedEntropies.sizes() == {T, 1}));

    auto valueErr = flattenedReturns.detach() - flattenedValues;
    auto valueLoss = valueErr.pow(2).mul(0.5).mean();
    auto policyLoss = -(discountedGaes.detach() * flattenedLogPas).mean();
    auto entropyLoss = -flattenedEntropies.mean();
    auto loss = policyLossWeight*policyLoss + valueLossWeight*valueLoss + entropyLossWeight*entropyLoss;

    pOptim.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(acModel->Parameters(), policyMaxGradNorm);
    pOptim.step();

}
