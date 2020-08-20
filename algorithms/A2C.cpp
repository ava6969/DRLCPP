//
// Created by dewe on 8/18/20.
//

#include "A2C.h"
#include <boost/fusion/algorithm/transformation/flatten.hpp>
#include <boost/fusion/include/flatten.hpp>


const int LEAVE_PRINT_EVERY_N_SECS = 60;
using namespace std::chrono;
using std::cout;
using std::endl;
auto currentTime = []() {return duration_cast<duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); };
using namespace torch::indexing;

std::tuple<torch::Tensor, vector<bool>> A2C::interaction_step(Tensor &states, MultiProcessEnv *env)
{
    Tensor actions;
    Tensor out;
    Tensor logPa;
    Tensor newState;
    Tensor reward = torch::zeros({nWorkers, 1});
    vector<double > rew;
    Tensor val;

    vector<bool> isTerminals;

    Tensor entropy;


    std::tie(out, isExploratories, logPa, entropy) = acModel->fullPass(states);
    actions = out.slice(1,0,-1).squeeze(-1);
    val = out.index({Slice(None,None),-1}).unsqueeze(-1);

    tie(newState, rew, isTerminals, std::ignore) = env->step(actions);

    int i = 0;
    for (auto const& r : rew)
        reward.index_put_({i++}, r);

    logPas.push_back(logPa);
    entropies.push_back(entropy);
    rewards.emplace_back(reward);
    values.push_back(val);

    runningReward += reward;
    runningTimestep += 1;
    runningExplore += isExploratories.unsqueeze(-1).toType(kUInt8);

    return { newState, isTerminals};

}

void A2C::OptimizeModel(optim::Adam &pOptim, float gamma, float tau)
{

    auto stackedLogPa = torch::stack(logPas).squeeze(-1);
    auto stackedEntropies = torch::stack(entropies).squeeze(-1);
    auto stackedVals = torch::stack(values).squeeze(-1);
    int T = rewards.size();

    auto rew = torch::empty({T, nWorkers});
    for (int t = 0; t < T; t++)
    {
        for(int w = 0; w < nWorkers; w++)
            rew.index_put_({t, w}, rewards[t][w]);
    }

    auto discounts = torch::logspace(0,T,T,gamma,dtype(kF32));
    // change to GPU
    auto ret = torch::empty({nWorkers, T}, kDouble);
    // copying RETURNS to RET
    for (int w = 0; w < nWorkers; w++)
    {
        for(int t = 0; t < T; t++)
            ret.index_put_({w, t}, torch::dot(discounts.index({Slice(None,T-t)}),
                                              rew.index({Slice(t, None), w})).item<float>());
    }

    auto vals = stackedVals.clone();
    auto tauDiscounts = torch::logspace(0,T-1,T-1,gamma*tau,dtype(kF32));

    auto advs = rew.slice(0,0,-1) +
            gamma * vals.slice(0,1,vals.size(0)) - vals.slice(0,0,-1);

    // processing gaes
    Tensor gaesT = torch::empty({nWorkers, T-1}, kDouble);
    for (int w =0; w < nWorkers; w++)
    {
        for(int t = 0; t < T-1; t++)
            gaesT.index_put_({w, t} ,torch::dot(tauDiscounts.index({Slice(None,T-1-t)}),
                                      advs.index({Slice(t, None), w})).item<float>());

    }

    auto discountedGaes = discounts.slice(0,0,-1) * gaesT;
    auto flattenedValues = stackedVals.index({Slice(0,-1)} ).view(-1).unsqueeze(1);
    auto flattenedLogPas = stackedLogPa.view(-1).unsqueeze(1);
    auto flattenedEntropies = stackedEntropies.view(-1).unsqueeze(1);

    auto retT = ret.transpose(0,1);
    auto flattenedReturns = retT.slice(0,0,-1).reshape(-1).unsqueeze(1);
    discountedGaes = discountedGaes.transpose(0,1).reshape(-1).unsqueeze(1);

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

std::tuple<ResultVec, double, double, double>
A2C::train(MultiProcessEnv*mainEnv, Env *evalEnv, optim::Adam& pOptim, int seed, float gamma,
           float tau, int saveFREQ, int64_t maxMinutes, int64_t maxEpisodes, int64_t goal_mean_100_reward)
{

    auto training_start = currentTime();
    auto lastDebugTime = -INFINITY;
    double wallClockElapsed;
    // save checkpoint dir
    torch::manual_seed(seed);
    ResultVec result(maxEpisodes, vector<double>(5));
    double training_time = 0;


    Tensor states = mainEnv->reset();
    Tensor isTerminals = torch::zeros({nWorkers}, kBool);

    runningTimestep =torch::zeros({nWorkers, 1}, kInt64);
    runningReward =torch::zeros({nWorkers, 1}, kF64);
    runningExplore =torch::zeros({nWorkers, 1}, kInt64);
    runningSecond =torch::full({nWorkers, 1},currentTime(),kF64);

    int episode = 1;
    int nStepStart;

    double mean100EvalScore = 0.0;
    double mean10Reward;
    double mean100Reward;
    double std100EvalScore = 0.0;
    double std10Reward;
    double std100Reward;
    double mean100ExploreRatio;
    double std100ExploreRatio;
    int64_t totalStep;
    int step = 1;
    while (true)
    {
            vector<bool> temp(nWorkers);
            std::tie(states, temp) = interaction_step(states, mainEnv);

            vector<int> isTerm(temp.begin(), temp.end());
            Tensor isTermT = torch::zeros({nWorkers});
            int i = 0;
            for (auto const& t : isTerm)
                isTermT.index_put_({i++}, t);
            bool sum = isTermT.sum(0).item<bool>();
            isTermT = isTermT.unsqueeze(-1);

            if (sum || step - nStepStart == maxNSteps)
            {
                // enforce timelimit
                auto nextValues = acModel->evaluate_state(states).detach() * (1 - isTermT.toType(kI8));
                cout << nextValues << endl;
                rewards.push_back(nextValues);
                values.push_back(nextValues);
                OptimizeModel( pOptim, gamma, tau);
                logPas.clear();
                rewards.clear();
                values.clear();
                entropies.clear();
                nStepStart = step;
            }
            if (sum)
            {
                auto episode_done = currentTime();
                auto[evalScore, _std] = evaluate(evalEnv);
                if (nStepStart/100 == saveFREQ)
                    saveCheckpoint(episode);

                for(int i = 0; i < nWorkers; ++i)
                {
                    if (isTermT[i].item<int>())
                    {
                        // stats
                        states[i] = mainEnv->reset(i);
                        trainingInfo.episodeTimestep.push_back(runningTimestep[i].item<float>());
                        trainingInfo.episodeReward.push_back(runningReward[i].item<float>());

                        trainingInfo.episodeExploration.push_back((runningExplore[i].true_divide(runningTimestep[i])).item<float>());
                        trainingInfo.episodeSeconds.push_back(episode_done - runningSecond[i].item<float>());
                        training_time += trainingInfo.episodeSeconds.back();
                        trainingInfo.evaluationScores.push_back(evalScore);

                        episode += 1;

                        mean10Reward = mean10Reward = Utils::mean(trainingInfo.episodeReward, 10);
                         std10Reward = Utils::std(trainingInfo.episodeReward, 10);
                        mean100Reward = Utils::mean(trainingInfo.episodeReward, 100);
                         std100Reward =  Utils::std(trainingInfo.episodeReward, 100);
                        mean100EvalScore = Utils::mean(trainingInfo.evaluationScores, 100);
                         std100EvalScore = Utils::std(trainingInfo.evaluationScores, 100);
                        vector<double> last100ExploreRatios = trainingInfo.explore_ratio( 100);
                         mean100ExploreRatio = Utils::mean(last100ExploreRatios);
                         std100ExploreRatio = Utils::std(last100ExploreRatios, 100);
                        totalStep = std::accumulate(begin(trainingInfo.episodeTimestep), end(trainingInfo.episodeTimestep), 0);
                        wallClockElapsed = currentTime() - training_start;
                        result[episode - 1] = {(double) totalStep, mean100Reward, mean100EvalScore, training_time, wallClockElapsed};
                    }
                }
                // debug stuffs
                auto _time = currentTime() - lastDebugTime;
                bool reachedDebugTime = _time >= LEAVE_PRINT_EVERY_N_SECS;
                bool reachedMaxMinutes = wallClockElapsed >= maxMinutes * 60;
                bool reachedMaxEpisodes = episode >= maxEpisodes;
                bool reachedGoalMeanReward = mean100EvalScore >= goal_mean_100_reward;
                bool trainingIsOver = reachedMaxMinutes || reachedMaxEpisodes || reachedGoalMeanReward;

                auto t = currentTime() - training_start;
                std::ostringstream ss;
                ss << (int) (t / 3600) << ":" << ((int) t % 3600) / 60 << ":" << ((int) t % 3600) % 60;
                string elapsedTime = ss.str();
                std::ostringstream debugMessage;
                debugMessage << "el ," << elapsedTime << " ep " << episode - 1 << ", ts " << totalStep << ", ar 10 "
                             << mean10Reward << (char) 241 << std10Reward << " 100 " << mean100Reward << (char) 241
                             << std100Reward <<
                             " ex 100 " << mean100ExploreRatio << (char) 241 << std100ExploreRatio << " ev " << mean100EvalScore
                             << (char) 241 << std100EvalScore;

                std::cout << "\r" << debugMessage.str() << std::flush;

                if (reachedDebugTime || trainingIsOver) {
                    std::cout << "\r" << debugMessage.str() << std::endl;
                    lastDebugTime = currentTime();
                }
                if (trainingIsOver) {
                    if (reachedMaxMinutes) std::cerr << "--> Reached Max Minute x \n";
                    if (reachedMaxEpisodes) std::cerr << "--> Reached Max Episodes x \n";
                    if (reachedGoalMeanReward) std::cout << "--> Reached Goal Mean reward " << (char) 251 << endl;
                    break;
                }
            }

            // reset runningSteps

            auto k = 1 - isTermT.toType(kI8);
            runningTimestep *= k;
            runningReward *= k;
            runningExplore *= k;
            runningSecond.index_put_({isTermT.toType(kBool)}, currentTime());
    }
    auto [finalEvalScore, scoreStd] = evaluate(evalEnv, nullptr, 100);

    double wallClockTime = currentTime() - training_start;

    std::cout << "Training Complete\n";
    std::cout << "Final Evaluation Score: " << finalEvalScore << (char)241 << scoreStd << " in " << training_time << "s training time, "
              << wallClockTime << "s wall-clock time." << std::endl;
    saveCheckpoint();

    return { result, finalEvalScore, training_time, wallClockTime };
}

std::tuple<double, double> A2C::evaluate(Env *evalEnv, Model *EvalPolicyModel, int64_t nEpisode)
{
    vector<double> res;

    for (int i = 0; i < nEpisode; i++)
    {
        auto state = evalEnv->reset();
        res.push_back(0);
        while (true)
        {
            auto a = acModel->selectGreedyAction(state.unsqueeze(0)).item<float>();
            auto out = evalEnv->step(a);
            state = get<0>(out);
            res.back() += get<1>(out);
            if (get<2>(out))
                break;
        }
    }
    double mean = Utils::mean(res);
    double std = Utils::std(res);
    return { mean, std };
}

void A2C::saveCheckpoint(int64_t episode, Model *model) {
    const std::string path = "/home/dewe/CLionProjects/DRLCPP/SavedModel/DQNCartPoleModel." + std::to_string((int)episode) + ".pt";
    if (episode > -1)
        acModel->Save(path);
    else{
        const std::string fpath = "/home/dewe/CLionProjects/DRLCPP/SavedModel/DQNCartPoleModel.final.pt";
        acModel->Save( path);
    }
}