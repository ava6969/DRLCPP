//
// Created by dewe on 8/17/20.
#include "VPG.h"

const int LEAVE_PRINT_EVERY_N_SECS = 60;
using namespace std::chrono;
using std::cout;
using std::endl;
auto currentTime = []() {return duration_cast<duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); };
using namespace torch::indexing;



void VPG::OptimizeModel(optim::Adam& pOptim, optim::RMSprop& vOptim, float gamma)
{
    int T = rewards.size();
    auto rew = torch::from_blob(rewards.data(),{T}).to(device);
    auto discounts = torch::logspace(0,T,T,gamma,dtype(kF32)).to(device);

    vector<float> returns;
    returns.reserve(T);
for(int t = 0; t < T; t++)
        returns.push_back(torch::dot(discounts.index({Slice(None,T-t)}),
                                     rew.index({Slice(t, None)})).item<float>());

    discounts = discounts.index({Slice(None, -1)}).unsqueeze(1);
    auto ret = torch::from_blob(returns.data(),{T}).index({Slice(None, -1)});
    ret = ret.to(device).toType(kF32).unsqueeze(1);

    auto lPas = torch::cat(logPas).unsqueeze(-1);
    auto entr = torch::cat(entropies).unsqueeze(-1);
    auto val = torch::cat(values).unsqueeze(-1);

    auto valError = ret - val;
    auto policyLoss = -(discounts*valError.detach()*lPas ).mean();
    auto entropy_loss = -entr.mean();

    Tensor loss = policyLoss + entropyLossWeight * entropy_loss;

    pOptim.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(policyModel->Parameters(), policyMaxGradNorm);
    pOptim.step();


    auto valLoss = valError.pow(2).mul(0.5).mean();

    vOptim.zero_grad();
    valLoss.backward();
    torch::nn::utils::clip_grad_norm_(valueModel->parameters(), valueMaxGradNorm);
    vOptim.step();

}

std::tuple<ResultVec, double, double, double>
VPG::train(Env *mainEnv, Env *evalEnv, optim::Adam& pOptim, optim::RMSprop& vOptim, int seed, float gamma, int saveFREQ,
                 int64_t maxMinutes, int64_t maxEpisodes, int64_t goal_mean_100_reward)
                 {

                     auto training_start = currentTime();
                     auto lastDebugTime = -INFINITY;

                     // save checkpoint dir
                     torch::manual_seed(seed);
                     ResultVec result(maxEpisodes, vector<double>(5));
                     double training_time = 0;


                     for (int64_t episode = 1; episode < maxEpisodes + 1; episode++) {

                         auto episode_start = currentTime();

                         Tensor state = mainEnv->reset();
                         bool isTerminal = false;
                         bool isTruncated = false;
                         trainingInfo.episodeReward.emplace_back(0.0);
                         trainingInfo.episodeTimestep.emplace_back(0.0);
                         trainingInfo.episodeExploration.emplace_back(0.0);

                         logPas.clear();
                         rewards.clear();
                         values.clear();
                         entropies.clear();

                         while (true)
                         {
                             std::tie(state, isTerminal, isTruncated) = interaction_step(state, mainEnv);
                             if (isTerminal)
                                 break;
                         }
                         bool is_failure = isTerminal && !isTruncated;
                         auto nextVal = is_failure ? 0 : valueModel->forward(state).detach().item<float>();
                         rewards.push_back(nextVal);
                         OptimizeModel( pOptim, vOptim, gamma);

                         // stats
                         int score = 0;
                         auto episodeElapsed = currentTime() - episode_start;
                         trainingInfo.episodeSeconds.push_back(episodeElapsed);
                         training_time += episodeElapsed;
                         std::tie(score, std::ignore) = evaluate(mainEnv);
                         if (episode % saveFREQ == 0)
                             saveCheckpoint(episode - 1);

                         int64_t totalStep = std::accumulate(begin(trainingInfo.episodeTimestep), end(trainingInfo.episodeTimestep), 0);
                         trainingInfo.evaluationScores.push_back(score);

                         // TODO: Fix Eval Scores
                         double mean10Reward = Utils::mean(trainingInfo.episodeReward, 10);
                         double std10Reward = Utils::std(trainingInfo.episodeReward, 10);
                         double mean100Reward = Utils::mean(trainingInfo.episodeReward, 100);
                         double std100Reward = Utils::std(trainingInfo.episodeReward, 100);
                         double mean100EvalScore = Utils::mean(trainingInfo.evaluationScores, 100);
                         double std100EvalScore = Utils::std(trainingInfo.evaluationScores, 100);
                         vector<double> last100ExploreRatios = trainingInfo.explore_ratio(100);
                         double mean100ExploreRatio = Utils::mean(last100ExploreRatios);
                         double std100ExploreRatio = Utils::std(last100ExploreRatios, 100);

                         auto wallClockElapsed = currentTime() - training_start;

                         result[episode - 1] = {(double) totalStep, mean100Reward, mean100EvalScore, training_time, wallClockElapsed};

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
                     auto [finalEvalScore, scoreStd] = evaluate(evalEnv, nullptr, 100);

                     double wallClockTime = currentTime() - training_start;

                     std::cout << "Training Complete\n";
                     std::cout << "Final Evaluation Score: " << finalEvalScore << (char)241 << scoreStd << " in " << training_time << "s training time, "
                               << wallClockTime << "s wall-clock time." << std::endl;
                     saveCheckpoint();

                     return { result, finalEvalScore, training_time, wallClockTime };
}

std::tuple<torch::Tensor, bool, bool> VPG::interaction_step(Tensor &state, Env *env)
{
    float action;

    Tensor actionT;
    Tensor exploratoryT;
    Tensor logPa;
    Tensor newState;
    double reward;
    bool isTerminal;
    Tensor entropy;
    string info;
    std::tie(actionT, exploratoryT, logPa, entropy) = policyModel->fullPass(state);
    std::tie(newState, reward, isTerminal, info) = env->step(action);

    action = actionT.item<float>();
    exploratoryActionTaken = exploratoryT.item<float>();

    logPas.push_back(logPa);
    rewards.push_back(reward);
    entropies.push_back(entropy);
    values.emplace_back(valueModel->forward(state));

    bool isTruncated = info != "";
    trainingInfo.episodeReward.back() += reward;
    trainingInfo.episodeTimestep.back() += 1.0;
    trainingInfo.episodeExploration.back() += int(exploratoryActionTaken);

    return { newState, isTerminal, isTruncated };
}

std::tuple<double, double> VPG::evaluate(Env *evalEnv, Model *EvalPolicyModel, int64_t nEpisode)
{
    vector<double> res;

    for (int i = 0; i < nEpisode; i++)
    {
        auto state = evalEnv->reset();
        res.push_back(0);
        while (true)
        {
            auto a = policyModel->selectGreedyAction(state.unsqueeze(0)).item<float>();
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

void VPG::saveCheckpoint(int64_t episode, Model *model) {
    const std::string path = "/home/dewe/Documents/libtorch/SavedModel/DQNCartPoleModel." + std::to_string((int)episode) + ".pt";
    if (episode > -1)
        policyModel->Save(path);
    else{
        const std::string fpath = "/home/dewe/Documents/libtorch/SavedModel/DQNCartPoleModel.final.pt";
        policyModel->Save( path);
    }
}

