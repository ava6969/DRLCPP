#include "DQN.h"
const int LEAVE_PRINT_EVERY_N_SECS = 60;
using namespace std::chrono;
using std::cout;
using std::endl;
auto currentTime = []() {return duration_cast<duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); };

void DQN::OptimizeModel(Tensor& states, Tensor& actions, Tensor& rewards, Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim, int gamma)
{
	// off policy TD Target

    int batchSz = states.size(0);
    Tensor max_a_q_sp{};
    if (DDQN)
    {
        auto argMax_a_q_sp1  = onlineModel->forward(nextStates).detach().max(1);
        auto argMax_a_q_sp = get<1>(argMax_a_q_sp1);
        torch::Tensor q_sp = targetModel->forward(nextStates).detach(); // get q_function at s', detach() disconnects reference and frees memory
        max_a_q_sp = q_sp.gather(1, argMax_a_q_sp.unsqueeze(1));
    }
    else
    {
        torch::Tensor q_sp = targetModel->forward(nextStates).detach(); // get q_function at s', detach() disconnects reference and frees memory
        max_a_q_sp = get<0>(q_sp.max(1)).unsqueeze(1);
    }

	auto target_q_s = rewards + gamma * max_a_q_sp * (1 - terminals);
	auto q_sa = onlineModel->forward(states).gather(1, actions); // get estimate of current states
	auto td_errors = q_sa - target_q_s;
	auto value_loss = td_errors.pow(2).mul(0.5).mean();

	optim.zero_grad();
	value_loss.backward();
	torch::nn::utils::clip_grad_norm_(onlineModel->Parameters(), maxGradientNorm);
	optim.step();

}

void DQN::OptimizeModel(Tensor& idx, Tensor& weights, Tensor& states, Tensor& actions, Tensor& rewards, Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim, float gamma)
{
    // off policy TD Target

    int batchSz = states.size(0);
    Tensor max_a_q_sp{};
    if (DDQN)
    {
        auto argMax_a_q_sp1  = onlineModel->forward(nextStates).detach().max(1);
        auto argMax_a_q_sp = get<1>(argMax_a_q_sp1);
        torch::Tensor q_sp = targetModel->forward(nextStates).detach(); // get q_function at s', detach() disconnects reference and frees memory
        max_a_q_sp = q_sp.gather(1, argMax_a_q_sp.unsqueeze(1));
    }
    else
    {
        torch::Tensor q_sp = targetModel->forward(nextStates).detach(); // get q_function at s', detach() disconnects reference and frees memory
        max_a_q_sp = get<0>(q_sp.max(1)).unsqueeze(1);
    }

    auto target_q_s = rewards + gamma * max_a_q_sp * (1 - terminals);
    auto q_sa = onlineModel->forward(states).gather(1, actions); // get estimate of current states
    auto td_errors = q_sa - target_q_s;
    auto value_loss = td_errors.pow(2).mul(0.5).mean();

    optim.zero_grad();
    value_loss.backward();
    torch::nn::utils::clip_grad_norm_(onlineModel->Parameters(), maxGradientNorm);
    optim.step();

}

std::tuple<torch::Tensor, bool> DQN::interaction_step(Tensor& state, Env* env)
{
	// select an epsilon greedy action
	float action = (float)trainingStrategy->selectAction(onlineModel, state.unsqueeze(0));
	exploratoryActionTaken = trainingStrategy->exploratoryActionTaken;
	// perform a step into the world and store experience
	auto [newState, reward, isTerminal, info] = env->step(action);


	bool is_failure = isTerminal && info != "TimeLimit.truncated";

	// create an experience tuple from response from world
	// store in experience buffer

	auto exp = Utils::ExperienceTuple{ state, action, reward, newState, is_failure };
	buffer->store(exp);

	// fill episode info
	trainingInfo.episodeReward.back() += reward;
	trainingInfo.episodeTimestep.back() += 1.0;
	trainingInfo.episodeExploration.back() += (float)exploratoryActionTaken;

	return { newState, isTerminal };
}

std::tuple<ResultVec, double, double, double> DQN::train(Env* mainEnv, Env* evalEnv, optim::RMSprop& optimizer,
                                                        int seed, float gamma, int saveFREQ,
                                                        int64_t maxMinutes, int64_t maxEpisodes,
                                                        int64_t goal_mean_100_reward)
{
	auto training_start = currentTime();
	auto lastDebugTime = -INFINITY;

	// save checkpoint dir
	torch::manual_seed(seed);
	ResultVec result(maxEpisodes, vector<double>(5));
	int batchSize = buffer->BatchSize();
	double training_time = 0;


	for (int64_t episode = 1; episode < maxEpisodes + 1; episode++) {

        auto episode_start = currentTime();

        Tensor state = mainEnv->reset();
        bool isTerminal = false;

        trainingInfo.episodeReward.emplace_back(0.0);
        trainingInfo.episodeTimestep.emplace_back(0.0);
        trainingInfo.episodeExploration.emplace_back(0.0);


        while (true) {

            std::tie(state, isTerminal) = interaction_step(state, mainEnv);
            // batch size represents max size of experience tuple used for optimization
            int min_samples = batchSize * nWarmupbatches;
            if (buffer->Size() > min_samples) {
                auto exp = buffer->sample();
                assert(experiences.size() == 5);

                OptimizeModel(exp[0], exp[1], exp[2], exp[3], exp[4], optimizer);
            }
            if (std::accumulate(begin(trainingInfo.episodeTimestep), end(trainingInfo.episodeTimestep), 0)
                % updateTargetEverySteps == 0)
                updateNetwork();

            if (isTerminal)
                break;
        }
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
        vector<double> last100ExploreRatios = trainingInfo.explore_ratio( 100);
        double mean100ExploreRatio = Utils::mean(last100ExploreRatios);
        double std100ExploreRatio = Utils::std(last100ExploreRatios, 100);

        auto wallClockElapsed = currentTime() - training_start;

        result[episode - 1] = {(double) totalStep, mean100Reward, mean100EvalScore, training_time, wallClockElapsed};

        auto _time = currentTime() - lastDebugTime;
        bool reachedDebugTime = _time >= LEAVE_PRINT_EVERY_N_SECS;
        bool reachedMaxMinutes = wallClockElapsed >= (int)maxMinutes * 60;
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


void DQN::updateNetwork()
{
	int i = 0;
	for (const auto& weight : onlineModel->Parameters())
	{
	    auto const& targetRatio = (1.0 - tau) * targetModel->Parameters()[i].data();
	    auto const& onlineRatio = tau * weight.data();
	    auto mixed_net = targetRatio + onlineRatio;
		targetModel->Parameters()[i].data().copy_(mixed_net);
		i++;
	}
}

std::tuple<double, double> DQN::evaluate(Env* evalEnv, Model* EvalPolicyModel, int64_t nEpisode)
{
	vector<double> res;

	for (int i = 0; i < nEpisode; i++)
	{
		auto state = evalEnv->reset();
		res.push_back(0);

		while (true)
		{

			float a = evalStrategy->selectAction(onlineModel, state.unsqueeze(0));
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


void DQN::saveCheckpoint(int64_t episode, Model* model)
{
	// TO DO: Create Directories from class and save

	const std::string path = "/home/dewe/Documents/libtorch/SavedModel/DQNCartPoleModel." + std::to_string((int)episode) + ".pt";
	if (episode > -1)
        onlineModel->Save(path);
	else{
        const std::string fpath = "/home/dewe/Documents/libtorch/SavedModel/DQNCartPoleModel.final.pt";
        onlineModel->Save( path);
	}


}



