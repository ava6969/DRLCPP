#include "NFQ.h"
#include <algorithm>
#include <sstream>

const int LEAVE_PRINT_EVERY_N_SECS = 60;

using namespace std::chrono;
using std::cout;
using std::endl;
auto currentTime = []() {return duration_cast<duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); };

NFQ::NFQ(
	FCQ& _model,
	Strategy<FCQ>* trainingStrategy,
	Strategy<FCQ>* evalStrategy,
	Device& _device,
	int64_t batchSize,
	float gamma
)
{

	this->batchSize = batchSize;
	model = _model;
	device = _device;
	this->gamma = gamma;
	this->trainingStrategy = trainingStrategy;
	this->evalStrategy = evalStrategy;
}

void NFQ::OptimizeModel(Tensor& states, Tensor& actions, Tensor& rewards, Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim)
{

	// off policy TD Target
	torch::Tensor q_sp = model->forward(nextStates).detach(); // get q_function at s', detach() disconnects reference and frees memory

	auto max_a_q_sp = get<0>(q_sp.max(1)).unsqueeze(1);

	auto target_q_s = rewards + gamma * max_a_q_sp * (1 - terminals);

	auto q_sa = model->forward(states).gather(1, actions); // get estimate of current states

	auto td_errors = q_sa - target_q_s;
	auto value_loss = td_errors.pow(2).mul(0.5).mean();

	optim.zero_grad();
	value_loss.backward();
	optim.step();
	
}

std::tuple<torch::Tensor, bool> NFQ::interaction_step(Tensor& state, Env* env)
{
	// select an epsilon greedy action
	int action= trainingStrategy->selectAction(model, state);
	bool exploratoryActionTaken = trainingStrategy->exploratoryActionTaken;
	// perform a step into the world and store experience
	auto [newState, reward, isTerminal, info] = env->step(action);
	
	// TODO: add truncate - ("isTruncated"!info["isTruncated"]
	bool is_failure = isTerminal && !info["TimeLimit.truncated"];

	// create an experience tuple from response from world
	// store in experience buffer
	experiences.emplace_back(Utils::ExperienceTuple<int>{state, action, reward, newState, is_failure });

	// fill episode info
	trainingInfo.episodeReward.back() += reward;
	trainingInfo.episodeTimestep.back() += 1.0;
	trainingInfo.episodeExploration.back() += int(exploratoryActionTaken);

	return { newState, isTerminal };

}


std::tuple<ResultVec, double, double, double> NFQ::train(Env* env, optim::RMSprop& optimizer, int seed, int epochs, int64_t maxMinutes, int64_t maxEpisodes, int64_t goalMean100Reward)
{
	
	auto training_start = currentTime();
	auto lastDebugTime = -INFINITY;

	// save checkpoint dir
	torch::manual_seed(seed);
	ResultVec result(maxEpisodes);

	double training_time = 0;
	Tensor statesTens = torch::zeros({ batchSize, 4 }).to(device);
	Tensor actionsTens = torch::zeros({ batchSize, 1 }, torch::kI64);
	actionsTens = actionsTens.to(device);
	Tensor rewardsTens = torch::zeros({ batchSize, 1 }, device);
	Tensor nextStatesTens = torch::zeros({ batchSize, 4 }).to(device);
	Tensor terminalsTens = torch::zeros({ batchSize, 1 }, device);

	for (int64_t episode = 1; episode < maxEpisodes+1; episode++)
	{

		auto episode_start = currentTime();
		
		Tensor state = env->reset();
		bool isTerminal = false;

		trainingInfo.episodeReward.emplace_back(0.0);
		trainingInfo.episodeTimestep.emplace_back(0.0);
		trainingInfo.episodeExploration.emplace_back(0.0);

	
		while (true)
		{
			
			std::tie(state, isTerminal) = interaction_step(state, env);
			// batch size represents max size of experience tuple used for optimization
			if (experiences.size() >= (size_t)batchSize)
			{
				for (int i = 0; i < batchSize; i++)
				{
						statesTens[i] = experiences[i].states;
						actionsTens[i] = experiences[i].actions;
						rewardsTens[i] = experiences[i].rewards;
						nextStatesTens[i] = experiences[i].nextStates;
						terminalsTens[i] = experiences[i].terminals;
				}
				for (int i = 0; i < epochs; i++)
					OptimizeModel(statesTens, actionsTens, rewardsTens, nextStatesTens, terminalsTens, optimizer);
			
				experiences.clear();
			}
			if (isTerminal)
			{
				break;
			}
				
		}
		// stats
		int score = 0;
		auto episodeElapsed = currentTime() - episode_start;
		trainingInfo.episodeSeconds.push_back(episodeElapsed);
		training_time += episodeElapsed;
		std::tie(score, std::ignore) = evaluate(env);
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

		result[episode - 1] = { (double)totalStep, mean100Reward, mean100EvalScore, training_time, wallClockElapsed };

		auto _time = currentTime() - lastDebugTime;
		bool reachedDebugTime = _time >= LEAVE_PRINT_EVERY_N_SECS;
		double reachedMaxMinutes = wallClockElapsed >= maxMinutes * 60;
		double reachedMaxEpisodes = episode >= maxEpisodes;
		double reachedGoalMeanReward = mean100EvalScore >= env->RewardThreshold();
		bool trainingIsOver = reachedMaxMinutes || reachedMaxEpisodes || reachedGoalMeanReward;

		auto t = currentTime() - training_start;
		std::ostringstream ss;
		ss  << (int)( t/ 3600) << ":" << ((int)t % 3600) / 60 << ":" << ((int)t % 3600 % 60);
		string elapsedTime = ss.str();
		std::ostringstream debugMessage;
		debugMessage << "el ," << elapsedTime << " ep " << episode - 1 << ", ts " << totalStep << ", ar 10 " << mean10Reward << (char)241 << std10Reward << " 100 " << mean100Reward << (char)241 << std100Reward <<
			" ex 100 " << mean100ExploreRatio << (char)241 << std100ExploreRatio << " ev " << mean100EvalScore << (char)241 << std100EvalScore;

		std::cout << "\r" << debugMessage.str() << std::flush;

		if (reachedDebugTime || trainingIsOver)
		{
			std::cout << "\r" << debugMessage.str() << std::endl;
			lastDebugTime = currentTime();
		}
		if (trainingIsOver)
		{
			if (reachedMaxMinutes) std::cerr << "--> Reached Max Minute x \n";
			if (reachedMaxEpisodes) std::cerr << "--> Reached Max Episodes x \n";
			if (reachedGoalMeanReward) std::cout << "--> Reached Goal Mean reward " << (char)251 << endl;
			break;
		}
	}
		auto [finalEvalScore, scoreStd] = evaluate(env, 100);

		double wallClockTime = currentTime() - training_start;

		std::cout << "Training Complete\n";
		std::cout << "Final Evaluation Score: " << finalEvalScore << (char)241 << scoreStd << " in " << training_time << "s training time, " 
			<< wallClockTime << "s wall-clock time."<<std::endl;
		env->close();
		//TODO: getCleanedCheckpoints();

		return { result, finalEvalScore, training_time, wallClockTime };

}

std::tuple<double, double> NFQ::evaluate(Env* evalEnv, int64_t nEpisode)
{
	vector<double> res;
	
	for (int i = 0; i < nEpisode; i++)
	{
		auto state = evalEnv->reset();
		res.push_back(0);

		while (true)
		{

			int a = evalStrategy->selectAction(model, state);
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

void NFQ::saveCheckpoint(int episode)
{
	// TO DO: Create Directories from class and save
	torch::save(model, "SavedModels\\NFQCartPoleModel." + std::to_string(episode) + ".pth");
}


