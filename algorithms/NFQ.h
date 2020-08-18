#pragma once
#include <memory>
#include "../utility/utils.h"
#include "../Experimental/Env.h"
#include <vector>
#include <torch/torch.h>
#include <tuple>
#include "../models/FCQ.h"
#include "../utility/Strategy.h"
#include <cstdint>

using std::vector;
using std::unique_ptr;

using ResultVec = vector< vector<double> >;
class NFQ
{
public:
	/*Constructor*/
	NFQ(Model* model,
		Strategy*trainingStrategy,
		Strategy*evalStrategy,
		Device& _device,
		int64_t batchSize,
		float gamma
	);

	void OptimizeModel(Tensor& states, Tensor& actions, Tensor& rewards, Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim);

	std::tuple<torch::Tensor, bool> interaction_step(Tensor& state, Env* env);

	std::tuple<ResultVec, double, double, double> train(Env* env, optim::RMSprop& optimizer, int seed, int epochs,
															int64_t max_minutes, int64_t max_episodes, 
															int64_t goal_mean_100_reward=std::numeric_limits<int>::max());

	std::tuple<double, double> evaluate(Env* evalEnv, int64_t nEpisode = 1);

	void saveCheckpoint(int episode);


private:


	int64_t batchSize;
	vector<Utils::ExperienceTuple> experiences;
    Model* model{nullptr};
	bool exploratoryActionTaken = false;
	Utils::TrainingInfo trainingInfo{};

	Device device = torch::kCPU;

	Strategy* trainingStrategy;
	Strategy* evalStrategy;
	// hyperparams
	float gamma;

};

