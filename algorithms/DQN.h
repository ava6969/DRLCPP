#pragma once
#include <memory>
#include "../utility/utils.h"
#include "../Experimental/Env.h"
#include <vector>
#include <torch/torch.h>
#include <tuple>
#include "../models/FCQ.h"
#include "../utility/Strategy.h"
#include "../buffers/ReplayBuffer.h"
#include <cstdint>
#include <Client.h>

using std::vector;
using std::unique_ptr;

using ResultVec = vector< vector<double> >;

class DQN
{
public:
	/*Constructor*/
	DQN(Model* onlineModel,
        Model* targetModel,
		ReplayBuffer* _buffer,
		Strategy* trainingStrategy,
		Strategy* evalStrategy,
		Device& _device,
		int _warmUpBatches,
		int _updateTargetEverySteps,
        float _maxGradientNorm=std::numeric_limits<float>::max(),
		bool ddqn=false,
		float tau=0,
		bool usePER=false
	):targetModel(targetModel),
		onlineModel(onlineModel),
		buffer(_buffer), 
		trainingStrategy(trainingStrategy), 
		evalStrategy(evalStrategy), 
		device(_device),nWarmupbatches(_warmUpBatches),
		updateTargetEverySteps(_updateTargetEverySteps),
        maxGradientNorm(_maxGradientNorm),
        DDQN(ddqn), tau(tau), usePER(usePER)
        {}

	void OptimizeModel(Tensor& states, Tensor& actions, Tensor& rewards,
                    Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim,int gamma=1.0);

    void OptimizeModel(Tensor& idx, Tensor& weights, Tensor& states, Tensor& actions, Tensor& rewards,
                            Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim, float gamma );

	std::tuple<torch::Tensor, bool> interaction_step(Tensor& state, Env* env);

	std::tuple<ResultVec, double, double, double> train(Env* mainEnv, Env* evalEnv, optim::RMSprop& optimizer,
                                                        int seed, float gamma, int saveFREQ,
		                                                int64_t max_minutes, int64_t max_episodes,
		                                                int64_t goal_mean_100_reward);

	void updateNetwork();

	std::tuple<double, double> evaluate(Env* evalEnv,  Model* EvalPolicyModel = nullptr, int64_t nEpisode = 1);

	void saveCheckpoint(int64_t episode=-1, Model* model=nullptr);


private:

	vector<Utils::ExperienceTuple> experiences;
    Model* targetModel{ nullptr };
    Model* onlineModel{ nullptr };
	bool exploratoryActionTaken = false;
	Utils::TrainingInfo trainingInfo{};
    float tau;
	Device device = torch::kCPU;

	Strategy* trainingStrategy;
	Strategy* evalStrategy;
	// hyperparams
	bool DDQN;
	float maxGradientNorm;
	int nWarmupbatches;
	int updateTargetEverySteps;
	ReplayBuffer* buffer;
	bool usePER;
};

