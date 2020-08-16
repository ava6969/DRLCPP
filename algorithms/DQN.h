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
	DQN(FCQ& onlineModel,
		FCQ& targetModel,
		ReplayBuffer* _buffer,
		Strategy<FCQ>* trainingStrategy,
		Strategy<FCQ>* evalStrategy,
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
                            Tensor& nextStates, Tensor& terminals, optim::RMSprop& optim, int gamma );

	std::tuple<torch::Tensor, bool> interaction_step(Tensor& state, Env* env);

	std::tuple<ResultVec, double, double, double> train(Env* env, optim::RMSprop& optimizer, int seed, int gamma,
		int64_t max_minutes, int64_t max_episodes,
		int64_t goal_mean_100_reward = -1);


    std::tuple<torch::Tensor, bool> interaction_step(Tensor& state, const boost::shared_ptr<Client>& env);

    std::tuple<ResultVec, double, double, double> train(const boost::shared_ptr<Client>& env, optim::RMSprop& optimizer,
                                                        int seed, int gamma, int saveFREQ,
                                                        int64_t max_minutes, int64_t max_episodes,
                                                        int64_t goal_mean_100_reward = -1);

	void updateNetwork();

	std::tuple<double, double> evaluate(Env* evalEnv,
                                     const FCQ& EvalPolicyModel = nullptr, int64_t nEpisode = 1);
    std::tuple<double, double> evaluate(const boost::shared_ptr<Client>& evalEnv,
                                        bool render=false,
                                        const FCQ& EvalPolicyModel= nullptr,
                                        int64_t nEpisode=1);
	void saveCheckpoint(int64_t episode=-1, const FCQ& Model=nullptr);


private:

	vector<Utils::ExperienceTuple<int>> experiences;
	FCQ targetModel{ nullptr };
	FCQ onlineModel{ nullptr };
	bool exploratoryActionTaken = false;
	Utils::TrainingInfo trainingInfo{};
    float tau;
	Device device = torch::kCPU;

	Strategy<FCQ>* trainingStrategy;
	Strategy<FCQ>* evalStrategy;
	// hyperparams
	bool DDQN;
	float maxGradientNorm;
	int nWarmupbatches;
	int updateTargetEverySteps;
	ReplayBuffer* buffer;
	bool usePER;
};

