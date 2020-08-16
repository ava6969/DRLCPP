#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Network.hpp>
#include <SFML/Audio.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <random>
#include <memory>
/*#include <SFML/Graphics/Vertex.hpp>*/
#include "torch/torch.h"
#include <unordered_map>
#include <math.h>
/*#include "Viewer.h"*/


using std::shared_ptr;

using std::vector;
using std::string;
using namespace torch;


class Env
{

protected:
	std::unordered_map<string, bool> info;
	/*Box2D state[4];*/
	Tensor state{};
	int action{};

	int stepsBeyondDone = -1;
	int maxEpisodeStep;
	int rewardThreshold;
	int stateSize{};
	int actionSize{};
	Device device = kCPU;
	/*shared_ptr<Viewer> viewer{};*/
	int stepCounter{};
public:


	Env(Device _device, int maxEpisodeStep, int rewardThreshold, int seed = 0, bool render = false):device(_device)
	{
	
		torch::manual_seed(seed);
		this->maxEpisodeStep = maxEpisodeStep;
		this->rewardThreshold = rewardThreshold;
	}

	virtual Tensor reset() = 0;

    virtual ~Env() = default;

	void close()
	{
        //viewer->close();
	}

	virtual std::tuple<Tensor, double, bool, std::unordered_map<string, bool> > step(double action) = 0;

	int getStateSpace() const { return stateSize; }
	int getActionSpace() const { return actionSize; }
	int MaxEpisodeStep() const { return maxEpisodeStep; }
	int RewardThreshold() const { return rewardThreshold; }
};

