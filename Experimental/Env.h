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
    Device device = kCPU;
    bool render;

public:

    int64_t nS{};
    int32_t nA{};

	explicit Env(Device _device,  int seed = 0, bool render = false):device(_device),render(render)
	{
		torch::manual_seed(seed);

	}

	virtual Tensor reset() = 0;


    virtual ~Env() = default;

	virtual void close()=0;

	virtual std::tuple<Tensor, double, bool, string> step(float action)
    {
        return {};
    }
	virtual std::tuple<Tensor, vector<double>, vector<bool>, vector<string>> step(Tensor const& actions)
    {
        return {};
    }


};

