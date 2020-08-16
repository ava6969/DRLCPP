#include  <torch/torch.h>
#include "../Experimental/CartPoleEnv.h"
#include "../utility/utils.h"
#include <torch/script.h> // One-stop header.

using namespace std;

int selectAction(Tensor state, torch::jit::script::Module model)
{
	Tensor qVals;
	{
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(state);
		torch::NoGradGuard no_grad;
		qVals = model.forward(inputs).toTensor().cpu().detach().squeeze();

	}
	auto high = qVals.size(0);

	return qVals.argmax().item<int>();
}
void evaluate(Env* evalEnv, int64_t nEpisode, torch::jit::script::Module model)
{
	vector<double> res;
	vector<double> resmean;
	vector<double> resstd;
	std::unordered_map<string, bool> info;

	int timeStep = 0;
	for (int i = 0; i < nEpisode; i++)
	{
		auto state = evalEnv->reset();
		bool done = false;
		res.push_back(0);
		int reward = 0;
		int a = 0;

		while (!done )
		{

			a = selectAction(state, model);
			std::tie(state, reward, done, info) = evalEnv->step(a);
			res.back() += reward;
			timeStep += 1;

			if (info["TimeLimit.truncated"])
				break;
		}


		if (i % 100 == 0)
		{
			double mean100EvalScore = Utils::mean(res, 100);
			double std100EvalScore = Utils::std(res, 100);
			std::cout << "mean100: " << mean100EvalScore << " std100: " << std100EvalScore << "rew len: " << res.size() << " " << "time step" << timeStep / 100 << std::endl;
			int timeStep = 0;
		}
		
	}

}

int main()
{
	Device device = torch::cuda::is_available() ? kCUDA : kCPU;

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("traced_model.pt");
		module.to(device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr << e.what();
        return -1;
    }

    std::cout << "ok\n";

	Env* cartPole = new CartPoleEnv(device, 5, true);
	evaluate(cartPole, 1000, module);
	delete cartPole;

}
