//
// Created by dewe on 8/18/20.
//

#ifndef DRLCPP_MULTIPROCESSGYMENV_H
#define DRLCPP_MULTIPROCESSGYMENV_H
#include "Env.h"
#include <Client.h>
#include <thread>
#include "OpenAIGymWrapper.h"
#include <boost/make_shared.hpp>
#include <future>

class MultiprocessGymEnv : public Env
{
    boost::shared_ptr<Space> action_space;
    boost::shared_ptr<Space> observation_space;
    string id;
    int seed{};
    int nWorkers{};
    vector<std::shared_ptr<OpenAIGymWrapper>> envs;
public:

    MultiprocessGymEnv(string const& id, Device _device,  int seed = 0, bool render = false, int nWorkers=1):Env(_device,seed, render)
    {

        for(int i = 0; i <  nWorkers; i++)
        {
            std::shared_ptr<OpenAIGymWrapper> env = std::make_shared<OpenAIGymWrapper>(id,device,seed, render);
            if (i == 0)
            {
                action_space = env->ActionSpace();
                observation_space = env->ObservationSpace();
            }
            envs.emplace_back(std::move(env));
        }
        this->id = id;
        nS = observation_space->sample().size();
        nA = action_space->discreet_n;
    }

    Tensor reset(int rank=0)
    {
        if (rank != 0)
        {
            return envs[rank]->reset();
        }
        // fix nS for 2d state
        Tensor results = torch::empty({nWorkers, nS});
        std::vector<std::future<Tensor>> futures;
        // launch reset
        for (const auto& env: envs)
        {
            futures.emplace_back(std::async(std::launch::async, &OpenAIGymWrapper::reset, env));
        }

        // wait to complete
        for (int i = 0; i < nWorkers; i++)
        {
            results.index_put_({i},futures[i].get());
        }

        return results;

    }

    std::tuple<Tensor, vector<double>, vector<bool>, vector<string>> step(Tensor const& actions) override
    {

        // fix nS for 2d state

        Tensor states = torch::empty({nWorkers, nS});
        std::vector<double> rewards(nWorkers);
        std::vector<bool> dones(nWorkers);
        std::vector<string> infos(nWorkers);
        std::vector<std::future< std::tuple<Tensor, double, bool, string> >> futures;
        // launch reset
        futures.reserve(nWorkers);
        for (int i = 0; i < nWorkers; i++)
        {
            futures[i] = std::async(std::launch::async, &OpenAIGymWrapper::step, envs[i], actions.index({i}).item<float>());
        }

        // wait to complete
        for (int i = 0; i < nWorkers; i++)
        {
            auto res = futures[i].get();
            states.index_put_({i}, get<0>(res));
            rewards[i] = get<1>(res);
            dones[i] = get<2>(res);
            infos[i] = get<3>(res);
        }

        return {states, rewards, dones, infos};

    }

};

#endif //DRLCPP_MULTIPROCESSGYMENV_H
