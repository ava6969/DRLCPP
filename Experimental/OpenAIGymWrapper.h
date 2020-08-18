//
// Created by dewe on 8/16/20.
//

#ifndef DRLCPP_OPENAIGYMWRAPPER_H
#define DRLCPP_OPENAIGYMWRAPPER_H

#include "Env.h"
#include <Client.h>
#include <boost/make_shared.hpp>

class OpenAIGymWrapper : public Env
{
    boost::shared_ptr<Client> env;
    boost::shared_ptr<Space> action_space;
    boost::shared_ptr<Space> observation_space;
    string id;

public:

    OpenAIGymWrapper(string const& id, Device _device,  int seed = 0, bool render = false):Env(_device,seed, render)
    {
        env = boost::make_shared<Client>("127.0.0.1", 5000);
        env->make(id);
        action_space = env->action_space();
        this->id = id;
        observation_space = env->observation_space();
        nS = observation_space->sample().size();
        nA = action_space->discreet_n;
    }

    Tensor reset() override
    {
        State s;
        env->reset(&s);
        return torch::tensor(s.observation, device);
    }


    void close() override
    {
        std::cout << id  << " closed" << std::endl;
    }

    std::tuple<Tensor, double, bool, string> step(float action) override
    {
        State s;
        env->step(vector<float>{action}, render, &s);
        bool isTerminal = s.done;
        float reward = s.reward;
        Tensor newState = torch::tensor(s.observation, c10::device(device));
        return {newState, reward, isTerminal, s.info};
    }


};



#endif //DRLCPP_OPENAIGYMWRAPPER_H
