#pragma once
#include <torch/torch.h>

using namespace torch;

template<class T>
struct Strategy
{
    virtual int  selectAction(T model, torch::Tensor state)
    {
        Tensor qVals;
        {
            torch::NoGradGuard no_grad;
            qVals = model->forward(state).cpu().detach().squeeze();
        }

        return  qVals.argmax().item<int>();
    }

    bool exploratoryActionTaken = false;

};

template<class T>
struct EGreedyStrategy: public Strategy<T>
{

    EGreedyStrategy(double _epsilon) :epsilon(_epsilon) {}

    int selectAction(T model, torch::Tensor state)
    {
        this->exploratoryActionTaken = false;

        Tensor qVals;
        {
            torch::NoGradGuard no_grad;
            qVals = model->forward(state).cpu().detach().squeeze();
        }

        auto high = qVals.size(0);
        int action;

        if (torch::rand({ 1 }).item<float>() > epsilon)
            action = qVals.argmax().item<int>();
        else
            action = torch::randint(high, { 1 }).item<int>();

        this->exploratoryActionTaken = (action != (qVals.argmax()).item<int>());
        return action;
    }

private:
    double epsilon = 0.1;

};

template<class T>
struct EGreedyLinearStrategy : public Strategy<T>
{
private:
    float eps;
    float init_eps;
    float decay_steps;
    float min_eps;
    Tensor epsilons;
    int64_t t = 0;

public:
    EGreedyLinearStrategy(float init_epsilon = 1.0, float min_epsilon = 0.1, float _decay_steps = 20000)
        :eps(init_epsilon), init_eps(init_epsilon), decay_steps(_decay_steps), min_eps(min_epsilon)
    {}

     double epsilon_update()
     {
         eps = 1 - t / decay_steps;
         eps = (init_eps - min_eps) * eps + min_eps;
         eps = std::clamp(this->eps, min_eps, init_eps);
         t += 1;
         return eps;
    
    }

    int selectAction(T model, torch::Tensor state) override
    {
        this->exploratoryActionTaken = false;
        Tensor qVals;
        {
            torch::NoGradGuard no_grad;
            qVals = model->forward(state).cpu().detach().squeeze();
        }

        auto high = qVals.size(0);
        int action;

        if (torch::rand({ 1 }).item<float>() > eps)
            action = qVals.argmax().item<int>();
        else
            action = torch::randint(high, { 1 }).item<int>();
        eps = epsilon_update();
        this->exploratoryActionTaken = (action != (qVals.argmax()).item<int>());
        return action;
    }

};

template<class T>
struct EGreedyExpStrategy : public Strategy<T>
{
    EGreedyExpStrategy(float init_epsilon = 1.0, float min_epsilon = 0.1, float _decay_steps = 20000)
        :eps(init_epsilon), init_eps(init_epsilon), decay_steps(_decay_steps), min_eps(min_epsilon)
    {
        epsilons = (0.01 / torch::logspace(-2, 0, decay_steps) - 0.01) * ((init_epsilon - min_epsilon) + min_epsilon);
    }

    double epsilon_update()
    {
        eps = 1 - t / decay_steps;
        eps = (init_eps - min_eps) * eps + min_eps;
        eps = std::clamp(eps, min_eps, init_eps);
        t += 1;
        return eps;
    }

    int selectAction(T model, torch::Tensor state) override
    {
        this->exploratoryActionTaken = false;
        Tensor qVals;
        {
            torch::NoGradGuard no_grad;
            qVals = model->forward(state).cpu().detach().squeeze();
        }

        auto high = qVals.size(0);
        int action;

        if (torch::rand({ 1 }).item<float>() > eps)
            action = qVals.argmax().item<int>();
        else
            action = torch::randint(high, { 1 }).item<int>();
        eps = epsilon_update();
        this->exploratoryActionTaken = (action != (qVals.argmax()).item<int>());
        return action;
    }

private:
    float eps;
    float init_eps;
    float decay_steps;
    float min_eps;
    Tensor epsilons;
    int64_t t = 0;

};

template<class T>
struct SoftMaxStrategy : public Strategy<T>
{

    SoftMaxStrategy(double _init_temp = 1.0, double _min_temp = 0.1, float exploration_ratio=0.8, float _max_steps = 250000)
        : init_temp(_init_temp), min_temp(_min_temp), max_steps(_max_steps)
    {
        this->exploratoryActionTaken = exploration_ratio;
    }

    double update_temp()
    {
        double temp = 1 - t / (max_steps * exploration_ratio);
        temp = (init_temp - min_temp) * temp + min_temp;
        temp = std::clamp(temp, min_temp, init_temp);
        t += 1;
        return temp;

    }

    int selectAction(T model, torch::Tensor state)
    {
        this->exploratoryActionTaken = false;
        double temp = update_temp();
        Tensor qVals;
        {
            torch::NoGradGuard no_grad;
            qVals = model->forward(state).cpu().detach().squeeze();
        }
        auto scaledQs = qVals / temp;
        auto normQs = scaledQs - scaledQs.max();
        auto e = torch::exp(normQs);
        auto probs = e / e.sum();
        /*assert(probs.sum().isclose(1).item<bool>());*/
    
        auto high = qVals.size(0);

        int action = torch::multinomial(probs, 1).item<int>();
        this->exploratoryActionTaken = (action != (qVals.argmax()).item<int>());
        return action;
    }

private:
    double max_steps;
    double min_temp;
    double init_temp;
    float exploration_ratio;
    int64_t t = 0;

};