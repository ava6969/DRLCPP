#pragma once
#pragma once
#include <random>
#include <memory>
#include "torch/torch.h"
#include <unordered_map>
#include <cmath>
#include "Env.h"
/*#include "CartPoleViewer.h"*/


using std::unique_ptr;
using std::vector;
using std::string;
using namespace torch;

const double PI = 3.141592653589793238463;
/*
	"""
	Description:
		A pole is attached by an un-actuated joint to a cart, which moves along
		a frictionless track. The pendulum starts upright, and the goal is to
		prevent it from falling over by increasing and reducing the cart's
		velocity.
	Source:
		This environment corresponds to the version of the cart-pole problem
		described by Barto, Sutton, and Anderson
	Observation:
		Type: Box(4)
		Num     Observation               Min                     Max
		0       Cart Position             -4.8                    4.8
		1       Cart Velocity             -Inf                    Inf
		2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
		3       Pole Angular Velocity     -Inf                    Inf
	Actions:
		Type: Discrete(2)
		Num   Action
		0     Push cart to the left
		1     Push cart to the right
		Note: The amount the velocity that is reduced or increased is not
		fixed; it depends on the angle the pole is pointing. This is because
		the center of gravity of the pole increases the amount of energy needed
		to move the cart underneath it
	Reward:
		Reward is 1 for every step taken, including the termination step
	Starting State:
		All observations are assigned a uniform random value in [-0.05..0.05]
	Episode Termination:
		Pole Angle is more than 12 degrees.
		Cart Position is more than 2.4 (center of the cart reaches the edge of
		the display).
		Episode length is greater than 200.
		Solved Requirements:
		Considered solved when the average return is greater than or equal to
		195.0 over 100 consecutive trials.
	"""
*/
//struct Box2D
//{
//	int number;
//	string obs;
//	float min;
//	float max;
//};

class CartPoleEnv : public Env
{
	//metadata = {
	//'render.modes': ['human', 'rgb_array'] ,
	//'video.frames_per_second' : 50
	//}
private:
	// GameRelated

    /*Box2D state[4];*/
    Tensor state{};
    int action{};
    string info = "";
    int stepsBeyondDone = -1;
    int maxEpisodeStep = 500;
    int rewardThreshold = 475.0;
    int stateSize{};
    int actionSize{};

    /*shared_ptr<Viewer> viewer{};*/
    int stepCounter{};
	float gravity = 9.8;
	float masscart = 1.0;
	float masspole = 0.1;
	float total_mass = masspole + masscart;
	float length = 0.5;  // actually half the pole's length
	float polemass_length = masspole * length;
	float force_mag = 10.0;
	float tau = 0.02;  // seconds between state updates
	string kinematics_integrator = "euler";
	// Angle at which to fail episode
	double thetaThresholdRadians = 12 * 2 * PI / 360;
	double xThreshold = 2.4;
//	std::thread worker;

	bool done{false};
	double reward{ 0.0 };



public:
	CartPoleEnv(Device _device, int seed = 0, bool render = false) :Env(_device,seed, render)
	{
		actionSize = 2;
		stateSize = 4;
/*		this->render = render;*/
/*		viewer = std::make_shared<CartPoleViewer>();
		if (render)
		{
			viewer->init_state(stateSize);
			worker = std::thread(&CartPoleViewer::run, viewer);
		}*/
			

	}

	std::tuple<Tensor, double, bool, string > step(float action) override

	{
		//TODO Assert action contains action space
		assert(action < actionSize);

		auto x = state[0].item<double>();
		double x_dot = state[1].item<double>();
		double theta = state[2].item<double>();
		double theta_dot = state[3].item<double>();
		double four_thirds = 4.0 / 3.0;
		
		float force = (action > 0) ? force_mag : -force_mag;

		double costheta = cos(theta);
		double sintheta = sin(theta);

		double temp = (force + polemass_length * theta_dot* theta_dot * sintheta) / total_mass;
		double thetaacc = (gravity * sintheta - costheta * temp) / (length * (four_thirds - masspole * costheta * costheta / total_mass));
		double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

		if (kinematics_integrator == "euler")
		{
			x += tau * x_dot;
			x_dot +=tau * xacc;
			theta += tau * theta_dot;
			theta_dot +=  tau * thetaacc;
		}
		else
		{
			x_dot += tau * xacc;
			x += tau * x_dot;
			theta_dot += tau * thetaacc;
			theta += tau * theta_dot;

		}

		
		state = torch::tensor({ x, x_dot, theta, theta_dot }).to(device);
/*		if (render)
			viewer->UpdateState(vector<double>{ x, x_dot, theta, theta_dot });*/

		done = x < -xThreshold || x > xThreshold || theta < -thetaThresholdRadians || theta > thetaThresholdRadians;

		++stepCounter;
		info = stepCounter == maxEpisodeStep ? "TimeLimit.truncated" : "";

		if (!done)
			reward = 1;

		else if (stepsBeyondDone == -1)
		{
			// Pole just fell!
			stepsBeyondDone = 0;
			reward = 1.0;
		}
		else
		{
			if (stepsBeyondDone == 0)
			{
				std::cout <<
					"You are calling 'step()' even though this "
					"environment has already returned done = True. You "
					"should always call 'reset()' once you receive 'done = "
					"True' -- any further steps are undefined behavior." << std::endl;
			}
			stepsBeyondDone += 1;
			reward = 0.0;
		}

		return {state, reward, done, info };
	}

	Tensor reset()
	{
		stepCounter = 0;
		stepsBeyondDone = -1;
		info = "";
		state = torch::rand({ stateSize }, device).uniform_(-0.05, 0.05);
		return state;
	}

/*	~CartPoleEnv()
	{
		if(render)
		worker.join();
	}*/

};

