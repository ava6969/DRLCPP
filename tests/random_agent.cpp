
#include <Client.h>

void run_single_environment(
	const boost::shared_ptr<Client>& client,
	const std::string& env_id,
	int episodes_to_run)
{
	client->make(env_id);
	boost::shared_ptr<Space> action_space = client->action_space();
	boost::shared_ptr<Space> observation_space = client->observation_space();

	for (int e = 0; e < episodes_to_run; ++e) {
		printf("%s episode %i...\n", env_id.c_str(), e);
		State s;
        client->reset(&s);
		float total_reward = 0;
		int total_steps = 0;
		while (1) {
			std::vector<float> action = action_space->sample();
            client->step(action, true, &s);
			assert(s.observation.size() == observation_space->sample().size());
			total_reward += s.reward;
			total_steps += 1;
			if (s.done) break;
		}
		printf("%s episode %i finished in %i steps with reward %0.2f\n",
			env_id.c_str(), e, total_steps, total_reward);
	}
}

int main(int argc, char** argv)
{
	try {
		boost::shared_ptr<Client> client(new Client("127.0.0.1", 5000));
		run_single_environment(client, "CartPole-v0", 3);

	}
	catch (const std::exception& e) {
		fprintf(stderr, "ERROR: %s\n", e.what());
		return 1;
	}

	return 0;
}
