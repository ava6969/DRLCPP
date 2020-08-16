
#include "../algorithms/NFQ.h"
#include "../Experimental/CartPoleEnv.h"
#include "../algorithms/DQN.h"
#include <Client.h>

using namespace torch;
int main()
{
    Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // hyperParams
    int seed = 5;
    int nWarmUpBatches = 5;
    int updateEvery = 10;
    int gamma = 1.0;
    int maxMinutes = 20;
    int maxEpisode = 10000;


    //Env* CartPole = new CartPoleEnv(device, seed);
    /*    int64_t nS = CartPole->getStateSpace();
    int32_t nA = CartPole->getActionSpace();*/

    auto id = "CartPole-v1";
    boost::shared_ptr<Client> CartPole(new Client("127.0.0.1", 5000));
    CartPole->make(id);
    boost::shared_ptr<Space> action_space = CartPole->action_space();
    boost::shared_ptr<Space> observation_space = CartPole->observation_space();

    int64_t nS = CartPole->observation_space()->sample().size();
    int32_t nA = CartPole->action_space()->discreet_n;

    FCQ onlineModel(nS, nA, device);
    FCQ targetModel(nS, nA, device);

    float value_optimizer_lr = 0.0005;


    auto evalStrategy = new Strategy<FCQ>();
    auto trainStrategy = new EGreedyLinearStrategy<FCQ>(1.0, 0.3);

   auto buffer = new ReplayBuffer({ nS }, nA, 50000, 64, device, seed);
    // NFQ agent( model, trainStrategy, evalStrategy , device, epsilon, 1024, 1.00);
    //    float epsilon = 0.5;
    //    int epochs = 40;
    // auto trainStrategy = new EGreedyStrategy<FCQ>(0.5);

    auto optimizer(optim::RMSprop(onlineModel->parameters(), value_optimizer_lr));
    DQN agent(onlineModel, targetModel, buffer, trainStrategy, evalStrategy, device, nWarmUpBatches, updateEvery);
    agent.train(CartPole, optimizer, seed, gamma, maxMinutes, maxEpisode, 495);

    // delete CartPole;
    delete evalStrategy;
    delete trainStrategy;
    delete buffer;
    return 0;
}
