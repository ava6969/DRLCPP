/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence
 *  Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  sharedLibraryInterfaceExample.cpp
 *
 *  Sample code for running an agent with the shared library interface.
 **************************************************************************** */

#include <iostream>
#include <ale_interface.hpp>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <SDL/SDL.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {


  ALEInterface ale;
  int dim[3] = {210, 160, 3 };

  // Get & Set the desired settings
  ale.setInt("random_seed", 123);
  //The default is already 0.25, this is just an example
  ale.setFloat("repeat_action_probability", 0.25);


  ale.setBool("display_screen", false);
  ale.setBool("sound", false);


  // Load the ROM file. (Also resets the system for new settings to
  // take effect.)
  auto id = "../atari_roms/breakout.bin";
  ale.loadROM(id);

  // Get the vector of legal actions
  ActionVect legal_actions = ale.getLegalActionSet();
    std::cout << legal_actions[1] << legal_actions.size() << std::endl;

    ale.reset_game();
    std::vector<unsigned char> state;
    ale.getScreenRGB(state);

    cv::Mat A(210, 160, CV_8UC3, state.data());

     // cout << "M = "<< endl << " "  << A << endl << endl;

    Mat gray;
    cvtColor(A, gray, COLOR_RGB2GRAY);
     cout << "M = "<< endl << " "  << gray.size() << endl << endl;
    Mat resized;
     resize(gray, resized, {84, 84}, INTER_AREA);
    cout << "M = "<< endl << " "  << resized.size() << endl << endl;
   resized.convertTo(resized,CV_32F);
   resized /= 255.0;
    cout << "M = "<< endl << " "  << resized<< endl << endl;

    // bACK TO OLD SHAPE
    // RESIZED_SCREEN.RESHAPE


  // Play 10 episodes
//  for (int episode = 0; episode < 10; episode++) {
//    float totalReward = 0;
//    while (!ale.game_over())
//    {
//      Action a = legal_actions[rand() % legal_actions.size()];
//      // Apply the action and get the resulting reward
//      float reward = ale.act(a);
//
//      totalReward += reward;
//    }
//    cout << "Episode " << episode << " ended with score: " << totalReward
//         << endl;
//    ale.reset_game();
//  }

  return 0;
}
