//
// Created by dewe on 8/16/20.
//

#ifndef DRLCPP_ALEWRAPPER_H
#define DRLCPP_ALEWRAPPER_H

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "Env.h"
#include "../ale/src/ale_interface.hpp"

using namespace cv;
class ALEWrapper : public Env
{

    ActionVect actionSpace;
    std::vector<int64_t> observation_space;
    string id;
    bool audio{false};
    ALEInterface ale;
    int repeat;
    vector<Mat> frameBuffer;
    bool done;

public:

    ALEWrapper(string const& romPath, Device _device,  int seed = 0, bool render = false, bool audio = false):Env(_device,seed, render)
    {
        ale.setInt("random_seed", seed);
        ale.setBool("display_screen", render);
        ale.setBool("sound", audio);
        ale.loadROM(romPath);
        actionSpace = ale.getLegalActionSet();
        observation_space = {4, 84, 84};
        nA = actionSpace.size();
        nS = 3;
        repeat = 4;
        frameBuffer.resize(2);
    }

    std::tuple<Tensor, double, bool, string> step(float action) override
    {
        auto[s, reward, done, info] = repeatAction(action);
        auto s_t = preprocess_data(s);
        auto stacked = stackFrame(s_t);
        return {stacked, reward, done, info};

    }

    torch::Tensor get_tensor_observation(const Mat& state) {
        Tensor out = torch::empty({84, 84}, TensorOptions().device(device).dtype(torch::kF32));
        for(int i=0; i < state.rows; i++)
        {
            for(int j=0; j < state.cols; j++)
            {
                auto val = *state.ptr<float>(i, j);
                out.index_put_({i, j},val);
            }
        }
        return out;
    }

    Tensor reset() override
    {
        ale.reset_game();
        std::vector<unsigned char> state;
        ale.getScreenRGB(state);
        cv::Mat A(210, 160, CV_8UC3, state.data());
        done = false;
        auto s_t = preprocess_data(A);

        auto stacked = stackFrame(s_t);
        return stacked;
    }


    void close() override
    {
        std::cout << id  << " closed" << std::endl;
    }

    std::tuple<std::vector<unsigned char> , double, bool, string> pre_step(float action)
    {
        std::vector<unsigned char> s;
        ale.getScreenRGB(s);
        int a = (int)action;
        auto legalAction = actionSpace[a];
        float reward = ale.act(legalAction);
        done = ale.game_over();
        return {s, reward, done, ""};
    }

     Tensor preprocess_data(const Mat& state)
    {
        Mat gray;
        cvtColor(state, gray, COLOR_RGB2GRAY);

        Mat resized;
        resize(gray, resized, {84, 84}, INTER_AREA);

        resized.convertTo(resized,CV_32F);
        resized /= 255.0;
        return get_tensor_observation(resized);
    }

    std::tuple<Mat, double, bool, string> repeatAction(float action)
    {
        float t_reward = 0.0;
        string info;
        for (int i =0; i < repeat; i++)
        {
            auto res = pre_step(action);
            auto obs = get<0>(res);
            auto r = get<1>(res);
            done = get<2>(res);
            info = get<3>(res);
            t_reward += r;
            int idx = i % 2;
            cv::Mat A(210, 160, CV_8UC3, obs.data());
            frameBuffer[idx] = A;
            if (done)
                break;
        }

        Mat maxframe;
        cv::max(frameBuffer[0], frameBuffer[1], maxframe);
        return {maxframe, t_reward, done, info};
    }

    Tensor stackFrame(Tensor const& frame)
    {
        int i  = 0;
        auto stacked = torch::zeros({repeat, 84, 84}, TensorOptions().device(device).dtype(kF32));
        while ( i < repeat)
        {
            stacked.index_put_({i}, frame);
            i++;
        }
        return stacked;
    }
};




#endif //DRLCPP_ALEWRAPPER_H
