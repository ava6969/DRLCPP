#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <vector>
#include <cassert>

using namespace torch;
using std::vector;

namespace Utils
{
	template<typename T>
	void print(T& str)
	{
		std::cout << str << std::endl;
	}

	struct ExperienceTuple
	{
		torch::Tensor states;
		float actions;
		double rewards;
		torch::Tensor nextStates;
		bool terminals;
	};



	struct TrainingInfo
	{
		vector<double> episodeReward;
		vector<int> episodeTimestep;
		vector<double> evaluationScores;
		vector<double> episodeSeconds;
		vector<int> episodeExploration;

		vector<double>  explore_ratio(int top_k)
		{
			assert(episodeExploration.size() == episodeTimestep.size());
			vector<double> ratios;


			int diff = episodeExploration.size() - top_k;
			bool greater = diff <= 0;

			int start = greater ? 0 : episodeExploration.size() - top_k;
			if (episodeExploration.size() > 1)

				std::transform(
					begin(episodeExploration) + start,
					end(episodeExploration),
					begin(episodeTimestep) + start,
					back_inserter(ratios),
					[](int a, int b) { return (double)a / b; });

			return ratios;

		}
	};

//    struct MultiTrainingInfo
//    {
//        vector<vector<double>> episodeReward;
//        vector<vector<int>> episodeTimestep;
//        vector<vector<double>> evaluationScores;
//        vector<vector<double>> episodeSeconds;
//        vector<int> episodeExploration;
//
//        vector<double>  explore_ratio(int top_k)
//        {
//            assert(episodeExploration.size() == episodeTimestep.size());
//            vector<double> ratios;
//
//
//            int diff = episodeExploration.size() - top_k;
//            bool greater = diff <= 0;
//
//            int start = greater ? 0 : episodeExploration.size() - top_k;
//            if (episodeExploration.size() > 1)
//
//                std::transform(
//                        begin(episodeExploration) + start,
//                        end(episodeExploration),
//                        begin(episodeTimestep) + start,
//                        back_inserter(ratios),
//                        [](int a, int b) { return (double)a / b; });
//
//            return ratios;
//
//        }
//    };

	template<typename T>
	double mean(vector<T> vect, int top_k = 0)
	{
		if (vect.size() == 0)
			return 0;
		else if (top_k == 0)
			return (double)std::accumulate(vect.begin(), vect.end(), 0) / vect.size();


		int diff = vect.size() - top_k;
		bool greater = diff <= 0;

		int start = greater ? 0 : vect.size() - top_k;


		double sum = std::accumulate(vect.begin() + start, vect.end(), 0);

		return (double)sum / (vect.size() - start);
	}


	template<typename T>
	double std(vector<T> vect, int top_k = 0)
	{
		if (vect.size() == 0)
			return 0.0;

		double _mean = mean(vect, top_k);

		double diff = 0.0;
		if (top_k == 0)
		{
			for (auto& val : vect)
				diff += std::pow(val - _mean, 2);

			return std::sqrt(diff / (vect.size()));
		}


		int diff2 = vect.size() - top_k;
		bool greater = diff2 <= 0;
		int start = greater ? 0 : vect.size() - top_k;

		for (auto itr = begin(vect) + start; itr != end(vect); itr++)
			diff += std::pow(*itr - _mean, 2);
		return std::sqrt(diff / (vect.size() - start));

	}


}