#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "utils.h"

using namespace Utils;


TEST_CASE("if topk > len(timestep/expoloration) , out length must equal  len(timestep/expoloration", "[topk var]")
{
	// Arrange
	TrainingInfo tInfo;
	tInfo.episodeExploration = { 10, 20, 15, 10, 4 , 11 };
	tInfo.episodeTimestep = { 20, 40, 30, 20 , 12, 33 };

	// Act
	auto res = tInfo.explore_ratio(10);

	// Assert
	REQUIRE(res.size() == tInfo.episodeExploration.size());

}

TEST_CASE("if topk < len(timestep/expoloration) , out length must equal  topk", "[topk var]")
{
	// Arrange
	TrainingInfo tInfo;
	tInfo.episodeExploration = { 10, 20, 15, 10, 4 , 11 };
	tInfo.episodeTimestep = { 20, 40, 30, 20 , 12, 33 };

	// Act
	auto res = tInfo.explore_ratio(4);

	// Assert
	REQUIRE(res.size() == 4);

}

TEST_CASE("Checking correct output of explore ratio with higher topk value", "[topk var]")
{
	// Arrange
	TrainingInfo tInfo;
	tInfo.episodeExploration = { 10, 20, 15, 10, 4 , 11 };
	tInfo.episodeTimestep = { 20, 40, 30, 20 , 12, 33 };

	// Act
	auto res = tInfo.explore_ratio(10);

	// Assert
	REQUIRE(res == std::vector<double>{0.5, 0.5, 0.5, 0.5, (double)4/12, (double)11/33});

}

TEST_CASE("Checking correct output of explore ratio with lower topk value", "[topk var]")
{
	// Arrange
	TrainingInfo tInfo;
	tInfo.episodeExploration = { 10, 20, 15, 10, 4 , 11 };
	tInfo.episodeTimestep = { 20, 40, 30, 20 , 12, 33 };

	// Act
	auto res = tInfo.explore_ratio(4);

	// Assert
	REQUIRE(res == std::vector<double>{(double)15/30, (double)10/20, (double)4 / 12, (double)11 / 33});
}


TEST_CASE("Correct output of mean when topk > len(rewards)", "[topk var]")
{
	// Arrange
	vector rewards = { 1, 1, 0, 1, 1 , 0 };

	// Act
	double res = mean(rewards, 10);

	// Assert
	REQUIRE(res == (double)4/6);

}

TEST_CASE("Correct output of mean when topk is default", "[topk 0]")
{
	// Arrange
	vector rewards = { 1, 1, 0, 1, 1 , 0 };

	// Act
	double res = mean(rewards);

	// Assert
	REQUIRE(res == (double)4 / 6);

}

TEST_CASE("Correct output of mean when topk < len(result)", "[topk var]")
{
	// Arrange
	vector rewards = { 1, 1, 0, 1, 1 , 0 };

	// Act
	double res = mean(rewards, 3);

	// Assert
	REQUIRE(res == (double)2 / 3);

}

TEST_CASE("Edge CASE Empy vector", "[topk var]")
{
	// Arrange
	vector <double>rewards;

	// Act
	double res = mean(rewards, 3);

	// Assert
	REQUIRE(res == 0);

}



TEST_CASE("Correct output of std when topk > len(rewards)", "[topk var]")
{
	// Arrange
	vector rewards = { 10, 12, 23, 23, 16, 23, 21, 16 };

	// Act
	double res = Utils::std(rewards, 10);

	// Assert
	REQUIRE((float)res == (float)4.8989794855664);

}

TEST_CASE("Correct output of std when topk is default", "[topk 0]")
{
	// Arrange
	vector rewards = { 10, 12, 23, 23, 16, 23, 21, 16 };

	// Act
	double res = Utils::std(rewards);

	// Assert
	REQUIRE((float)res == (float)4.8989794855664);

}

TEST_CASE("Correct output of std when topk < len(result)", "[topk var]")
{
	// Arrange
	vector rewards = { 10, 12, 23, 23, 16, 23, 21, 16 };

	// Act
	double res = Utils::std(rewards, 3);

	// Assert
	REQUIRE((float)res == (float)2.9439202887759
	);

}

TEST_CASE("Edge CASE Empty std vector", "[topk var]")
{
	// Arrange
	vector <double>rewards;

	// Act
	double res = Utils::std(rewards, 3);

	// Assert
	REQUIRE(res == 0);

}




