#pragma once

#include <string>
#include <vector>

#include "evaluation_result.h"

class Evaluator
{
public:
	Evaluator(std::vector<int> groundTruth);

	void addResult(int result);
	void addResults(const std::vector<int>& results);

	EvaluationResult evaluate();

private:
	std::vector<int> groundTruth;
	std::vector<int> results;
};
