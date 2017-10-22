#include "evaluator.h"

#include <fstream>
#include <algorithm>

Evaluator::Evaluator(std::vector<int> groundTruth): groundTruth(groundTruth)
{
	
}

EvaluationResult Evaluator::evaluate()
{
	EvaluationResult result;
	size_t size = std::min(this->results.size(), this->groundTruth.size());

	for (int i = 0; i < size; i++)
	{
		result.add(this->groundTruth[i], this->results[i]);
	}
	return result;
}

void Evaluator::addResult(int result)
{
	this->results.push_back(result);
}
void Evaluator::addResults(const std::vector<int>& results)
{
	this->results.insert(this->results.end(), results.begin(), results.end());
}
