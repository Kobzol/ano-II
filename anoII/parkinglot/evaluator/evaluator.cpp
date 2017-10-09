#include "evaluator.h"

#include <fstream>

Evaluator::Evaluator(std::vector<int> groundTruth): groundTruth(groundTruth)
{
	
}

EvaluationResult Evaluator::evaluate()
{
	if (this->groundTruth.size() != this->results.size()) throw "Wrong result count";

	EvaluationResult result;
	for (int i = 0; i < this->groundTruth.size(); i++)
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
