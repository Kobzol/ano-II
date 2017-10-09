#include "evaluation_result.h"

#include <cstring>
#include <cmath>

EvaluationResult::EvaluationResult()
{
	std::memset(this->data, 0, sizeof(int) * 4);
}

void EvaluationResult::add(int groundTruth, int prediction)
{
	this->data[(groundTruth << 1) | prediction]++;
	this->differences += std::abs(groundTruth - prediction);
}

float EvaluationResult::getF1Score() const
{
	const float tp = 2.0f * this->truePositive;

	return tp / (tp + this->falsePositive + this->falseNegative);
}

float EvaluationResult::getAccuracy() const
{
	return ((float)(this->truePositive + this->trueNegative)) / (this->truePositive + this->trueNegative + this->falsePositive + this->falseNegative);
}

int EvaluationResult::getDifferences() const
{
	return this->differences;
}

std::ostream& operator<<(std::ostream& os, const EvaluationResult& result)
{
	os << "F1: " << result.getF1Score() << ", Acc: " << result.getAccuracy() << ", Wrong: " << result.getDifferences();
	return os;
}
