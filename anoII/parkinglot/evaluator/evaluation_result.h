#pragma once

#include <ostream>

struct EvaluationResult
{
public:
	EvaluationResult();

	void add(int groundTruth, int prediction);

	float getF1Score() const;
	float getAccuracy() const;
	int getDifferences() const;

private:
	union
	{
		struct
		{
			// ground, detection
			int trueNegative;		// 00
			int falsePositive;		// 01
			int falseNegative;		// 10
			int truePositive;		// 11
		};
		int data[4];
	};


	int differences = 0;
};

std::ostream& operator<<(std::ostream& os, const EvaluationResult& result);
