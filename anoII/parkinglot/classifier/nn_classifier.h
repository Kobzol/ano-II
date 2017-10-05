#pragma once

#include "model_classifier.h"

#include <memory>

class NNClassifier : public ModelClassifier<cv::ml::ANN_MLP>
{
public:
	virtual void train(const std::vector<Example>& examples) override;

private:
	void initialize(int inputSize);

	bool initialized = false;
};
