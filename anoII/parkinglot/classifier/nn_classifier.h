#pragma once

#include "model_classifier.h"

#include <memory>

class NNClassifier : public ModelClassifier<cv::ml::ANN_MLP>
{
public:
	NNClassifier(std::string name);

	virtual void train(const std::vector<Example>& examples) override;

protected:
	virtual void initialize(int inputSize);

	bool initialized = false;
};
