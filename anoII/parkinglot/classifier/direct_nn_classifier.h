#pragma once

#include "model_classifier.h"

#include <memory>

class DirectNNClassifier : public ModelClassifier<cv::ml::ANN_MLP>
{
public:
	DirectNNClassifier(std::string name);

	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(cv::Mat image) override;
	virtual bool supportsFeatures() override;

protected:
	virtual void initialize(int inputSize);

	bool initialized = false;
};
