#pragma once

#include "nn_classifier.h"

class DirectNNClassifier : public NNClassifier
{
public:
	DirectNNClassifier(std::string name);

	virtual bool supportsFeatures() override;
	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(cv::Mat image);

protected:
	virtual void initialize(int inputSize) override;

private:
	std::vector<float> getImageFeatures(cv::Mat image);
};
