#pragma once

#include "nn_classifier.h"

#include <opencv2/dnn.hpp>

class DNNClassifier : public Classifier
{
public:
	DNNClassifier(std::string name);

	virtual bool supportsFeatures() override;
	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(cv::Mat image);

	virtual void save(const std::string& path) override;
	virtual void load(const std::string& path) override;

private:
	cv::dnn::Net net;
};
