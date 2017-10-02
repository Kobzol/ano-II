#pragma once

#include <string>
#include <vector>
#include "../example.h"

class Classifier
{
public:
	virtual ~Classifier() = default;

	virtual void train(const std::vector<Example>& examples) = 0;
	virtual int predict(const std::vector<float>& features);
	virtual int predict(cv::Mat image);

	virtual bool supportsFeatures();

	virtual void serialize(const std::string& path) = 0;

protected:
	void train(cv::ml::StatModel& model, const std::vector<Example>& examples);
};
