#pragma once

#include <string>
#include <vector>
#include "../example.h"

class Classifier
{
public:
	Classifier(std::string name);
	virtual ~Classifier() = default;

	virtual void train(const std::vector<Example>& examples) = 0;
	virtual float predict(const std::vector<float>& features);
	virtual float predict(cv::Mat image);

	virtual bool supportsFeatures();

	virtual void save(const std::string& path) = 0;
	virtual void load(const std::string& path) = 0;

	virtual std::string getName() const;

protected:
	void train(cv::ml::StatModel& model, const std::vector<Example>& examples);

private:
	std::string name;
};
