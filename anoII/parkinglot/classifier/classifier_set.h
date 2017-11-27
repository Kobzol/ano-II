#pragma once

#include <string>
#include <vector>
#include <memory>

#include "classifier.h"
#include "../extractor/extractor.h"

class ClassifierSet
{
public:
	void train(const std::vector<Example>& examples);
	std::vector<float> predict(const Example& example, cv::Mat frame);
	std::vector<std::vector<float>> predictMultiple(const std::vector<std::unique_ptr<Extractor>>& extractors, const std::vector<cv::Mat>& frames);
	int predictClass(const std::vector<float>& response);

	void load(const std::string& name);
	void save(const std::string& name);

	void add(std::unique_ptr<Classifier> classifier);

	std::vector<std::unique_ptr<Classifier>> classifiers;
};
