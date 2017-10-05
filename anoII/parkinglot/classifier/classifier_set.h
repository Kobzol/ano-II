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
	std::vector<int> predict(const Example& example, cv::Mat frame);
	std::vector<std::vector<int>> predictMultiple(const std::vector<std::unique_ptr<Extractor>>& extractors, const std::vector<cv::Mat>& frames);
	int predictClass(std::vector<int>& response);

	void load(const std::string& name);
	void save(const std::string& name);

	void add(std::unique_ptr<Classifier> classifier);

private:
	std::vector<std::unique_ptr<Classifier>> classifiers;
};
