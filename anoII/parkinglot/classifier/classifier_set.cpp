#include "classifier_set.h"

void ClassifierSet::train(const std::vector<Example>& examples)
{
	for (auto& classifier : this->classifiers)
	{
		classifier->train(examples);
	}
}

std::vector<int> ClassifierSet::predict(const Example& example, cv::Mat frame)
{
	std::vector<int> responses;
	for (auto& classifier : this->classifiers)
	{
		if (classifier->supportsFeatures())
		{
			responses.push_back(classifier->predict(example.features));
		}
		else responses.push_back(classifier->predict(frame));
	}

	return responses;
}
std::vector<std::vector<int>> ClassifierSet::predictMultiple(const std::vector<std::unique_ptr<Extractor>>& extractors, const std::vector<cv::Mat>& frames)
{
	std::vector<std::vector<int>> responses;
	for (auto& frame : frames)
	{
		Example example = Example::create(extractors, frame, -1);
		responses.push_back(this->predict(example, frame));
	}

	return responses;
}
int ClassifierSet::predictClass(std::vector<int>& response)
{
	int results[2] = { 0 };
	for (int r : response)
	{
		results[r]++;
	}
	return results[0] > results[1] ? 0 : 1;
}

void ClassifierSet::load(const std::string& name)
{
	for (int i = 0; i < this->classifiers.size(); i++)
	{
		std::string path = name + std::to_string(i) + ".xml";
		this->classifiers[i]->load(path);
	}
}
void ClassifierSet::save(const std::string& name)
{
	for (int i = 0; i < this->classifiers.size(); i++)
	{
		std::string path = name + std::to_string(i) + ".xml";
		this->classifiers[i]->save(path);
	}
}

void ClassifierSet::add(std::unique_ptr<Classifier> classifier)
{
	this->classifiers.push_back(std::move(classifier));
}
