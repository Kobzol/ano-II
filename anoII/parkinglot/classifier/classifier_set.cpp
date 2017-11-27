#include "classifier_set.h"

void ClassifierSet::train(const std::vector<Example>& examples)
{
	for (auto& classifier : this->classifiers)
	{
		classifier->train(examples);
	}
}

std::vector<float> ClassifierSet::predict(const Example& example, cv::Mat frame)
{
	std::vector<float> responses;
	for (auto& classifier : this->classifiers)
	{
		float response = 0.0f;
		if (classifier->supportsFeatures())
		{
			response = classifier->predict(example.features);
		}
		else response = classifier->predict(frame);

		responses.push_back(response);
	}

	return responses;
}
std::vector<std::vector<float>> ClassifierSet::predictMultiple(const std::vector<std::unique_ptr<Extractor>>& extractors, const std::vector<cv::Mat>& frames)
{
	std::vector<std::vector<float>> responses;
	for (auto& frame : frames)
	{
		Example example = Example::create(extractors, frame, -1);
		responses.push_back(this->predict(example, frame));
	}

	return responses;
}
int ClassifierSet::predictClass(const std::vector<float>& response)
{
	float sum = 0.0f;
	for (float p : response)
	{
		sum += p;
	}

	return sum >= ((float) response.size() * 0.5f) ? 1 : 0;
}

void ClassifierSet::load(const std::string& name)
{
	for (int i = 0; i < this->classifiers.size(); i++)
	{
		std::string path = name + std::to_string(i) + ".dat";
		this->classifiers[i]->load(path);
	}
}
void ClassifierSet::save(const std::string& name)
{
	for (int i = 0; i < this->classifiers.size(); i++)
	{
		std::string path = name + std::to_string(i) + ".dat";
		this->classifiers[i]->save(path);
	}
}

void ClassifierSet::add(std::unique_ptr<Classifier> classifier)
{
	this->classifiers.push_back(std::move(classifier));
}
