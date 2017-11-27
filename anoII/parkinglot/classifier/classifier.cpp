#include "classifier.h"

Classifier::Classifier(std::string name): name(name)
{

}

float Classifier::predict(const std::vector<float>& features)
{
	return 0.0f;
}

float Classifier::predict(cv::Mat image)
{
	return 0.0f;
}

bool Classifier::supportsFeatures()
{
	return true;
}

void Classifier::train(cv::ml::StatModel& model, const std::vector<Example>& examples)
{
	cv::Mat trainingData(static_cast<int>(examples.size()), static_cast<int>(examples[0].features.size()), CV_32FC1);
	std::vector<int> labels;
	for (int i = 0; i < examples.size(); i++)
	{
		labels.push_back(examples[i].classIndex);
		for (int j = 0; j < examples[i].features.size(); j++)
		{
			trainingData.at<float>(i, j) = examples[i].features[j];
		}
	}

	model.train(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, labels);
}

std::string Classifier::getName() const
{
	return this->name;
}
