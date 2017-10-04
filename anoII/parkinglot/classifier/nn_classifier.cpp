#include "nn_classifier.h"

#include "../utils.h"

std::unique_ptr<NNClassifier> NNClassifier::deserialize(const std::string& path)
{
	return std::make_unique<NNClassifier>(cv::ml::ANN_MLP::load<cv::ml::ANN_MLP>(path));
}

NNClassifier::NNClassifier()
{
	this->net = cv::ml::ANN_MLP::create();
}

NNClassifier::NNClassifier(cv::Ptr<cv::ml::ANN_MLP> net): net(net)
{

}

void NNClassifier::train(const std::vector<Example>& examples)
{
	if (!this->initialized)
	{
		this->initialize(examples[0].features.size());
	}

	cv::Mat trainingData(static_cast<int>(examples.size()), static_cast<int>(examples[0].features.size()), CV_32FC1);
	cv::Mat labels(static_cast<int>(examples.size()), 2, CV_32FC1);
	for (int i = 0; i < examples.size(); i++)
	{
		labels.at<float>(i, 0) = 1 - examples[i].classIndex;
		labels.at<float>(i, 1) = examples[i].classIndex;
		for (int j = 0; j < examples[i].features.size(); j++)
		{
			trainingData.at<float>(i, j) = examples[i].features[j];
		}
	}

	this->net->train(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, labels);
}

int NNClassifier::predict(const std::vector<float>& features)
{
	cv::Mat input(1, features.size(), CV_32FC1);
	for (int i = 0; i < features.size(); i++)
	{
		input.at<float>(0, i) = features[i];
	}

	return static_cast<int>(this->net->predict(input));
}

void NNClassifier::serialize(const std::string& path)
{
	this->net->save(path);
}

void NNClassifier::initialize(size_t inputSize)
{
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	cv::Mat layerSize(1, 3, CV_32SC1);
	layerSize.at<int>(0) = inputSize;
	layerSize.at<int>(1) = inputSize / 2;
	layerSize.at<int>(2) = 2;

	this->net->setLayerSizes(layerSize);
	this->net->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
	this->net->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	this->net->setBackpropMomentumScale(0.1f);
	this->net->setBackpropWeightScale(0.5f);
	this->net->setTermCriteria(criteria);

	this->initialized = true;
}
