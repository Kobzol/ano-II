#include "direct_nn_classifier.h"

#include <iterator>

#define IMG_SIZE 28

DirectNNClassifier::DirectNNClassifier(std::string name): NNClassifier(name)
{

}

bool DirectNNClassifier::supportsFeatures()
{
	return false;
}

void DirectNNClassifier::train(const std::vector<Example>& examples)
{
	if (!this->initialized)
	{
		this->initialize(static_cast<int>(examples[0].features.size()));
	}

	cv::Mat trainingData(static_cast<int>(examples.size()), static_cast<int>(IMG_SIZE * IMG_SIZE), CV_32FC1);
	cv::Mat labels(static_cast<int>(examples.size()), 2, CV_32FC1);
	for (int i = 0; i < examples.size(); i++)
	{
		labels.at<float>(i, 0) = static_cast<float>(1 - examples[i].classIndex);
		labels.at<float>(i, 1) = static_cast<float>(examples[i].classIndex);
	
		std::vector<float> features = this->getImageFeatures(examples[i].image);
		for (int j = 0; j < features.size(); j++)
		{
			trainingData.at<float>(i, j) = features[j];
		}
	}

	this->model->train(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, labels);
}

int DirectNNClassifier::predict(cv::Mat image)
{
	auto features = this->getImageFeatures(image);
	return NNClassifier::predict(features);
}

void DirectNNClassifier::initialize(int inputSize)
{
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.01f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	cv::Mat layerSize(1, 3, CV_32SC1);
	layerSize.at<int>(0) = IMG_SIZE * IMG_SIZE;
	layerSize.at<int>(1) = 30;
	layerSize.at<int>(2) = 2;

	this->model->setLayerSizes(layerSize);
	this->model->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
	this->model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	this->model->setBackpropMomentumScale(0.1f);
	this->model->setBackpropWeightScale(0.5f);
	this->model->setTermCriteria(criteria);

	this->initialized = true;
}

std::vector<float> DirectNNClassifier::getImageFeatures(cv::Mat image)
{
	cv::cvtColor(image, image, CV_BGR2GRAY);

	cv::Mat resized;
	cv::resize(image, resized, cv::Size(IMG_SIZE, IMG_SIZE));

	std::vector<float> features;
	for (int i = 0; i < resized.rows; i++)
	{
		for (int j = 0; j < resized.cols; j++)
		{
			features.push_back(((float) resized.at<uchar>(i, j)) / 255.0f);
		}
	}

	assert(features.size() == IMG_SIZE * IMG_SIZE);
	
	return features;
}
