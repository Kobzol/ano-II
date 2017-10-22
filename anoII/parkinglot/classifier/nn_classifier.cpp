#include "nn_classifier.h"

#include "../utils.h"

NNClassifier::NNClassifier(std::string name): ModelClassifier<cv::ml::ANN_MLP>(name)
{

}

void NNClassifier::train(const std::vector<Example>& examples)
{
	if (!this->initialized)
	{
		this->initialize(static_cast<int>(examples[0].features.size()));
	}

	cv::Mat trainingData(static_cast<int>(examples.size()), static_cast<int>(examples[0].features.size()), CV_32FC1);
	cv::Mat labels(static_cast<int>(examples.size()), 2, CV_32FC1);
	for (int i = 0; i < examples.size(); i++)
	{
		labels.at<float>(i, 0) = static_cast<float>(1 - examples[i].classIndex);
		labels.at<float>(i, 1) = static_cast<float>(examples[i].classIndex);
		for (int j = 0; j < examples[i].features.size(); j++)
		{
			trainingData.at<float>(i, j) = examples[i].features[j];
		}
	}

	this->model->train(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, labels);
}

void NNClassifier::initialize(int inputSize)
{
	CvTermCriteria criteria;
	criteria.max_iter = 1000;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	cv::Mat layerSize(1, 4, CV_32SC1);
	layerSize.at<int>(0) = inputSize;
	layerSize.at<int>(1) = inputSize / 2;
	layerSize.at<int>(2) = inputSize / 4;
	layerSize.at<int>(3) = 2;

	this->model->setLayerSizes(layerSize);
	this->model->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
	this->model->setActivationFunction(cv::ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
	this->model->setBackpropMomentumScale(0.1f);
	this->model->setBackpropWeightScale(0.1f);
	this->model->setTermCriteria(criteria);

	this->initialized = true;
}
