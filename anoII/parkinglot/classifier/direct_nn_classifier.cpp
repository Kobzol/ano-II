#include "direct_nn_classifier.h"

#include "../utils.h"

#define IMG_WIDTH 36
#define IMG_HEIGHT 36

static std::vector<float> imgToData(cv::Mat image)
{
	cv::cvtColor(image, image, CV_BGR2GRAY);

	cv::resize(image, image, cv::Size(IMG_WIDTH, IMG_HEIGHT));
	cv::medianBlur(image, image, 3);

	cv::Mat sobel = image.clone();
	cv::Canny(image, sobel, 30, 30 * 3);

	std::vector<float> data;
	for (int i = 0; i < sobel.rows; i++)
	{
		for (int j = 0; j < sobel.cols; j++)
		{
			data.push_back((float)(sobel.at<uchar>(i, j)) / 255.0f);
		}
	}

	return data;
}

DirectNNClassifier::DirectNNClassifier(std::string name) : ModelClassifier<cv::ml::ANN_MLP>(name)
{

}

void DirectNNClassifier::train(const std::vector<Example>& examples)
{
	if (!this->initialized)
	{
		this->initialize(IMG_WIDTH * IMG_HEIGHT);
	}

	cv::Mat trainingData(static_cast<int>(examples.size()), IMG_WIDTH * IMG_HEIGHT, CV_32FC1);
	cv::Mat labels(static_cast<int>(examples.size()), 2, CV_32FC1);
	for (int i = 0; i < examples.size(); i++)
	{
		labels.at<float>(i, 0) = static_cast<float>(1 - examples[i].classIndex);
		labels.at<float>(i, 1) = static_cast<float>(examples[i].classIndex);

		auto data = imgToData(examples[i].image);
		for (int j = 0; j < data.size(); j++)
		{
			trainingData.at<float>(i, j) = data[j];
		}
	}

	this->model->train(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, labels);
}

float DirectNNClassifier::predict(cv::Mat image)
{
	auto data = imgToData(image);
	return static_cast<float>(this->model->predict(data));
}

bool DirectNNClassifier::supportsFeatures()
{
	return false;
}

void DirectNNClassifier::initialize(int inputSize)
{
	CvTermCriteria criteria;
	criteria.max_iter = 2000;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	cv::Mat layerSize(1, 3, CV_32SC1);
	layerSize.at<int>(0) = IMG_WIDTH * IMG_HEIGHT;
	layerSize.at<int>(1) = 30;
	layerSize.at<int>(2) = 2;

	this->model->setLayerSizes(layerSize);
	this->model->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.2f, 0.2f);
	this->model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	this->model->setTermCriteria(criteria);

	this->initialized = true;
}
