#include "lbp_classifier.h"

static cv::Mat modifyImage(cv::Mat input)
{
	cv::Mat img = input.clone();
	cv::cvtColor(img, img, CV_BGR2GRAY);
	cv::resize(img, img, cv::Size(64, 64));

	return img;
}

LBPClassifier::LBPClassifier(const std::string& name): Classifier(name)
{
	this->model = cv::face::LBPHFaceRecognizer::create(1, 8, 8, 8);
}

void LBPClassifier::train(const std::vector<Example>& examples)
{
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	for (auto& example : examples)
	{
		images.push_back(modifyImage(example.image));
		labels.push_back(example.classIndex);
	}

	this->model->train(images, labels);
}

int LBPClassifier::predict(cv::Mat image)
{
	return this->model->predict(modifyImage(image));
}

bool LBPClassifier::supportsFeatures()
{
	return false;
}

void LBPClassifier::save(const std::string& path)
{
	this->model->save(path);
}

void LBPClassifier::load(const std::string& path)
{
	this->model->load<cv::face::LBPHFaceRecognizer>(path);
}
