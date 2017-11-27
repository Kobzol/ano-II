#include "dnn_classifier.h"

#include <iterator>

#include <opencv2/dnn/all_layers.hpp>

#define IMG_SIZE 28

DNNClassifier::DNNClassifier(std::string name): Classifier(name)
{
	cv::dnn::LayerParams params;
	params.set("kernel_h", 16);
	params.set("kernel_w", 16);
	params.set("num_output", 256);
	auto convLayer = this->net.addLayer("input", "Convolution", params);
}

bool DNNClassifier::supportsFeatures()
{
	return false;
}

void DNNClassifier::train(const std::vector<Example>& examples)
{
	
}

float DNNClassifier::predict(cv::Mat image)
{
	this->net.setInput(image);
	return this->net.forward("output").at<float>(0, 0);
}

void DNNClassifier::save(const std::string& path)
{
	
}

void DNNClassifier::load(const std::string& path)
{

}
