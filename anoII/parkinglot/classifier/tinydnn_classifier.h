#pragma once

#include "classifier.h"
#include <tiny_dnn/tiny_dnn.h>

#include <memory>

class TinyDNNClassifier : public Classifier
{
public:
	static std::unique_ptr<TinyDNNClassifier> deserialize(const std::string& path);

	TinyDNNClassifier();
	explicit TinyDNNClassifier(std::unique_ptr<tiny_dnn::network<tiny_dnn::sequential>> net);

	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(cv::Mat image) override;

	virtual bool supportsFeatures() override;

	virtual void serialize(const std::string & path) override;


private:
	std::unique_ptr<tiny_dnn::network<tiny_dnn::sequential>> net;
};