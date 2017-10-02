#pragma once

#include "classifier.h"

#include <memory>

class NNClassifier : public Classifier
{
	using Classifier::train;

public:
	static std::unique_ptr<NNClassifier> deserialize(const std::string& path);

	NNClassifier();
	explicit NNClassifier(cv::Ptr<cv::ml::ANN_MLP> net);

	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(const std::vector<float>& features) override;
	virtual void serialize(const std::string& path) override;

private:
	cv::Ptr<cv::ml::ANN_MLP> net;
};
