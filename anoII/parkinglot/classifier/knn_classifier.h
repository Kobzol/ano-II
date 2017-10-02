#pragma once

#include "classifier.h"

#include <memory>

class KnnClassifier : public Classifier
{
	using Classifier::train;

public:
	static std::unique_ptr<KnnClassifier> deserialize(const std::string& path);

	KnnClassifier();
	explicit KnnClassifier(cv::Ptr<cv::ml::KNearest> kNearest);

	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(const std::vector<float>& features) override;
	virtual void serialize(const std::string& path) override;

private:
	cv::Ptr<cv::ml::KNearest> kNearest;
};
