#pragma once

#include "classifier.h"

#include <memory>

class SvmClassifier : public Classifier
{
	using Classifier::train;

public:
	static std::unique_ptr<SvmClassifier> deserialize(const std::string& path);

	SvmClassifier();
	explicit SvmClassifier(cv::Ptr<cv::ml::SVM> svm);

	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(const std::vector<float>& features) override;
	virtual void serialize(const std::string & path) override;

private:
	cv::Ptr<cv::ml::SVM> svm;
};
