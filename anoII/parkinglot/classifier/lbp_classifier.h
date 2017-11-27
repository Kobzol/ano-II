#pragma once

#include "classifier.h"

#include <opencv2/face.hpp>

class LBPClassifier : public Classifier
{
public:
	LBPClassifier(const std::string& name);

	virtual void train(const std::vector<Example>& examples) override;
	virtual float predict(cv::Mat image);

	virtual bool supportsFeatures();

	virtual void save(const std::string & path) override;
	virtual void load(const std::string & path) override;

private:
	cv::Ptr<cv::face::LBPHFaceRecognizer> model;
};
