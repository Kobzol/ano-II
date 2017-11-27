#pragma once

#include "model_classifier.h"

#include <memory>
#include <cmath>

class SvmClassifier : public ModelClassifier<cv::ml::SVM>
{
public:
	SvmClassifier(std::string name, int kernelType = cv::ml::SVM::KernelTypes::LINEAR);

	virtual float predict(const std::vector<float>& features) override;
};
