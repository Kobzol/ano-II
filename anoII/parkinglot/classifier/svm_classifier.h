#pragma once

#include "model_classifier.h"

#include <memory>

class SvmClassifier : public ModelClassifier<cv::ml::SVM>
{
public:
	SvmClassifier(std::string name, int kernelType = cv::ml::SVM::KernelTypes::LINEAR);
};
