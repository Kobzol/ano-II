#pragma once

#include "model_classifier.h"

#include <memory>

class KnnClassifier : public ModelClassifier<cv::ml::KNearest>
{
public:
	KnnClassifier(std::string name);
};
