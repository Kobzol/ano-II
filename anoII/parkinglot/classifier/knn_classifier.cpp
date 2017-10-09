#include "knn_classifier.h"

KnnClassifier::KnnClassifier(std::string name): ModelClassifier(name)
{
	this->model->setIsClassifier(true);
	this->model->setDefaultK(1);
	this->model->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
}
