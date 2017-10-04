#include "knn_classifier.h"

KnnClassifier::KnnClassifier()
{
	this->model->setIsClassifier(true);
	this->model->setDefaultK(1);
	this->model->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
}
