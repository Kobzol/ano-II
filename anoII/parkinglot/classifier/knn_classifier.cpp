#include "knn_classifier.h"

std::unique_ptr<KnnClassifier> KnnClassifier::deserialize(const std::string& path)
{
	return std::make_unique<KnnClassifier>(cv::ml::KNearest::load<cv::ml::KNearest>(path));
}

KnnClassifier::KnnClassifier()
{
	this->kNearest = cv::ml::KNearest::create();
	this->kNearest->setIsClassifier(true);
	this->kNearest->setDefaultK(1);
	this->kNearest->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
}

KnnClassifier::KnnClassifier(cv::Ptr<cv::ml::KNearest> kNearest) : kNearest(kNearest)
{

}

void KnnClassifier::train(const std::vector<Example>& examples)
{
	this->train(*this->kNearest.get(), examples);
}

int KnnClassifier::predict(const std::vector<float>& features)
{
	return static_cast<int>(this->kNearest->predict(features));
}

void KnnClassifier::serialize(const std::string& path)
{
	this->kNearest->save(path);
}
