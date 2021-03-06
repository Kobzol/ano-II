#include "svm_classifier.h"

SvmClassifier::SvmClassifier(std::string name, int kernelType): ModelClassifier<cv::ml::SVM>(name)
{
	cv::TermCriteria crit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5000, 0.001f);

	this->model->setType(cv::ml::SVM::C_SVC);
	this->model->setKernel(kernelType);
	this->model->setGamma(0.1);
	this->model->setC(2.0);
	this->model->setNu(0.1);
	this->model->setTermCriteria(crit);
}

float SvmClassifier::predict(const std::vector<float>& features)
{
	float dist = this->model->predict(features, cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
	return 1.0f / (1.0f + std::exp(dist));
}
