#include "svm_classifier.h"

std::unique_ptr<SvmClassifier> SvmClassifier::deserialize(const std::string& path)
{
	return std::make_unique<SvmClassifier>(cv::ml::SVM::load<cv::ml::SVM>(path));
}

SvmClassifier::SvmClassifier()
{
	cv::TermCriteria crit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5000, 0.01);

	this->svm = cv::ml::SVM::create();
	this->svm->setType(cv::ml::SVM::C_SVC);
	this->svm->setKernel(cv::ml::SVM::LINEAR);
	this->svm->setGamma(0.1);
	this->svm->setC(2.0);
	this->svm->setNu(0.1);
	this->svm->setTermCriteria(crit);
}

SvmClassifier::SvmClassifier(cv::Ptr<cv::ml::SVM> svm) : svm(svm)
{

}

void SvmClassifier::train(const std::vector<Example>& examples)
{
	this->train(*this->svm.get(), examples);
}

int SvmClassifier::predict(const std::vector<float>& features)
{
	return static_cast<int>(this->svm->predict(features));
}

void SvmClassifier::serialize(const std::string& path)
{
	this->svm->save(path);
}
