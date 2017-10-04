#include "svm_classifier.h"

SvmClassifier::SvmClassifier()
{
	cv::TermCriteria crit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5000, 0.01);

	this->model->setType(cv::ml::SVM::C_SVC);
	this->model->setKernel(cv::ml::SVM::LINEAR);
	this->model->setGamma(0.1);
	this->model->setC(2.0);
	this->model->setNu(0.1);
	this->model->setTermCriteria(crit);
}
