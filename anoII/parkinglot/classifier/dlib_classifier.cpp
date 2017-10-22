#include "dlib_classifier.h"

#include <dlib/opencv.h>
#include <dlib/dnn/trainer.h>

static int classIndexToDlib(int index)
{
	return index == 0 ? -1 : 1;
}
static int dlibToClassIndex(float result)
{
	return result < 0.0f ? 0 : 1;
}

static cv::Mat transformImage(const cv::Mat& img)
{
	cv::Mat image = img.clone();
	//cv::cvtColor(image, image, CV_BGR2GRAY);
	//cv::Canny(image, image, 30, 30 * 3);
	cv::resize(image, image, cv::Size(20, 28));

	return image;

	/*cv::cvtColor(image, image, CV_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	return channels[0];*/
}

DlibClassifier::DlibClassifier(std::string name): Classifier(name)
{

}

void DlibClassifier::train(const std::vector<Example>& examples)
{
	const std::string backup = "dnn_backup";

	dlib::dnn_trainer<decltype(this->learner), dlib::adam> trainer(this->learner, dlib::adam(0.001, 0.9, 0.999));
	trainer.set_learning_rate(0.01f);
	trainer.set_min_learning_rate(0.0001f);
	trainer.set_mini_batch_size(128);
	trainer.set_max_num_epochs(500);
	trainer.be_verbose();

	std::vector<float> labels;
	std::vector<dlib::matrix<pixel_type>> images;
	std::vector<cv::Mat> cvImages(examples.size());

	int i = 0;
	for (auto& example : examples)
	{
		labels.push_back(classIndexToDlib(example.classIndex));

		cvImages[i] = transformImage(example.image);

		dlib::cv_image<pixel_type> cvTest(cvImages[i]);
		dlib::matrix<pixel_type> mat = dlib::mat(cvTest);
		images.push_back(mat);

		i++;
	}

	trainer.set_synchronization_file(backup, std::chrono::seconds(20));

	try
	{
		trainer.train(images, labels);
		this->learner.clean();
		this->predictor = this->learner;

		std::remove(backup.c_str());
		std::remove((backup + "_").c_str());
	}
	catch (const std::exception& exc)
	{
		std::cerr << exc.what() << std::endl;
	}

	std::cerr << "Training finished" << std::endl;
}

int DlibClassifier::predict(cv::Mat image)
{
	cv::Mat transformed = transformImage(image);
	dlib::cv_image<pixel_type> cvImg(transformed);
	dlib::matrix<pixel_type> mat = dlib::mat(cvImg);

	return dlibToClassIndex(this->predictor(mat));
}

bool DlibClassifier::supportsFeatures()
{
	return false;
}

void DlibClassifier::save(const std::string& path)
{
	this->learner.clean();
	dlib::serialize(path) << this->learner;
}

void DlibClassifier::load(const std::string& path)
{
	dlib::deserialize(path) >> this->predictor;
}
