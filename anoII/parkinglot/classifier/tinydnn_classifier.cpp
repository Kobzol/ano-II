#include "tinydnn_classifier.h"

using namespace tiny_dnn;

#define IMG_WIDTH 36
#define IMG_HEIGHT 36

static void convert_image(cv::Mat image,
	int width, int height,
	std::vector<vec_t>& data)
{
	cv::cvtColor(image, image, CV_BGR2GRAY);
	cv::resize(image, image, cv::Size(width, height));
	cv::medianBlur(image, image, 3);
	cv::Canny(image, image, 30, 30 * 3);

	cv::Mat_<uint8_t> resized;
	cv::resize(image, resized, cv::Size(width, height));

	vec_t d;
	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
		[=](uint8_t c) { return c; });
	data.push_back(d);
}

std::unique_ptr<TinyDNNClassifier> TinyDNNClassifier::deserialize(const std::string& path)
{
	auto net = std::make_unique<network<sequential>>();
	net->load(path);
	return std::make_unique<TinyDNNClassifier>(std::move(net));
}

TinyDNNClassifier::TinyDNNClassifier(std::string name): Classifier(name)
{
	this->net = std::make_unique<network<sequential>>();

	auto netRef = this->net.get();
	/**netRef
		<< convolutional_layer(80, 80, 5, 1, 6, padding::same) << relu_layer()			// in:32x32x1, 5x5conv, 6fmaps
		<< average_pooling_layer(80, 80, 6, 2) << relu_layer()							// in:32x32x6, 2x2pooling
		<< convolutional_layer(40, 40, 5, 6, 16, padding::same) << relu_layer()			// in:16x16x6, 5x5conv, 16fmaps
		<< average_pooling_layer(40, 40, 16, 2) << relu_layer()							// in:16x16x16, 2x2pooling
		<< fully_connected_layer(20 * 20 * 16, 100) << tanh_layer()						// in:8x8x16, out:100
		<< fully_connected_layer(100, 2) << tanh_layer();								// in:100 out:2*/
	*netRef << fully_connected_layer(IMG_WIDTH * IMG_HEIGHT, 30) << tanh_layer()
		<< fully_connected_layer(30, 2) << tanh_layer()
		<< fully_connected_layer(2, 2) << sigmoid_layer();
}

TinyDNNClassifier::TinyDNNClassifier(std::unique_ptr<tiny_dnn::network<tiny_dnn::sequential>> net): net(std::move(net)), Classifier("TinyDNN")
{

}

void TinyDNNClassifier::train(const std::vector<Example>& examples)
{
	std::vector<vec_t> data;
	std::vector<label_t> classes;

	for (auto& example : examples)
	{
		convert_image(example.image, IMG_WIDTH, IMG_HEIGHT, data);
		classes.push_back(example.classIndex);
	}

	adam opt;
	this->net->train<cross_entropy>(opt, data, classes, 50, 25);
	auto loss = this->net->test(data, classes);
	loss.print_detail(std::cerr);
}

float TinyDNNClassifier::predict(cv::Mat image)
{
	std::vector<vec_t> data;
	convert_image(image, IMG_WIDTH, IMG_HEIGHT, data);

	return (float) this->net->predict_label(data[0]);
}

bool TinyDNNClassifier::supportsFeatures()
{
	return false;
}

void TinyDNNClassifier::save(const std::string& path)
{
	this->net->save(path);
}
void TinyDNNClassifier::load(const std::string& path)
{
	this->net->load(path);
}
