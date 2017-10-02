#include "utils.h"

#include <fstream>

std::vector<Place> loadGeometry(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open()) throw "geometry not found";

	std::vector<Place> places;

	int polygonCount;
	file >> polygonCount;
    for (int i = 0; i < polygonCount; i++)
	{
        int vertexCount;
		file >> vertexCount;
		file.ignore(2);	// skip ->

		Place place;
		for (int v = 0; v < vertexCount * 2; v++)
		{
			file >> place.coords[v];
			file.ignore(1);
		}
		places.push_back(place);
    }

	return places;
}
std::vector<std::string> loadPathFile(const std::string& filename)
{
	std::vector<std::string> paths;

	std::ifstream file(filename);
	if (!file.is_open()) throw "path file not found";

	std::string line;
	while (std::getline(file, line))
	{
		paths.push_back(line);
	}

	return paths;
}

void trainDataSet(const std::vector<std::string>& paths, const std::vector<Place>& places,
	const std::vector<std::unique_ptr<Extractor>>& extractors, int classIndex,
	std::vector<Example>& examples)
{
#pragma omp parallel for
	for (int i = 0; i < paths.size(); i++)
	{
		auto& path = paths[i];
		cv::Mat trainingImage = cv::imread(path, 1);
		std::vector<cv::Mat> frames = extractParkingPlaces(places, trainingImage);

		std::vector<Example> localExamples;
		for (auto& frame: frames)
		{
			localExamples.push_back(generateExample(extractors, frame, classIndex));
		}

#pragma omp critical
		{
			examples.insert(examples.end(), localExamples.begin(), localExamples.end());
		}

		std::cerr << "Trained from " << path << std::endl;
	}
}
std::vector<cv::Mat> extractParkingPlaces(const std::vector<Place>& places, cv::Mat image)
{
	std::vector<cv::Mat> placeFrames;
	for (int i = 0; i < places.size(); i++)
	{
		auto& place = places[i];

		cv::Mat srcMat(4, 2, CV_32F);
		cv::Mat outMat(158, 172, CV_8U, 3);
		srcMat.at<float>(0, 0) = (float) place.x01;
		srcMat.at<float>(0, 1) = (float) place.y01;
		srcMat.at<float>(1, 0) = (float) place.x02;
		srcMat.at<float>(1, 1) = (float) place.y02;
		srcMat.at<float>(2, 0) = (float) place.x03;
		srcMat.at<float>(2, 1) = (float) place.y03;
		srcMat.at<float>(3, 0) = (float) place.x04;
		srcMat.at<float>(3, 1) = (float) place.y04;

		cv::Mat destMat(4, 2, CV_32F);
		destMat.at<float>(0, 0) = 0.0f;
		destMat.at<float>(0, 1) = 0.0f;
		destMat.at<float>(1, 0) = (float) outMat.cols;
		destMat.at<float>(1, 1) = 0.0f;
		destMat.at<float>(2, 0) = (float) outMat.cols;
		destMat.at<float>(2, 1) = (float) outMat.rows;
		destMat.at<float>(3, 0) = 0.0f;
		destMat.at<float>(3, 1) = (float) outMat.rows;

		cv::Mat H = cv::findHomography(srcMat, destMat, 0);
		cv::warpPerspective(image, outMat, H, cv::Size(158, 172));

		placeFrames.push_back(outMat);
	}

	return placeFrames;
}
Example generateExample(const std::vector<std::unique_ptr<Extractor>>& extractors, cv::Mat place, int classIndex)
{
	Example example(classIndex, place);
	for (auto& extractor : extractors)
	{
		auto features = extractor->extract(place);
		example.features.insert(example.features.end(), features.begin(), features.end());
	}

	return example;
}

std::vector<int> predict(Classifier& classifier, const std::vector<std::unique_ptr<Extractor>>& extractors, std::vector<cv::Mat>& frames)
{
	std::vector<int> classes;
	for (auto& frame: frames)
	{
		int value = 0;
		if (classifier.supportsFeatures())
		{
			value = classifier.predict(generateExample(extractors, frame, -1).features);
		}
		else value = classifier.predict(frame);

		classes.push_back(value);
	}

	return classes;
}

/*void convert_image(cv::Mat image,
	int w,
	int h,
	std::vector<tiny_dnn::vec_t>& data)
{
	cv::cvtColor(image, image, CV_BGR2GRAY);

	cv::Mat_<uint8_t> resized;
	cv::resize(image, resized, cv::Size(w, h));

	tiny_dnn::vec_t d;
	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
		[=](uint8_t c) { return c; });
	data.push_back(d);
}
std::unique_ptr<tiny_dnn::network<tiny_dnn::sequential>> trainDNN(std::vector<Example>& examples)
{
	using namespace tiny_dnn;

	auto net = std::make_unique<network<sequential>>();
	network<sequential>& netRef = *net;

	netRef << convolutional_layer(32, 32, 5, 1, 6, padding::same) << tanh_layer()  // in:32x32x1, 5x5conv, 6fmaps
		<< max_pooling_layer(32, 32, 6, 2) << tanh_layer()                // in:32x32x6, 2x2pooling
		<< convolutional_layer(16, 16, 5, 6, 16, padding::same) << tanh_layer() // in:16x16x6, 5x5conv, 16fmaps
		<< max_pooling_layer(16, 16, 16, 2) << tanh_layer()               // in:16x16x16, 2x2pooling
		<< fully_connected_layer(8 * 8 * 16, 100) << tanh_layer()                       // in:8x8x16, out:100
		<< fully_connected_layer(100, 10) << softmax_layer();                       // in:100 out:10

	std::vector<vec_t> data;
	std::vector<size_t> classes;

	for (auto& example : examples)
	{
		convert_image(example.image, 32, 32, data);
		classes.push_back(example.classIndex);
	}

	adagrad opt;
	net->train<cross_entropy, adagrad>(opt, data, classes, 20, 5);
	return net;
}
std::vector<int> predictDNN(tiny_dnn::network<tiny_dnn::sequential>& net, const std::vector<cv::Mat>& images)
{
	using namespace tiny_dnn;
	std::vector<vec_t> data;
	for (auto& image : images)
	{
		convert_image(image, 32, 32, data);
	}

	std::vector<int> classes;
	for (auto& item : data)
	{
		classes.push_back(net.predict_max_value(item));
	}
	return classes;
}*/

cv::Mat markDetection(const std::vector<Place>& places, const std::vector<int>& classes, cv::Mat image)
{
	cv::Mat detected = image.clone();

	for (int i = 0; i < places.size(); i++)
	{
		int classIndex = classes[i];
		auto& place = places[i];

		cv::Scalar color(0.0f, classIndex == 0 ? 255.0f : 0.0f, classIndex == 1 ? 255.0f : 0.0f);
		for (int i = 0; i < 4; i++)
		{
			int next = ((i + 1) * 2) % 8;
			cv::line(detected,
				cv::Point(place.coords[i * 2], place.coords[i * 2 + 1]),
				cv::Point(place.coords[next], place.coords[next + 1]),
				color
			);
		}
	}

	return detected;
}
