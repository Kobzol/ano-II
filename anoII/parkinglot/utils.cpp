#include "utils.h"

#include <fstream>
#include <opencv2/opencv.hpp>

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

void trainDataSet(const std::vector<std::string>& paths, const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	int classIndex, std::vector<Example>& examples)
{
	for (auto& path : paths)
	{
		cv::Mat trainingImage = cv::imread(path, 1);
		std::vector<cv::Mat> frames = extractParkingPlaces(places, trainingImage);
		for (auto& frame: frames)
		{
			examples.push_back(generateExample(extractors, frame, classIndex));
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
		srcMat.at<float>(0, 1) = (float)place.y01;
		srcMat.at<float>(1, 0) = (float)place.x02;
		srcMat.at<float>(1, 1) = (float)place.y02;
		srcMat.at<float>(2, 0) = (float)place.x03;
		srcMat.at<float>(2, 1) = (float)place.y03;
		srcMat.at<float>(3, 0) = (float)place.x04;
		srcMat.at<float>(3, 1) = (float)place.y04;

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
	Example example(classIndex);
	for (auto& extractor : extractors)
	{
		auto features = extractor->extract(place);
		example.features.insert(example.features.end(), features.begin(), features.end());
	}

	return example;
}
cv::Ptr<cv::ml::SVM> trainSVM(std::vector<Example>& examples)
{
	cv::TermCriteria crit;
	crit.epsilon = CV_TERMCRIT_EPS;
	crit.maxCount = 2000;
	crit.type = 1;

	auto svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::RBF);
	svm->setGamma(0.1);
	svm->setC(2.0);
	svm->setNu(0.1);
	svm->setTermCriteria(crit);

	cv::Mat trainingData(static_cast<int>(examples.size()), static_cast<int>(examples[0].features.size()), CV_32F);
	std::vector<int> labels;
	for (int i = 0; i < examples.size(); i++)
	{
		labels.push_back(examples[i].classIndex);
		for (int j = 0; j < examples[i].features.size(); j++)
		{
			trainingData.at<float>(i, j) = examples[i].features[j];
		}
	}

	svm->train(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, labels);
	return svm;
}

std::vector<int> predictSVM(cv::ml::SVM& svm, const std::vector<std::unique_ptr<Extractor>>& extractors, std::vector<cv::Mat>& frames)
{
	std::vector<int> classes;

	for (auto& frame: frames)
	{
		Example example = generateExample(extractors, frame, -1);

		float value = svm.predict(example.features);
		classes.push_back(static_cast<int>(value));
	}

	return classes;
}
cv::Mat markDetection(const std::vector<Place>& places, const std::vector<int>& classes, cv::Mat image)
{
	cv::Mat detected = image.clone();

	for (int i = 0; i < places.size(); i++)
	{
		int classIndex = classes[i];
		auto& place = places[i];

		cv::Scalar color(0.0f, classIndex == 0 ? 255.0f : 0.0f, classIndex == 1 ? 255.0f : 0.0f);
		std::vector<cv::Point> points;
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
