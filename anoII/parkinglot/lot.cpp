#include <opencv2/opencv.hpp>

#include "example.h"
#include "utils.h"
#include "extractor/sobel_extractor.h"

#define TRAIN
#define SVM_PATH ("svm.xml")

void parkinglot()
{
	std::string geometry = "strecha1_map.txt";
	std::vector<Place> places = loadGeometry(geometry);
	std::vector<std::unique_ptr<Extractor>> extractors;
	extractors.push_back(std::move(std::make_unique<SobelExtractor>()));

#ifdef TRAIN
	std::vector<std::string> positivePaths = loadPathFile("../train_images/full/full.txt");
	std::vector<std::string> negativePaths = loadPathFile("../train_images/free/free.txt");

	std::vector<Example> examples;
	trainDataSet(positivePaths, places, extractors, 1, examples);
	trainDataSet(negativePaths, places, extractors, 0, examples);

	auto svm = trainSVM(examples);
	svm->save(SVM_PATH);
#else
	auto svm = cv::ml::SVM::load<cv::ml::SVM>(SVM_PATH);
#endif
	std::vector<std::string> testPaths = loadPathFile("../test_images/test.txt");

	for (auto& testPath: testPaths)
	{
		cv::Mat image = cv::imread(testPath);
		std::vector<cv::Mat> placeFrames = extractParkingPlaces(places, image);
		std::vector<int> classes = predictSVM(*svm, extractors, placeFrames);
		cv::Mat detected = markDetection(places, classes, image);
		cv::imshow("Detection", detected);
		cv::waitKey(0);
	}
}
