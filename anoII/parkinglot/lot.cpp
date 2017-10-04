#include <opencv2/opencv.hpp>

#include "example.h"
#include "utils.h"
#include "extractor/sobel_extractor.h"
#include "classifier/svm_classifier.h"
#include "classifier/knn_classifier.h"
#include "classifier/nn_classifier.h"
#include "classifier/model_classifier.h"
//#include "classifier/tinydnn_classifier.h"

#define TRAIN
#define CLASSIFIER_PATH ("classifier.xml")

using CLASSIFIER_CLASS = NNClassifier;

std::unique_ptr<Classifier> deserializeClassifier(const std::string& path)
{
	return CLASSIFIER_CLASS::deserialize(path);
}

void parkinglot()
{
	std::string geometry = "strecha1_map.txt";
	std::vector<Place> places = loadGeometry(geometry);
	std::vector<std::unique_ptr<Extractor>> extractors;
	extractors.push_back(std::move(std::make_unique<SobelExtractor>(85)));

#ifdef TRAIN
	std::vector<std::string> positivePaths = loadPathFile("../train_images/full/full.txt");
	std::vector<std::string> negativePaths = loadPathFile("../train_images/free/free.txt");

	std::vector<Example> examples;
	trainDataSet(positivePaths, places, extractors, 1, examples);
	trainDataSet(negativePaths, places, extractors, 0, examples);

	auto classifier = std::make_unique<CLASSIFIER_CLASS>();
	classifier->train(examples);
	classifier->serialize(CLASSIFIER_PATH);
#else
	auto classifier = deserializeClassifier(CLASSIFIER_PATH);
#endif
	std::vector<std::string> testPaths = loadPathFile("../test_images/test.txt");

	for (auto& testPath: testPaths)
	{
		cv::Mat image = cv::imread(testPath);
		std::vector<cv::Mat> placeFrames = extractParkingPlaces(places, image);
		std::vector<int> classes = predict(*classifier, extractors, placeFrames);
		cv::Mat detected = markDetection(places, classes, image);
		cv::imshow("Detection", detected);
		cv::waitKey(0);
	}
}
