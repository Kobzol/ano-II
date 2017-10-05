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
#define CLASSIFIER_PATH "classifier"

static std::vector<std::unique_ptr<Extractor>> createExtractors()
{
	std::vector<std::unique_ptr<Extractor>> extractors;
	extractors.push_back(std::make_unique<SobelExtractor>(85));

	return extractors;
}
static std::vector<Example> createExamples(
	const std::vector<Place>& places,
	const std::vector<std::unique_ptr<Extractor>>& extractors,
	const std::string& positivePath,
	const std::string& negativePath)
{
	std::vector<std::string> positivePaths = loadPathFile(positivePath);
	std::vector<std::string> negativePaths = loadPathFile(negativePath);

	std::vector<Example> examples;
	appendExamples(positivePaths, places, extractors, 1, examples);
	appendExamples(negativePaths, places, extractors, 0, examples);

	return examples;
}
static ClassifierSet createClassifiers()
{
	ClassifierSet set;
	set.add(std::make_unique<NNClassifier>());
	set.add(std::make_unique<ModelClassifier<cv::ml::RTrees>>());
	set.add(std::make_unique<ModelClassifier<cv::ml::Boost>>());
	set.add(std::make_unique<KnnClassifier>());
	set.add(std::make_unique<SvmClassifier>());
	return set;
}
static void testClassifiers(const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	ClassifierSet& classifiers)
{
	std::vector<std::string> testPaths = loadPathFile("../test_images/test.txt");

	for (auto& testPath : testPaths)
	{
		cv::Mat image = cv::imread(testPath);
		std::vector<cv::Mat> placeFrames = extractParkingPlaces(places, image);
		std::vector<std::vector<int>> responses = classifiers.predictMultiple(extractors, placeFrames);
		cv::Mat detected = markDetection(places, image, classifiers, responses);
		cv::imshow("Detection", detected);
		cv::waitKey(0);
	}
}

void parkinglot()
{
	std::vector<Place> places = loadGeometry("strecha1_map.txt");
	auto extractors = createExtractors();
	auto classifiers = createClassifiers();

#ifdef TRAIN
	auto examples = createExamples(places, extractors,
		"../train_images/full/full.txt",
		"../train_images/free/free.txt"
	);

	classifiers.train(examples);
	classifiers.save(CLASSIFIER_PATH);
#else
	classifiers.load(CLASSIFIER_PATH);
#endif
	
	testClassifiers(places, extractors, classifiers);
}
