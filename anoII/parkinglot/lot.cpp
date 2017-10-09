#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>

#include "example.h"
#include "utils.h"
#include "evaluator/evaluator.h"
#include "extractor/sobel_extractor.h"
#include "extractor/canny_extractor.h"
#include "extractor/hog_extractor.h"
#include "extractor/hsv_extractor.h"
#include "classifier/svm_classifier.h"
#include "classifier/knn_classifier.h"
#include "classifier/nn_classifier.h"
#include "classifier/model_classifier.h"
#include "classifier/direct_nn_classifier.h"
//#include "classifier/tinydnn_classifier.h"

// Sobel(30) - 0.974, 22

#define TRAIN
#define CLASSIFIER_PATH "classifier"
//#define VISUAL_TEST

static std::vector<std::unique_ptr<Extractor>> createExtractors()
{
	std::vector<std::unique_ptr<Extractor>> extractors;
	extractors.push_back(std::make_unique<SobelExtractor>(30));
	//extractors.push_back(std::make_unique<CannyExtractor>(30));
	//extractors.push_back(std::make_unique<HOGExtractor>());
	//extractors.push_back(std::make_unique<HSVExtractor>());

	return extractors;
}
static ClassifierSet createClassifiers()
{
	ClassifierSet set;
	//set.add(std::make_unique<NNClassifier>("NN"));
	//set.add(std::make_unique<DirectNNClassifier>("Direct NN"));
	//set.add(std::make_unique<TinyDNNClassifier>("TinyDNN"));

	auto boost = std::make_unique<ModelClassifier<cv::ml::Boost>>(std::string("Boost"));
	set.add(std::move(boost));

	set.add(std::make_unique<KnnClassifier>("KNN"));
	set.add(std::make_unique<SvmClassifier>("SVM Linear", cv::ml::SVM::INTER));
	return set;
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
static std::vector<Evaluator> testClassifiers(
	const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	ClassifierSet& classifiers, const std::vector<int>& groundTruth)
{
	std::vector<std::string> testPaths = loadPathFile("../test_images/test.txt");

	std::vector<Evaluator> evaluators(classifiers.classifiers.size() + 1, Evaluator(groundTruth));

	for (auto& testPath : testPaths)
	{
		cv::Mat image = cv::imread(testPath);
		std::vector<cv::Mat> placeFrames = extractParkingPlaces(places, image);
		std::vector<std::vector<int>> responses = classifiers.predictMultiple(extractors, placeFrames);

		for (auto& response : responses)
		{
			for (int c = 0; c < response.size(); c++)
			{
				evaluators[c].addResult(response[c]);
			}
			evaluators[evaluators.size() - 1].addResult(classifiers.predictClass(response));
		}

#ifdef VISUAL_TEST
		cv::Mat detected = markDetection(places, image, classifiers, responses);
		cv::imshow("Detection", detected);
		cv::waitKey(0);
#endif
	}

	return evaluators;
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
	
	std::vector<int> truth = loadGroundTruth("groundtruth.txt");
	auto evaluations = testClassifiers(places, extractors, classifiers, truth);

	std::cout << std::endl;
	for (int i = 0; i < evaluations.size() - 1; i++)
	{
		std::cout << std::setw(16) << std::left << classifiers.classifiers[i]->getName() + ":" << evaluations[i].evaluate() << std::endl;
	}
	std::cout << std::setw(16) << std::left << "Combined:" << evaluations[evaluations.size() - 1].evaluate() << std::endl;

	getchar();
}
