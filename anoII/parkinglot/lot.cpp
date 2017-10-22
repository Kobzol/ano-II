#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "example.h"
#include "utils.h"
#include "evaluator/evaluator.h"
#include "extractor/sobel_extractor.h"
#include "extractor/canny_extractor.h"
#include "extractor/hog_extractor.h"
#include "extractor/surf_extractor.h"
#include "extractor/kaze_extractor.h"
#include "extractor/haar_extractor.h"
#include "extractor/histogram_extractor.h"
#include "classifier/svm_classifier.h"
#include "classifier/knn_classifier.h"
#include "classifier/nn_classifier.h"
#include "classifier/model_classifier.h"
#include "classifier/dnn_classifier.h"
#include "classifier/direct_nn_classifier.h"
#include "classifier/dlib_classifier.h"
//#include "classifier/tinydnn_classifier.h"

// Sobel(30) - 0.974, 22
// Sobel(30) + SURF(400) - 0.975, 21 (SVM Linear)
// Sobel(30) + Histogram - 0.975, 21
// DNN(36, 0.2, 0.2) + Canny(30) - 0.977, 19
// Dlib - 10

//#define TRAIN
#define CLASSIFIER_PATH "classifier"
#define PKLOT_04_BIN_0 "pklot04_0.bin"
#define PKLOT_04_BIN_1 "pklot04_1.bin"
#define PKLOT_05_BIN_0 "pklot05_0.bin"
#define PKLOT_05_BIN_1 "pklot05_1.bin"
#define PKLOT_PUC_BIN_0 "pklotPUC_0.bin"
#define PKLOT_PUC_BIN_1 "pklotPUC_1.bin"
#define SHUFFLE_EXAMPLES
//#define VISUAL_TEST

static std::vector<std::unique_ptr<Extractor>> createExtractors(const std::vector<float>& params = {})
{
	std::vector<std::unique_ptr<Extractor>> extractors;
	//extractors.push_back(std::make_unique<SobelExtractor>(29));
	//extractors.push_back(std::make_unique<CannyExtractor>(30));
	//extractors.push_back(std::make_unique<HOGExtractor>());
	//extractors.push_back(std::make_unique<SurfExtractor>());
	//extractors.push_back(std::make_unique<KazeExtractor>());
	//extractors.push_back(std::make_unique<HistogramExtractor>());

	/*auto classifier = std::make_unique<cv::CascadeClassifier>();
	classifier->load("cars.xml");
	extractors.push_back(std::make_unique<HaarExtractor>(std::move(classifier)));*/
	return extractors;
}
static ClassifierSet createClassifiers()
{
	ClassifierSet set;
	//set.add(std::make_unique<NNClassifier>("NN"));
	//set.add(std::make_unique<DirectNNClassifier>("Direct NN"));
	//set.add(std::make_unique<DNNClassifier>("DNN"));
	//set.add(std::make_unique<TinyDNNClassifier>("TinyDNN"));
	set.add(std::make_unique<DlibClassifier>("Dlib CNN"));

	/*auto boost = std::make_unique<ModelClassifier<cv::ml::Boost>>(std::string("Boost"));
	set.add(std::move(boost));

	set.add(std::make_unique<KnnClassifier>("KNN"));
	set.add(std::make_unique<SvmClassifier>("SVM Linear", cv::ml::SVM::INTER));
	set.add(std::make_unique<SvmClassifier>("SVM RBF", cv::ml::SVM::RBF));*/
	return set;
}

static void createExamples(
	const std::vector<Place>& places,
	const std::vector<std::unique_ptr<Extractor>>& extractors,
	const std::string& positivePath,
	const std::string& negativePath,
	std::vector<Example>& examples)
{
	std::vector<std::string> positivePaths = loadPathFile(positivePath);
	std::vector<std::string> negativePaths = loadPathFile(negativePath);

	appendExamples(positivePaths, places, extractors, 1, examples);
	appendExamples(negativePaths, places, extractors, 0, examples);
}
static void loadPklot(const std::string& pathFile, const std::string& savePath, std::vector<cv::Mat>& images)
{
	if (!fileExists(savePath))
	{
		std::vector<std::string> paths = loadPathFile(pathFile);
		int count = 0;
		for (auto& path : paths)
		{
			images.push_back(cv::imread(path, cv::IMREAD_COLOR));
			count++;
			if (count % 10000 == 0)
			{
				std::cerr << count << std::endl;
			}
		}
		writeImages(images, savePath);
	}
	else images = readImages(savePath);
}
static void createExamplesPklot(
	const std::vector<std::unique_ptr<Extractor>>& extractors,
	const std::string& positivePath,
	const std::string& negativePath,
	const std::string& positiveSave,
	const std::string& negativeSave,
	std::vector<Example>& examples)
{
	std::vector<cv::Mat> imagesPositive, imagesNegative;
	loadPklot(positivePath, positiveSave, imagesPositive);
	loadPklot(negativePath, negativeSave, imagesNegative);

	appendDirectExamples(imagesPositive, extractors, 1, examples);
	appendDirectExamples(imagesNegative, extractors, 0, examples);
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
			evaluators.back().addResult(classifiers.predictClass(response));
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
	std::vector<int> truth = loadGroundTruth("groundtruth_fix.txt");

	auto extractors = createExtractors();
	auto classifiers = createClassifiers();

#ifdef TRAIN
	std::vector<Example> examples;
	createExamples(places, extractors,
		"../train_images/full/full.txt",
		"../train_images/free/free.txt",
		examples
	);
	createExamplesPklot(extractors,
		"../UFPR05/full.txt",
		"../UFPR05/empty.txt",
		PKLOT_05_BIN_1,
		PKLOT_05_BIN_0,
		examples
	);
	createExamplesPklot(extractors,
		"../UFPR04/full.txt",
		"../UFPR04/empty.txt",
		PKLOT_04_BIN_1,
		PKLOT_04_BIN_0,
		examples
		);
	/*createExamplesPklot(extractors,
		"../PUC/full.txt",
		"../PUC/empty.txt",
		PKLOT_PUC_BIN_1,
		PKLOT_PUC_BIN_0,
		examples
	);*/

	std::cerr << "Loaded " << examples.size() << " images" << std::endl;

#ifdef SHUFFLE_EXAMPLES
	std::random_shuffle(examples.begin(), examples.end());
#endif

	classifiers.train(examples);
	classifiers.save(CLASSIFIER_PATH);
#else
	classifiers.load(CLASSIFIER_PATH);
#endif
	auto evaluations = testClassifiers(places, extractors, classifiers, truth);

	for (int i = 0; i < evaluations.size() - 1; i++)
	{
		std::cout << std::setw(16) << std::left << classifiers.classifiers[i]->getName() + ":" << evaluations[i].evaluate() << std::endl;
	}
	std::cout << std::setw(16) << std::left << "Combined:" << evaluations[evaluations.size() - 1].evaluate() << std::endl;
	std::cout << std::endl;

	getchar();
}
