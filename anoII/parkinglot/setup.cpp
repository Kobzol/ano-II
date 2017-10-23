#include "setup.h"

#include "utils.h"
#include "classifier/dlib_classifier.h"

std::vector<std::unique_ptr<Extractor>> createExtractors(const std::vector<float>& params)
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
ClassifierSet createClassifiers()
{
	ClassifierSet set;
	//set.add(std::make_unique<NNClassifier>("NN"));
	//set.add(std::make_unique<DirectNNClassifier>("Direct NN"));
	//set.add(std::make_unique<DNNClassifier>("DNN"));
	//set.add(std::make_unique<TinyDNNClassifier>("TinyDNN"));
	set.add(std::make_unique<DlibClassifier>("Dlib CNN"));
	set.add(std::make_unique<DlibClassifier>("Dlib CNN"));

	/*auto boost = std::make_unique<ModelClassifier<cv::ml::Boost>>(std::string("Boost"));
	set.add(std::move(boost));

	set.add(std::make_unique<KnnClassifier>("KNN"));
	set.add(std::make_unique<SvmClassifier>("SVM Linear", cv::ml::SVM::INTER));
	set.add(std::make_unique<SvmClassifier>("SVM RBF", cv::ml::SVM::RBF));*/
	return set;
}

void createExamples(
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
void loadPklot(const std::string& pathFile, const std::string& savePath, std::vector<cv::Mat>& images)
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
void createExamplesPklot(
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
std::vector<Evaluator> testClassifiers(
	const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	ClassifierSet& classifiers, const std::vector<int>& groundTruth, const std::string& testPath, bool visualTest)
{
	std::vector<std::string> testPaths = loadPathFile(testPath);
	std::vector<Evaluator> evaluators(classifiers.classifiers.size() + 1, Evaluator(groundTruth));

	int diff = 0;
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

		if (visualTest)
		{
			int currentDiff = evaluators.back().evaluate().getDifferences();
			if (currentDiff != diff)
			{
				diff = currentDiff;
				cv::Mat detected = markDetection(places, image, classifiers, responses);
				cv::imshow("Detection", detected);
				cv::waitKey(0);
			}
		}
	}

	return evaluators;
}
