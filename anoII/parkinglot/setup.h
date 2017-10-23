#pragma once

#include "example.h"
#include "place.h"
#include "evaluator/evaluator.h"
#include "classifier/classifier_set.h"

std::vector<std::unique_ptr<Extractor>> createExtractors(const std::vector<float>& params = {});
ClassifierSet createClassifiers();

void createExamples(
	const std::vector<Place>& places,
	const std::vector<std::unique_ptr<Extractor>>& extractors,
	const std::string& positivePath,
	const std::string& negativePath,
	std::vector<Example>& examples);
void loadPklot(const std::string& pathFile, const std::string& savePath, std::vector<cv::Mat>& images);
void createExamplesPklot(
	const std::vector<std::unique_ptr<Extractor>>& extractors,
	const std::string& positivePath,
	const std::string& negativePath,
	const std::string& positiveSave,
	const std::string& negativeSave,
	std::vector<Example>& examples);
std::vector<Evaluator> testClassifiers(
	const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	ClassifierSet& classifiers, const std::vector<int>& groundTruth, const std::string& testPath, bool visualTest);
