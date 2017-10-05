#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "place.h"
#include "example.h"
#include "extractor/extractor.h"
#include "classifier/classifier_set.h"

std::vector<Place> loadGeometry(const std::string& filename);
std::vector<std::string> loadPathFile(const std::string& filename);

void appendExamples(const std::vector<std::string>& paths, const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	int classIndex, std::vector<Example>& examples);
std::vector<cv::Mat> extractParkingPlaces(const std::vector<Place>& places, cv::Mat image);

cv::Mat markDetection(const std::vector<Place>& places, cv::Mat image, ClassifierSet& set, std::vector<std::vector<int>> &responses);
