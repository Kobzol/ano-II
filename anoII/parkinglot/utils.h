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
std::vector<int> loadGroundTruth(const std::string& path);

void appendExamples(const std::vector<std::string>& paths, const std::vector<Place>& places, const std::vector<std::unique_ptr<Extractor>>& extractors,
	int classIndex, std::vector<Example>& examples);
void appendDirectExamples(const std::vector<cv::Mat>& images, const std::vector<std::unique_ptr<Extractor>>& extractors, int classIndex, std::vector<Example>& examples);
std::vector<cv::Mat> extractParkingPlaces(const std::vector<Place>& places, cv::Mat image);

cv::Mat markDetection(const std::vector<Place>& places, cv::Mat image, ClassifierSet& set, std::vector<std::vector<int>> &responses);

void writeImages(const std::vector<cv::Mat>& images, const std::string& path);
std::vector<cv::Mat> readImages(const std::string& path);

bool writeMatBinary(std::ofstream& ofs, const cv::Mat& mat);
bool readMatBinary(std::ifstream& ifs, cv::Mat& mat);

bool fileExists(const std::string& name);
