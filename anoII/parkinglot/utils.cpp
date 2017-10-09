#include "utils.h"

#include <fstream>

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

std::vector<int> loadGroundTruth(const std::string& path)
{
	std::ifstream file(path);
	if (!file.is_open()) throw "Ground truth file not found";

	std::vector<int> truth;

	std::string line;
	while (file >> line)
	{
		truth.push_back(std::strtol(line.c_str(), nullptr, 10));
	}

	return truth;
}

void appendExamples(const std::vector<std::string>& paths, const std::vector<Place>& places,
	const std::vector<std::unique_ptr<Extractor>>& extractors, int classIndex,
	std::vector<Example>& examples)
{
#pragma omp parallel for
	for (int i = 0; i < paths.size(); i++)
	{
		auto& path = paths[i];
		cv::Mat trainingImage = cv::imread(path, 1);
		std::vector<cv::Mat> frames = extractParkingPlaces(places, trainingImage);

		std::vector<Example> localExamples;
		for (auto& frame: frames)
		{
			localExamples.push_back(Example::create(extractors, frame, classIndex));
		}

#pragma omp critical
		{
			examples.insert(examples.end(), localExamples.begin(), localExamples.end());
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
		srcMat.at<float>(0, 1) = (float) place.y01;
		srcMat.at<float>(1, 0) = (float) place.x02;
		srcMat.at<float>(1, 1) = (float) place.y02;
		srcMat.at<float>(2, 0) = (float) place.x03;
		srcMat.at<float>(2, 1) = (float) place.y03;
		srcMat.at<float>(3, 0) = (float) place.x04;
		srcMat.at<float>(3, 1) = (float) place.y04;

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

static std::string serializeResponse(std::vector<int>& response)
{
	std::string str;
	for (auto r: response)
	{
		str += std::to_string(r);
	}
	return str;
}
cv::Mat markDetection(const std::vector<Place>& places, cv::Mat image, ClassifierSet& set, std::vector<std::vector<int>>& responses)
{
	cv::Mat detected = image.clone();

	for (int i = 0; i < places.size(); i++)
	{
		int classIndex = set.predictClass(responses[i]);
		auto& place = places[i];

		cv::Scalar color(0.0f, classIndex == 0 ? 255.0f : 0.0f, classIndex == 1 ? 255.0f : 0.0f);
		cv::polylines(detected, std::vector<cv::Point> {
				cv::Point(place.coords[0], place.coords[1]),
				cv::Point(place.coords[2], place.coords[3]),
				cv::Point(place.coords[4], place.coords[5]),
				cv::Point(place.coords[6], place.coords[7])
			},
			true, color
		);

		cv::Point center;
		for (int i = 0; i < 4; i++)
		{
			center.x += place.coords[i * 2];
			center.y += place.coords[i * 2 + 1];
		}

		cv::putText(detected, serializeResponse(responses[i]), center / 4, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar::all(255.0f), 1, CV_AA);
	}

	return detected;
}
