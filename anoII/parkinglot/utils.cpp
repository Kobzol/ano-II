#include "utils.h"

#include <fstream>
#include <omp.h>

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

std::vector<cv::Mat> expandDataset(const std::vector<cv::Mat>& frames)
{
	std::vector<cv::Mat> images;
	for (auto& frame : frames)
	{
		images.push_back(frame);
		
		/*cv::Mat tmp;
		cv::flip(frame, tmp, 1);
		images.push_back(tmp);

		/*for (int i = 0; i < 3; i++)
		{
			cv::Mat rotated;
			cv::rotate(frame, rotated, i);
			images.push_back(rotated);
		}

		cv::Mat rotated;
		cv::rotate(frame, rotated, cv::RotateFlags::ROTATE_180);
		images.push_back(rotated);*/
	}

	return images;
}

void appendExamples(const std::vector<std::string>& paths, const std::vector<Place>& places,
	const std::vector<std::unique_ptr<Extractor>>& extractors, int classIndex,
	std::vector<Example>& examples)
{
	size_t size = paths.size();
	std::vector<std::vector<Example>> results(omp_get_max_threads(), {});

#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		auto& path = paths[i];
		cv::Mat trainingImage = cv::imread(path, cv::IMREAD_COLOR);
		std::vector<cv::Mat> frames = extractParkingPlaces(places, trainingImage);
		frames = expandDataset(frames);
		
		auto& vec = results[omp_get_thread_num()];
		for (auto& frame: frames)
		{
			vec.push_back(Example::create(extractors, frame, classIndex));
		}
	}

	for (auto& sub : results)
	{
		examples.insert(examples.end(), sub.begin(), sub.end());
	}
}
void appendDirectExamples(const std::vector<cv::Mat>& places, const std::vector<std::unique_ptr<Extractor>>& extractors, int classIndex, std::vector<Example>& examples)
{
	size_t size = places.size();
	std::vector<std::vector<Example>> results(omp_get_max_threads(), {});

#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		std::vector<cv::Mat> images = { places[i] };
		images = expandDataset(images);
		for (auto& img : images)
		{
			results[omp_get_thread_num()].push_back(Example::create(extractors, img, classIndex));
		}
	}

	for (auto& sub : results)
	{
		examples.insert(examples.end(), sub.begin(), sub.end());
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

static std::string serializeResponse(std::vector<float>& response)
{
	std::string str;
	for (auto r: response)
	{
		str += std::to_string(r);
	}
	return str;
}
cv::Mat markDetection(const std::vector<Place>& places, cv::Mat image, ClassifierSet& set, std::vector<std::vector<float>>& responses)
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

void writeImages(const std::vector<cv::Mat>& images, const std::string& path)
{
	std::ofstream fs(path, std::ios::binary);
	size_t count = images.size();
	fs.write((const char*) &count, sizeof(count));

	for (auto& img : images)
	{
		writeMatBinary(fs, img);
	}
	fs.flush();
}

std::vector<cv::Mat> readImages(const std::string& path)
{
	std::ifstream fs(path, std::ios::binary);
	size_t count;
	fs.read((char*) &count, sizeof(count));

	std::vector<cv::Mat> images(count);

	for (auto& img : images)
	{
		readMatBinary(fs, img);
	}

	return images;
}

bool writeMatBinary(std::ofstream& ofs, const cv::Mat& mat)
{
	if (!ofs.is_open())
	{
		return false;
	}
	if (mat.empty())
	{
		int s = 0;
		ofs.write((const char*)(&s), sizeof(int));
		return true;
	}
	int type = mat.type();
	ofs.write((const char*)(&mat.rows), sizeof(int));
	ofs.write((const char*)(&mat.cols), sizeof(int));
	ofs.write((const char*)(&type), sizeof(int));
	ofs.write((const char*)(mat.data), mat.elemSize() * mat.total());

	return true;
}

bool readMatBinary(std::ifstream& ifs, cv::Mat& mat)
{
	if (!ifs.is_open())
	{
		return false;
	}

	int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	if (rows == 0)
	{
		return true;
	}
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));

	mat.release();
	mat.create(rows, cols, type);
	ifs.read((char*)(mat.data), mat.elemSize() * mat.total());

	return true;
}

bool fileExists(const std::string& name)
{
	std::ifstream fs(name);
	return fs.good();
}
