#pragma once

#include "extractor.h"

class HistogramExtractor : public Extractor
{
public:
	std::vector<float> extract(cv::Mat image) override
	{
		const int size = 80;

		cv::Mat resized;
		cv::resize(image, resized, cv::Size(size, size));
		cv::medianBlur(resized, resized, 5);
		cv::cvtColor(resized, resized, CV_BGR2HSV);

		float sum = 0.0f;
		for (int x = 0; x < resized.rows - 1; x++)
		{
			uchar a = resized.at<cv::Vec3b>(x, size / 2)[0];
			uchar b = resized.at<cv::Vec3b>(x + 1, size / 2)[0];

			sum += (float) std::pow(b - a, 2);
		}

		return { sum };
	}

	void histogram()
	{
		/*std::vector<float> features;
		float totalHue = 0.0f;

		int divisions = 20;
		int subSize = size / divisions;

		for (int x = 0; x < resized.rows; x += (resized.rows / divisions))
		{
		for (int y = 0; y < resized.cols; y += (resized.cols / divisions))
		{
		auto submat = resized(cv::Rect(x, y, subSize, subSize)).clone();
		cv::cvtColor(submat, submat, CV_BGR2HSV);
		float avg = 0.0f;

		for (int i = 0; i < submat.rows; i++)
		{
		for (int j = 0; j < submat.cols; j++)
		{
		avg += submat.at<cv::Vec3b>(i, j)[0];
		}
		}

		float avgHue = avg / (submat.rows * submat.cols);
		features.push_back(avgHue);
		totalHue += avgHue;
		}
		}

		totalHue /= features.size();
		float sum = 0.0f;
		for (float f : features)
		{
		sum += std::pow(f - totalHue, 2);
		}*/
	}
};
