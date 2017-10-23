#pragma once

#include "extractor.h"

class CannyExtractor : public Extractor
{
public:
	explicit CannyExtractor(uchar threshold) : threshold(threshold)
	{

	}

	std::vector<float> extract(cv::Mat image) override
	{
		const int size = 80;

		cv::Mat resized;
		cv::cvtColor(image, image, CV_BGR2GRAY);
		cv::resize(image, resized, cv::Size(size, size));
		cv::medianBlur(resized, resized, 3);

		cv::Mat sobel = this->sobel(resized);

		cv::Mat visited = cv::Mat::zeros(sobel.size(), CV_8UC1);
		std::vector<int> edges;

		for (int i = 0; i < sobel.rows; i++)
		{
			for (int j = 0; j < sobel.cols; j++)
			{
				int edge = 0;
				std::vector<cv::Point> candidates = { cv::Point(i, j) };

				while (!candidates.empty())
				{
					cv::Point point = candidates[candidates.size() - 1];
					candidates.pop_back();

					if (point.x < 0 || point.x >= visited.rows || point.y < 0 || point.y >= visited.cols) continue;
					if (visited.at<uchar>(point.x, point.y) > 0) continue;
					visited.at<uchar>(point.x, point.y) = 255;
					
					if (sobel.at<uchar>(point.x, point.y) == 255)
					{
						edge++;
					}

					for (int x = -1; x < 2; x++)
					{
						for (int y = -1; y < 2; y++)
						{
							candidates.push_back(cv::Point(i + x, j + y));
						}
					}
				}

				if (edge != 0)
				{
					edges.push_back(edge);
				}
			}
		}

		float avgLength = 0.0f;
		for (int length : edges)
		{
			avgLength += length;
		}
		avgLength /= edges.size();
		
		return { (float)edges.size(), avgLength };

		/*std::vector<float> features;
		features.push_back(this->extractPixels(sobel));
		return features;*/
	}

private:
	cv::Mat sobel(const cv::Mat& image)
	{
		cv::Mat sobel;
		cv::Canny(image, sobel, this->threshold, this->threshold * 3);

		return sobel;
	}

	float extractPixels(const cv::Mat& image)
	{
		float count = 0.0f;
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				if (image.at<uchar>(i, j) == 255)
				{
					count++;
				}
			}
		}

		return count;
	}

	uchar threshold;
};
