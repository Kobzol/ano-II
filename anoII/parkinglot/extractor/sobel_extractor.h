#pragma once

#include "extractor.h"

class SobelExtractor : public Extractor
{
public:
	std::vector<float> extract(cv::Mat image) override
	{
		cv::cvtColor(image, image, CV_BGR2GRAY);	// convert to grayscale
		cv::medianBlur(image, image, 3);			// blur
		
		std::vector<float> features = {
			this->extractPixels(this->sobel(image), 64)
		};

		/*int width = std::ceil(image.cols / 3.0);
		int height = std::ceil(image.rows / 3.0);

		for (int i = 0; i < image.rows; i += height)
		{
			for (int j = 0; j < image.cols; j += width)
			{
				if (i + height < image.rows && j + width < image.cols)
				{
					features.push_back(this->extractPixels(this->sobel(
						image(cv::Rect(i, j, width, height))
					), 64));
				}
			}
		}*/

		return features;
	}

private:
	cv::Mat sobel(cv::Mat image)
	{
		cv::Mat gradX;
		cv::Sobel(image, gradX, CV_16S, 1, 0);
		cv::convertScaleAbs(gradX, gradX);

		cv::Mat gradY;
		cv::Sobel(image, gradY, CV_16S, 0, 1);
		cv::convertScaleAbs(gradY, gradY);

		cv::Mat sobel;
		cv::addWeighted(gradX, 0.5, gradY, 0.5, 0.0, sobel);

		return sobel;
	}

	float extractPixels(cv::Mat image, uchar threshold)
	{
		float count = 0.0f;
		image.forEach<uchar>([&count, threshold](auto& pixel, const int* pos)
		{
			if (pixel > threshold)
			{
				count++;
			}
		});

		return count;
	}
};
