#pragma once

#include "extractor.h"

class HOGExtractor : public Extractor
{
public:
	virtual std::vector<float> extract(cv::Mat image) override
	{
		cv::HOGDescriptor hog;
		std::vector<float> descriptors;
		hog.compute(image, descriptors);

		std::vector<float> features;
		int subdivisions = 10;
		int width = (int) std::sqrt(descriptors.size());
		int stride = width / subdivisions;

		for (int i = 0; i < subdivisions; i++)
		{
			for (int j = 0; j < subdivisions; j++)
			{
				double avg = 0.0;
				for (int x = 0; x < stride; x++)
				{
					for (int y = 0; y < stride; y++)
					{
						int sX = i * stride + x;
						int sY = j * stride + y;

						avg += descriptors[sX * width + sY];
					}
				}
				features.push_back(avg);
			}
		}

		return features;
	}
	virtual bool needsCopy() const override
	{
		return false;
	}
};
