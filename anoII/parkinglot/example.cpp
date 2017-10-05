#include "example.h"

Example Example::create(const std::vector<std::unique_ptr<Extractor>>& extractors, cv::Mat place, int classIndex)
{
	Example example(classIndex, place);
	for (auto& extractor : extractors)
	{
		auto features = extractor->extract(place);
		example.features.insert(example.features.end(), features.begin(), features.end());
	}

	return example;
}

Example::Example(int classIndex, cv::Mat image): classIndex(classIndex), image(image)
{

}
