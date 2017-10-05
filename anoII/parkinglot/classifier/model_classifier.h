#pragma once

#include "classifier.h"

#include <memory>

template <typename T>
class ModelClassifier: public Classifier
{
	using Classifier::train;

public:
	static std::unique_ptr<ModelClassifier> deserialize(const std::string& path)
	{
		return std::make_unique<ModelClassifier>(T::load<T>(path));
	}

	ModelClassifier()
	{
		this->model = T::create();
	}
	ModelClassifier(cv::Ptr<T> model) : model(model)
	{

	}

	virtual void train(const std::vector<Example>& examples) override
	{
		this->train(*this->model.get(), examples);
	}
	virtual int predict(const std::vector<float>& features) override
	{
		return static_cast<int>(this->model->predict(features));
	}

	virtual void save(const std::string& path) override
	{
		this->model->save(path);
	}
	virtual void load(const std::string& path) override
	{
		this->model = T::load<T>(path);
	}

protected:
	cv::Ptr<T> model;
};
