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

	explicit ModelClassifier(std::string name): Classifier(name)
	{
		this->model = T::create();
	}
	explicit ModelClassifier(cv::Ptr<T> model) : model(model)
	{

	}

	virtual void train(const std::vector<Example>& examples) override
	{
		this->train(*this->model.get(), examples);
	}
	virtual float predict(const std::vector<float>& features) override
	{
		return static_cast<float>(this->model->predict(features));
	}

	virtual void save(const std::string& path) override
	{
		this->model->save(path);
	}
	virtual void load(const std::string& path) override
	{
		this->model = cv::Algorithm::load<T>(path);
	}

	cv::Ptr<T>& getModel()
	{
		return this->model;
	}

protected:
	cv::Ptr<T> model;
};
