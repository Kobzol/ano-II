#pragma once

#include "classifier.h"
#include <dlib/dnn/core.h>
#include <dlib/dnn/loss.h>
#include <dlib/dnn/layers.h>
#include <dlib/dnn/input.h>

#include <memory>

// Canny - 32
// BGR - 10 (20/28, dropout/750fc, 15/5/5con)

using pixel_type = dlib::bgr_pixel;
using net_inner = dlib::relu<dlib::fc<750,
	dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<15, 5, 5, 1, 1,
	dlib::input<dlib::matrix<pixel_type>>
	>>>>>;

// CUSTOM
using net_learn = dlib::loss_binary_log<
	dlib::fc<1,
	dlib::dropout<net_inner>>>;
using net_predict = dlib::loss_binary_log<
	dlib::fc<1,
	dlib::multiply<net_inner>>>;

// PAPER
/*using net_type = dlib::loss_binary_log<
	dlib::fc<1,
	dlib::relu<dlib::fc<10,
	dlib::relu<dlib::fc<20,
	dlib::relu<dlib::fc<30,
	dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::bn_con<dlib::con<30, 5, 5, 1, 1,
	dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::bn_con<dlib::con<20, 5, 5, 1, 1,
	dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::bn_con<dlib::con<10, 5, 5, 1, 1,
	dlib::input<dlib::matrix<pixel_type>>
	>>>>>>>>>>>>>>>>>>>>;*/

class DlibClassifier : public Classifier
{
public:
	DlibClassifier(std::string name);

	virtual void train(const std::vector<Example>& examples) override;
	virtual int predict(cv::Mat image) override;

	virtual bool supportsFeatures() override;

	virtual void save(const std::string& path) override;
	virtual void load(const std::string& path) override;

private:
	net_learn learner;
	net_predict predictor;
};
