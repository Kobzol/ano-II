#pragma once

#include "classifier.h"
#include <dlib/dnn/core.h>
#include <dlib/dnn/loss.h>
#include <dlib/dnn/layers.h>
#include <dlib/dnn/input.h>

#include <memory>

// Canny - 32
// BGR - 5 (20/28, dropout/750fc, 15/5/5con)

/*
male/female
83.9
using net_inner = dlib::relu<dlib::fc<750,
dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::bn_con<dlib::con<50, 3, 3, 1, 1,
dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::bn_con<dlib::con<30, 3, 3, 1, 1,
dlib::input<dlib::matrix<pixel_type>>
>>>>>>>>>>;
*/

/* BEST
using net_learn = dlib::loss_multiclass_log<
dlib::fc<1,
dlib::dropout<net_inner>>>;
using net_predict = dlib::loss_multiclass_log<
dlib::fc<1,
dlib::multiply<net_inner>>>;
*/

using pixel_type = dlib::bgr_pixel;
using net_inner = dlib::relu<dlib::fc<750,
	dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<15, 5, 5, 1, 1,
	dlib::input<dlib::matrix<pixel_type>>
	>>>>>;

// CUSTOM
using net_learn = dlib::loss_multiclass_log<
	dlib::fc<2,
	dlib::dropout<net_inner>>>;
using net_predict = dlib::softmax<
	dlib::fc<2,
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
	virtual float predict(cv::Mat image) override;

	virtual bool supportsFeatures() override;

	virtual void save(const std::string& path) override;
	virtual void load(const std::string& path) override;

private:
	net_learn learner;
	net_predict predictor;
};
