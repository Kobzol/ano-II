#include <opencv2/opencv.hpp>

std::vector<cv::Rect> detectFaces(cv::CascadeClassifier& classifier, cv::Mat& frame)
{
	std::vector<cv::Rect> faces;
	classifier.detectMultiScale(frame, faces,
		1.2,	// kolikrat se zvetsuje sliding window
		3		// 3 - minimalne 4 obdelniky musi byt na jednom miste, aby byla urcena detekce
	);
	return faces;
}

void cviko1()
{
	cv::CascadeClassifier classifier;
	cv::namedWindow("Detect");

	cv::VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);

	if (!classifier.load("../opencv/etc/lbpcascades/lbpcascade_frontalface.xml"))
	{
		exit(1);
	}

	while (true)
	{
		cv::Mat frame;
		cap >> frame;

		auto faces = detectFaces(classifier, frame);
		for (auto& face : faces)
		{
			cv::rectangle(frame, face, cv::Scalar(0.0f, 0.0f, 255.0f));
		}

		cv::imshow("Detect", frame);
		cv::waitKey(2);
	}
}
