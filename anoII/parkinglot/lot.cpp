#include <fstream>
#include <iomanip>
#include <vector>

#include "place.h"
#include "utils.h"
#include "setup.h"

// Sobel(30) - 0.974, 22
// Sobel(30) + SURF(400) - 0.975, 21 (SVM Linear)
// Sobel(30) + Histogram - 0.975, 21
// DNN(36, 0.2, 0.2) + Canny(30) - 0.977, 19
// Dlib - (5+8 => 2)

//#define TRAIN
#define CLASSIFIER_PATH "classifier"
#define PKLOT_04_BIN_0 "pklot04_0.bin"
#define PKLOT_04_BIN_1 "pklot04_1.bin"
#define PKLOT_05_BIN_0 "pklot05_0.bin"
#define PKLOT_05_BIN_1 "pklot05_1.bin"
#define PKLOT_PUC_BIN_0 "pklotPUC_0.bin"
#define PKLOT_PUC_BIN_1 "pklotPUC_1.bin"
#define SIMKANIC_BIN_0 "simkanic_0.bin"
#define SIMKANIC_BIN_1 "simkanic_1.bin"
#define SHUFFLE_EXAMPLES
#define VISUAL_TEST (true)


void parkinglot()
{
	std::vector<Place> places = loadGeometry("geometry/strecha1_map.txt");
	std::vector<int> truth = loadGroundTruth("groundtruth.txt");

	auto extractors = createExtractors();
	auto classifiers = createClassifiers();

#ifdef TRAIN
	std::vector<Example> examples;
	createExamples(places, extractors,
		"../train_images/full/full.txt",
		"../train_images/free/free.txt",
		examples
	);
	createExamplesPklot(extractors,
		"../UFPR05/full.txt",
		"../UFPR05/empty.txt",
		PKLOT_05_BIN_1,
		PKLOT_05_BIN_0,
		examples
	);
	createExamplesPklot(extractors,
		"../UFPR04/full.txt",
		"../UFPR04/empty.txt",
		PKLOT_04_BIN_1,
		PKLOT_04_BIN_0,
		examples
	);
	createExamplesPklot(extractors,
		"../train_images_2/full.txt",
		"../train_images_2/empty.txt",
		SIMKANIC_BIN_1,
		SIMKANIC_BIN_0,
		examples
	);
	/*createExamplesPklot(extractors,
		"../PUC/full.txt",
		"../PUC/empty.txt",
		PKLOT_PUC_BIN_1,
		PKLOT_PUC_BIN_0,
		examples
	);*/

	std::cerr << "Loaded " << examples.size() << " images" << std::endl;

#ifdef SHUFFLE_EXAMPLES
	std::random_shuffle(examples.begin(), examples.end());
#endif

	classifiers.train(examples);
	classifiers.save(CLASSIFIER_PATH);
#else
	classifiers.load(CLASSIFIER_PATH);
#endif
	auto evaluations = testClassifiers(places, extractors, classifiers, truth, "../test_images/test.txt", VISUAL_TEST);

	for (int i = 0; i < evaluations.size() - 1; i++)
	{
		std::cout << std::setw(16) << std::left << classifiers.classifiers[i]->getName() + ":" << evaluations[i].evaluate() << std::endl;
	}
	std::cout << std::setw(16) << std::left << "Combined:" << evaluations[evaluations.size() - 1].evaluate() << std::endl;
	std::cout << std::endl;

	getchar();
}
