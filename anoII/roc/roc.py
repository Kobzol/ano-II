import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import glob
import os


def classifier_name(f):
    return os.path.basename(f)[12:-4]


y_test = np.loadtxt('../groundtruth.txt')

rocFiles = [f for f in glob.glob("../probability*.txt")]

rocList = []
for i in range(0, len(rocFiles)):
    predicted = np.loadtxt(rocFiles[i])
    fpr, tpr, _ = roc_curve(y_test, predicted)

    roc = auc(fpr, tpr)
    rocList.append({
        "name": classifier_name(rocFiles[i]),
        "roc": roc,
        "fpr": fpr,
        "tpr": tpr
    })

rocList.sort(key=lambda c: c["roc"], reverse=True)

# Calculate the AUC
for i in range(0,len(rocList)):
    print "{0} ROC AUC: {1}".format(rocList[i]["name"], round(rocList[i]["roc"], 3))

plt.rcParams.update({'font.size': 14})
# Plot of a ROC curve for a specific class
plt.figure()
for i in range(0,len(rocList)):
    plt.plot(rocList[i]["fpr"], rocList[i]["tpr"], label="${0} : {1}$".format(rocList[i]["name"],
                                                                              round(rocList[i]["roc"], 3)),
             linewidth=2.2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
