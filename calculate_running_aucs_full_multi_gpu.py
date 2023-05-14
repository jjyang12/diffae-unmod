import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from torchmetrics import AUROC
from torchmetrics import AveragePrecision
from collections import defaultdict

losses = []
labels = []

NUM_GPUS = 4
for i in range(NUM_GPUS):
    with open("mnist_noflips_32actual_noema_fixeddl_randsto_ssim_obj_{}.log".format(i), "r") as fp:
        for line in fp:
            numbers = line.strip().split(" ")[2]
            numbers = numbers.split("|")[:-1]
            for number in numbers:
                losses.append(float(number))
                #losses.append(10 * np.log10(1/float(number)))
losses = np.array(losses)
print("Num examples: ", str(len(losses)))
print(max(losses))
print(min(losses))
#Load test set labels
with open("/home/eprakash/mnist/test_labels.txt", "r") as fp:
    for line in fp:
        labels.append(int(line.strip()))
test_len = len(labels)
print("Num labels", test_len)
print("sum of labels",sum(labels))

#Load SOTA scores
#scores = np.load("scores/final_test_scores.npy")
#scores = np.load("scores/final_deep_features_scores.npy")
#scores = np.load("scores/final_pose_scores.npy")
#scores = np.load("scores/final_velocity_scores.npy")
#Find running labels and SOTA scores for centerframes
curr_labels = []
#curr_scores = []
normal_scores = []
anomaly_scores = []
for n in range(len(losses)):
    label = labels[n]
    if (label == 1):
        anomaly_scores.append(losses[n])
    else:
        normal_scores.append(losses[n])
    label = label != 1
    curr_labels.append(label)
    #curr_scores.append(1 - scores[idx_to_centerframe[n]])

labels = np.array(curr_labels)
#curr_scores = np.array(curr_scores)
print("Anomaly mean score: ", np.mean(anomaly_scores), ", normal mean score: ", np.mean(normal_scores))
print("Percentage of anomalies: ", str(len(anomaly_scores)/len(losses)))
#losses = (losses + curr_scores)/2
#losses = curr_scores

#Calculate AUCs
print(sum(labels))
print(min(labels))
print(losses)
auroc = roc_auc_score(labels, losses)
auprc = average_precision_score(labels, losses)
print(len(losses), len(labels))
print(losses)
print(labels)
print("AUROC: " + str(auroc))
print("AUPRC: " + str(auprc))
