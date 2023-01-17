import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

plt.rcParams.update({'font.size': 18})
# Load calculated data

patient_mask = np.load("../input/bigimaging-complete/patient_mask.npz")['arr_0']

patient_indices = patient_mask.nonzero()[0]

pred_min = np.load("../input/bigimaging-complete/min_vols.npz")['arr_0'][patient_indices]

pred_max = np.load("../input/bigimaging-complete/max_vols.npz")['arr_0'][patient_indices]



# Load ground truth data

path = os.path.join("..", "input", "second-annual-data-science-bowl", "train.csv")

ground_truth = pd.read_csv(path)

true_min = np.array(ground_truth.Systole)[patient_indices]

true_max = np.array(ground_truth.Diastole)[patient_indices]



min_limit, max_limit = 0, 600

# Clip at [0, 600] ml

pred_min = pred_min.clip(min_limit, max_limit)

pred_max = pred_max.clip(min_limit, max_limit)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16,8])



ax1.plot(true_min, pred_min, "*")

ax1.set_xlim([min_limit, max_limit])

ax1.set_ylim([min_limit, max_limit])

ax1.set_xlabel("True systole (min) volumes")

ax1.set_ylabel("Predicted systole (min) volumes")



ax2.plot(true_max, pred_max, "*")

ax2.set_xlim([min_limit, max_limit])

ax2.set_ylim([min_limit, max_limit])

ax2.set_xlabel("True diastole (max) volumes")

ax2.set_ylabel("Predicted diastole (max) volumes")



plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[15,10])



ax1.hist(pred_min, bins=100, edgecolor="black")

ax1.set_xlabel("Predicted systole (min) volume")

ax1.set_ylabel("Frequency")

ax1.set_ylim([0,60])

ax1.set_xlim([min_limit, max_limit])



ax2.hist(pred_max, bins=100, edgecolor="black")

ax2.set_xlabel("Predicted diastole (max) volume")

ax2.set_ylabel("Frequency")

ax2.set_ylim([0,60])

ax2.set_xlim([min_limit, max_limit])



ax3.hist(true_min, bins=100, edgecolor="black")

ax3.set_xlabel("True systole (min) volume")

ax3.set_ylabel("Frequency")

ax3.set_ylim([0,60])

ax3.set_xlim([min_limit, max_limit])



ax4.hist(true_min, bins=100, edgecolor="black")

ax4.set_xlabel("True diastole (max) volume")

ax4.set_ylabel("Frequency")

ax4.set_ylim([0,60])

ax4.set_xlim([min_limit, max_limit])



plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[20,15])



ax1.plot(patient_indices+1, true_min, '*', label="true")

ax1.plot(patient_indices+1, pred_min, '+', color='r', label="pred")

ax1.set_ylabel('Volume (ml)')

ax1.set_xlabel('Patient id')

ax1.set_title('True vs Predicted Systole')

ax1.legend()



ax2.plot(patient_indices+1, true_max, '*', label="true")

ax2.plot(patient_indices+1, pred_max, '+', color='r', label="pred")

ax2.set_ylabel('Volume (ml)')

ax2.set_xlabel('Patient id')

ax2.set_title('True vs Prediced Diastole')

ax2.legend()



fig.show()
rmse_min = np.sqrt(((true_min - pred_min)**2).mean())

rmse_max = np.sqrt(((true_max - pred_max)**2).mean())

print("MSE Systole:", rmse_min)

print("MSE Diastole:", rmse_max)
corr_max = np.corrcoef(pred_min, true_min)

corr_min = np.corrcoef(pred_max, true_max)



print(corr_max[0,1], "Correlation predicted and true max values")

print(corr_min[0,1], "Correlation predicted and true min values")
def ejection_rate(v_max, v_min): return (v_max - v_min) / v_max



pred_ejections = np.array([ejection_rate(vd, vs) for vd,vs in zip(pred_max, pred_min)])

true_ejections = np.array([ejection_rate(vd, vs) for vd,vs in zip(true_max, true_min)])



rmse_ejection = np.sqrt(((true_ejections- pred_ejections)**2).mean())

print("RMSE Ejection Rate:", rmse_ejection)
fig, ax = plt.subplots(1, 1, figsize=[20,5])



ax.plot(patient_indices, true_ejections, '*', label="true")

ax.plot(patient_indices, pred_ejections, '+', color='r', label="pred")

ax.set_title('Ejection rate, true vs pred', fontsize=16)

ax.set_ylabel('Ejection rate')

ax.set_xlabel('Patient')

ax.legend()



fig.show()