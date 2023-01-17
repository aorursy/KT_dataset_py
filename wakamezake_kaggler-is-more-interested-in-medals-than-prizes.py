import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



warnings.filterwarnings('ignore')

input_path = Path("../input")
comp_path = input_path / "Competitions.csv"

comp = pd.read_csv(comp_path)
# Use only columns related to medal and prize 

use_cols = ["Title", "EnabledDate", "DeadlineDate", "TotalCompetitors", "TotalTeams", "NumPrizes", "RewardQuantity", "RewardType", "CanQualifyTiers"]

data = comp[use_cols]
data[data["CanQualifyTiers"] == True].head()
data["is_prize"] = data["NumPrizes"].apply(lambda x: False if x == 0 else True)
# apply datetime type

data["EnabledDate"] = pd.to_datetime(data["EnabledDate"])

data["DeadlineDate"] = pd.to_datetime(data["DeadlineDate"])

# from 2018-01-01 to now

query = (data["EnabledDate"] >= "2018-01-01")

recently_held_comp = data[query]
# Reference

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(y_true, y_pred, classes,

                          y_label='True label',

                          x_label='Predicted label',

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel=y_label,

           xlabel=x_label)



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
cm = confusion_matrix(data["CanQualifyTiers"].astype(int).values,

                      data["is_prize"].astype(int).values)

(_, _counts), (_, _) = cm

"{:.2f} %".format((_counts / np.sum(cm)) * 100) 
# combination CanQualifyTiers and is_prize

plot_confusion_matrix(y_true=data["CanQualifyTiers"].astype(int).values,

                      y_pred=data["is_prize"].astype(int).values, 

                      classes=np.array(["False", "True"]),

                     y_label="CanQualifyTiers", x_label="is_prize")
query = ((data["is_prize"] == True) & (data["CanQualifyTiers"] == True))

sns.distplot(data[query]["TotalCompetitors"].values, axlabel="TotalCompetitors")
# Basic statistics

data[query]["TotalCompetitors"].describe()
query = ((data["is_prize"] == True) & (data["CanQualifyTiers"] == False))

sns.distplot(data[query]["TotalCompetitors"].values, axlabel="TotalCompetitors")
# Basic statistics

data[query]["TotalCompetitors"].describe()
query = ((data["is_prize"] == False) & (data["CanQualifyTiers"] == True))

sns.distplot(data[query]["TotalCompetitors"].values, axlabel="TotalCompetitors")
# Basic statistics

data[query]["TotalCompetitors"].describe()
query = ((data["is_prize"] == False) & (data["CanQualifyTiers"] == False))

sns.distplot(data[query]["TotalCompetitors"].values, axlabel="TotalCompetitors")
# Basic statistics

data[query]["TotalCompetitors"].describe()