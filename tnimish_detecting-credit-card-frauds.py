import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)

Data=W_Data

Data.sample(frac=0.1).head(n=5)
Data.describe()
Positives=W_Data[W_Data['Class']==1]

Negatives=W_Data[W_Data['Class']==0]
print((len(Positives)/len(W_Data))*100,"%")
sns.kdeplot(Positives['Amount'],shade=True,color="red")