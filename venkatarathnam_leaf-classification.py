import numpy as np

import pandas as pd

import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
X = train_data.iloc[:,2:]

y = train_data.iloc[:,1]
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.25, random_state=42)

print ("Length of Train: {0}".format(len(train_x)))

print ("Length of Validation: {0}".format(len(val_x)))
clf = RandomForestClassifier(n_estimators = 40)

clf.fit(train_x, train_y)
preds = clf.predict(val_x)

print (classification_report(val_y, preds))
test_x = test_data.iloc[:,1:]

test_id = test_data.iloc[:,0]

test_pred = clf.predict(test_x)
pred_df = pd.read_csv("../input/sample_submission.csv")

for index, pred in enumerate(test_pred):

    #pred_df.loc[pred_df['id'] == test_id[index],1:] = 0.0

    pred_df.loc[pred_df['id'] == test_id[index],pred] = 1.0
pred_df
pred_df.to_csv("../working/prediction.csv", index=False)