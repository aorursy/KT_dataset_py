import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
# kaggle path

input_dir = "/kaggle/input"

csv_file = "heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
# load heart failure data

hf_data = pd.read_csv(os.path.join(input_dir,csv_file))

hf_data
# check the basic information of the loaded data frame

hf_data.info()
# the responses are quite unbalanced

response_counts = hf_data["DEATH_EVENT"].groupby(hf_data["DEATH_EVENT"]).count()

response_counts.plot.pie(legend=True)

response_counts.div(response_counts.sum())
# set train:val about 3:1

np.random.seed(71)

n = hf_data.shape[0] # sample size

ind = np.arange(n)

np.random.shuffle(ind)

ind_train = ind[:int(n*0.75)]

ind_val = ind[int(n*0.75):]

train_data = hf_data.iloc[ind_train]

val_data = hf_data.iloc[ind_val]
y_train = train_data["DEATH_EVENT"].to_numpy()

y_val = val_data["DEATH_EVENT"].to_numpy()
# continuous variables

x_train_continuous = train_data[["creatinine_phosphokinase","ejection_fraction",

                             "platelets","serum_creatinine","serum_sodium","time"]].to_numpy()

x_val_continuous = val_data[["creatinine_phosphokinase","ejection_fraction",

                             "platelets","serum_creatinine","serum_sodium","time"]].to_numpy()
# binary

x_train_categorical = train_data[["anaemia","diabetes","high_blood_pressure","sex","smoking"]].to_numpy()

x_val_categorical = val_data[["anaemia","diabetes","high_blood_pressure","sex","smoking"]].to_numpy()
# fit a standard scaler using continuous predictors

scaler = StandardScaler()

x_train_continuous_normalized = scaler.fit_transform(x_train_continuous)

x_val_continuous_normalized = scaler.transform(x_val_continuous)
# concatenate the data matrix

x_train = np.concatenate([x_train_continuous_normalized,x_train_categorical],axis=1)

x_val = np.concatenate([x_val_continuous_normalized,x_val_categorical],axis=1)
clf_lr = LogisticRegression(random_state=71)

clf_lr.fit(x_train,y_train)

train_acc = clf_lr.score(x_train,y_train)

val_acc = clf_lr.score(x_val,y_val)

print("The training accuracy is: {}".format(train_acc))

print("The validation accuracy is: {}".format(val_acc))
# data imbalance

np.bincount(y_train).max()/len(y_train),np.bincount(y_val).max()/len(y_val)
# check the f1 scores

y_pred_train = clf_lr.predict(x_train)

y_pred_val = clf_lr.predict(x_val)

train_f1 = f1_score(y_train,y_pred_train)

val_f1 = f1_score(y_val,y_pred_val)

print("The training F1-score is: {}".format(train_f1))

print("The validation F1-score is: {}".format(val_f1))
# add class weights to the classifier

clf_lr_balanced = LogisticRegression(class_weight="balanced",random_state=71)

clf_lr_balanced.fit(x_train,y_train)

train_acc = clf_lr_balanced.score(x_train,y_train)

val_acc = clf_lr_balanced.score(x_val,y_val)

print("The training accuracy is: {}".format(train_acc))

print("The validation accuracy is: {}".format(val_acc))

# check the f1 scores

y_pred_train = clf_lr_balanced.predict(x_train)

y_pred_val = clf_lr_balanced.predict(x_val)

train_f1 = f1_score(y_train,y_pred_train)

val_f1 = f1_score(y_val,y_pred_val)

print("The training F1-score is: {}".format(train_f1))

print("The validation F1-score is: {}".format(val_f1))
clf_rf = RandomForestClassifier(criterion="entropy",random_state=71)

clf_rf.fit(x_train,y_train)

train_acc = clf_rf.score(x_train,y_train)

val_acc = clf_rf.score(x_val,y_val)

print("The training accuracy is: {}".format(train_acc))

print("The validation accuracy is: {}".format(val_acc))
y_pred_train = clf_rf.predict(x_train)

y_pred_val = clf_rf.predict(x_val)

train_f1 = f1_score(y_train,y_pred_train)

val_f1 = f1_score(y_val,y_pred_val)

print("The training F1-score is: {}".format(train_f1))

print("The validation F1-score is: {}".format(val_f1))
# the random forest classifier is clearly overfitting the training set

# increase the number of estimators

clf_rf_300 = RandomForestClassifier(n_estimators=300,criterion="entropy",random_state=71)

clf_rf_300.fit(x_train,y_train)

train_acc = clf_rf_300.score(x_train,y_train)

val_acc = clf_rf_300.score(x_val,y_val)

print("The training accuracy is: {}".format(train_acc))

print("The validation accuracy is: {}".format(val_acc))

y_pred_train = clf_rf_300.predict(x_train)

y_pred_val = clf_rf_300.predict(x_val)

train_f1 = f1_score(y_train,y_pred_train)

val_f1 = f1_score(y_val,y_pred_val)

print("The training F1-score is: {}".format(train_f1))

print("The validation F1-score is: {}".format(val_f1))
# decrease the maximum depth

clf_rf_shallow = RandomForestClassifier(n_estimators=300,max_depth=5,criterion="entropy",random_state=71)

clf_rf_shallow.fit(x_train,y_train)

train_acc = clf_rf_shallow.score(x_train,y_train)

val_acc = clf_rf_shallow.score(x_val,y_val)

print("The training accuracy is: {}".format(train_acc))

print("The validation accuracy is: {}".format(val_acc))

y_pred_train = clf_rf_shallow.predict(x_train)

y_pred_val = clf_rf_shallow.predict(x_val)

train_f1 = f1_score(y_train,y_pred_train)

val_f1 = f1_score(y_val,y_pred_val)

print("The training F1-score is: {}".format(train_f1))

print("The validation F1-score is: {}".format(val_f1))