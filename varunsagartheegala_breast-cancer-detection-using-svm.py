import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os
Data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
Data.head()
Data.describe()
Data.info()
Data.drop(columns=["id","Unnamed: 32"],inplace=True)
Data.head()
# Number of missing values

for i in Data.columns:

    print(f"{i} has {Data[i].isna().sum()} missing values")
# Number of unique values

for i in Data.columns:

    print(f"{i} has {Data[i].nunique()} unique values")
Case_value = []

for i in Data.diagnosis:

    if i == "B":

        Case_value.append(1)

    else:

        Case_value.append(2)

Data["Case_severity"] = Case_value
for index,i in enumerate(Data.corr()):

    counter = 0

    for a in Data.corr():

        if Data.corr().loc[i,a] > 0.95:

            counter = counter + 1

    print(f"{i} is highly correlated to {counter} variables")

            

            
# Finding the pairs of features with high correlation ( > 0.95)
Highly_correlated = []

Highly_correlated_pairs = []

Variable_1 = []

Variable_2 = []

Coef = []

for index,i in enumerate(Data.corr()):

    for a in Data.corr():

        if i != a:

            if Data.corr().loc[i,a] > 0.95:

                print(f"{i} & {a} - {Data.corr().loc[i,a]}")

                Highly_correlated.append(i)

                Highly_correlated.append(a)

                Highly_correlated_pairs.append((i,a))

                Variable_1.append(i)

                Variable_2.append(a)

                Coef.append(Data.corr().loc[i,a])

Highly_correlated_variables = {i for i in Highly_correlated}

            
HighlyCorrelated = pd.DataFrame(list(zip(Variable_1,Variable_2,Coef)),columns=["Variable_1","Variable_2","Coef"])
# Picking the top 10 pairs
Top10 = HighlyCorrelated.sort_values("Coef",ascending=False)[0:10]
Top10.index = np.arange(0,10)
Top10
# Picking the features from the pairs which is highly correlated with the labels
Retained_Variables = []

for index,i in enumerate(Top10.Variable_1):

    if Data.corr().loc[i,"Case_severity"] > Data.corr().loc[Top10.Variable_2[index],"Case_severity"]:

        Retained_Variables.append(i)

    else:

        Retained_Variables.append(Top10.Variable_2[index])
Retained_Variables = [i for i in np.unique(Retained_Variables)]
Retained_Variables
Data = Data[["perimeter_mean","perimeter_worst","radius_mean","radius_worst","diagnosis","Case_severity"]]
Data.corr()
# Data for benign cases

Data[Data["diagnosis"] != "M"].describe()
# Data for Malignant cases

Data[Data["diagnosis"] == "M"].describe()
sns.pairplot(data = Data.drop("Case_severity",axis=1),hue="diagnosis")
plt.figure(figsize = (10,10))

plt.tight_layout(pad=4,w_pad=5,h_pad=8)

plt.subplot(221)

ax1 = sns.violinplot(data=Data,x="diagnosis",y="perimeter_mean")

ax1.set_xticklabels(["Malignant","Benign"])

plt.title("Perimeter Mean")

plt.tight_layout(h_pad=4,w_pad=4)



plt.subplot(222)

ax2 = sns.violinplot(data=Data,x="diagnosis",y="perimeter_worst")

ax2.set_xticklabels(["Malignant","Benign"])

plt.title("Perimeter Worst")

plt.tight_layout(h_pad=4,w_pad=4)



plt.subplot(223)

ax3 = sns.violinplot(data=Data,x="diagnosis",y="radius_mean")

ax3.set_xticklabels(["Malignant","Benign"])

plt.title("Radius Mean")

plt.tight_layout(h_pad=4,w_pad=4)



plt.subplot(224)

ax4 = sns.violinplot(data=Data,x="diagnosis",y="radius_worst")

ax4.set_xticklabels(["Malignant","Benign"])

plt.title("Radius Worst")

plt.tight_layout(h_pad=4,w_pad=4)
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
X_data = Data[["perimeter_mean","perimeter_worst","radius_mean","radius_worst"]]

y_data = Data["Case_severity"]
X_train , X_test , y_train , y_test = train_test_split(X_data,y_data,train_size = 0.75,random_state = 2607)
model = SVC(probability=True,).fit(X_train,y_train)
ModelPredictions = model.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score, classification_report , roc_auc_score, confusion_matrix,roc_curve
accuracy_score(y_test,ModelPredictions)
precision_score(y_test,ModelPredictions)
recall_score(y_test,ModelPredictions)
plt.figure()

ax = plt.subplot()

sns.heatmap(confusion_matrix(y_test,ModelPredictions),annot=True,cbar=False)

ax.set_xlabel("Predicted Values",labelpad = 10)

ax.set_ylabel("True Values")

ax.xaxis.set_ticklabels(["Benign","Malignant"])

ax.yaxis.set_ticklabels(["Benign","Malignant"])

ax.set_title("CONFUSION MATRIX")
print(classification_report(y_test,ModelPredictions))
model_predict_prob = model.predict_proba(X_test)
model_predict_prob = model_predict_prob[:,1]
fpr, tpr , threshold = roc_curve(y_test,model_predict_prob,pos_label=2)
plt.figure(figsize=(10,5))

plt.plot(fpr,tpr,linestyle = "--")

plt.fill_between(fpr,tpr,alpha = 0.1)

plt.title("ROC for SVC")

plt.xlabel("False Positive rate")

plt.ylabel("True Positive rate")

print("ROC_AUC Score for SVC :",roc_auc_score(y_test,model_predict_prob))

from sklearn.model_selection import GridSearchCV
empty = SVC()
params = {"C":[0.001,0.001,0.01,0.1,1,10,100,1000],"kernel":["rbf","linear"],"gamma":[0.001,0.001,0.01,0.1,1,10,100,1000]}
Grid = GridSearchCV(empty,params,refit=True).fit(X_train,y_train)
Grid.best_params_
Grid.best_score_
tuned_model = SVC(C= 1, gamma= 0.001, kernel= 'linear',probability=True).fit(X_train,y_train)
tuned_model_predictions = tuned_model.predict(X_test)
print(classification_report(y_test,tuned_model_predictions))
print(f"Accuracy of tuned model = {accuracy_score(y_test,tuned_model_predictions)}")

print(f"Precision of tuned model = {precision_score(y_test,tuned_model_predictions)}")

print(f"Recall of tuned model = {accuracy_score(y_test,tuned_model_predictions)}")
plt.figure()

ax = plt.subplot()

sns.heatmap(confusion_matrix(y_test,tuned_model_predictions),annot=True,cbar=False)

ax.set_xlabel("Predicted Values",labelpad = 10)

ax.set_ylabel("True Values")

ax.xaxis.set_ticklabels(["Benign","Malignant"])

ax.yaxis.set_ticklabels(["Benign","Malignant"])

ax.set_title("CONFUSION MATRIX for tuned SVC")
tuned_model_predict_prob = tuned_model.predict_proba(X_test)
tuned_model_predict_prob = tuned_model_predict_prob[:,1]
fpr, tpr , threshold = roc_curve(y_test,tuned_model_predict_prob,pos_label=2)
plt.figure(figsize=(10,5))

plt.plot(fpr,tpr,linestyle = "--")

plt.fill_between(fpr,tpr,alpha = 0.1)

plt.title("ROC for SVC")

plt.xlabel("False Positive rate")

plt.ylabel("True Positive rate")

print("ROC_AUC Score for tuned SVC :",roc_auc_score(y_test,tuned_model_predict_prob))
