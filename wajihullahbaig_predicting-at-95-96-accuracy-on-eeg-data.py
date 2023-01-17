# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib inline
print(os.listdir("../input"))

inputData = pd.read_csv(r"../input/eeg_clean.csv");
print(inputData.dtypes)
print(inputData.columns)
print("Data shape:",inputData.shape)
print(inputData.head())
print(inputData.describe())
print(inputData.info())
# Check for any nulls
print(inputData.isnull().sum())


inputData['eye']=inputData["eye"].astype('category')
inputData["eye"] = inputData["eye"].cat.codes
from sklearn.model_selection import train_test_split
splitRatio = 0.2
train , test = train_test_split(inputData,test_size = splitRatio,random_state = 123,shuffle=True)

plt.figure(figsize=(12,6))
plt.subplot(121)
train["eye"].value_counts().plot.pie(labels = ["1-open","0-closed"],
                                              autopct = "%1.0f%%",
                                              shadow = True,explode=[0,.1])
plt.title("proportion of target class in train data")
plt.ylabel("")
plt.subplot(122)
test["eye"].value_counts().plot.pie(labels = ["1-open","0-closed"],
                                             autopct = "%1.0f%%",
                                             shadow = True,explode=[0,.1])
plt.title("proportion of target class in test data")
plt.ylabel("")
plt.show()


print ("***********************")
print ("RAW DATA AND PREDICTION")
print ("***********************")


train_X = train[[x for x in train.columns if x not in ["eye"]]]
train_Y = train[["eye"]]
test_X  = test[[x for x in test.columns if x not in ["eye"]]]
test_Y  = test[["eye"]]


from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc


classifier =  XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=1,
 gamma=0.015,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 random_state=777,
 )
# we need to use values.ravel to ensure the labels are sent to classifier correctly
classifier.fit(train_X,train_Y.values.ravel())
print(classifier)

predictions = classifier.predict(test_X)
print ("\naccuracy_score :",accuracy_score(test_Y,predictions))
print ("\nclassification report :\n",(classification_report(test_Y,predictions)))
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(test_Y,predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)
predicting_probabilites = classifier.predict_proba(test_X)[:,1]
fpr,tpr,thresholds = roc_curve(test_Y,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)    
dataframe = pd.DataFrame(classifier.feature_importances_,train_X.columns).reset_index()
dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
dataframe = dataframe.sort_values(by="coefficients",ascending = False)
plt.subplot(223)
ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
plt.title("FEATURE IMPORTANCES",fontsize =20)
for i,j in enumerate(dataframe["coefficients"]):
    ax.text(.011,i,j,weight = "bold")
plt.show()

